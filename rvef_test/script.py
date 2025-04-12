#!/usr/bin/env python3
"""
基於輸入遮罩的加權增強範例
僅利用預先計算的遮罩 (numpy 檔) 改良影像資料，
以達到局部區域增強的效果。若 alpha 設為 1 則表示不加權。
"""

import os
import math
import time
import random
import argparse

import numpy as np
import pandas as pd
import cv2
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import sklearn.metrics
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 全域參數與環境變數設定
# -----------------------------
CSV_FILE    = "/a4c-video-dir/FileList.csv"
VIDEOS_ROOT = "/a4c-video-dir/Videos"
MASK_DIR    = "precomputed_masks"     

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['NCCL_P2P_LEVEL'] = "NVL"

# -----------------------------
# 工具函式定義
# -----------------------------
def loadvideo(filename: str) -> np.ndarray:
    """讀取 avi 檔案，並轉換為 shape=(3, frames, H, W) 的 uint8 陣列。"""
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    cap = cv2.VideoCapture(filename)
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 轉換 BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame)
    cap.release()
    if len(frames_list) == 0:
        return np.zeros((3, 0, 0, 0), dtype=np.uint8)
    arr = np.stack(frames_list, axis=0)
    arr = arr.transpose((3, 0, 1, 2))  # (3, F, H, W)
    return arr

def get_mean_and_std(dataset, samples=128, batch_size=8, num_workers=4):
    from torch.utils.data import Subset
    n = len(dataset)
    if samples is not None and n > samples:
        idx = random.sample(range(n), samples)
        dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    s1 = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    count = 0
    for (x, *_) in tqdm.tqdm(loader):
        B, C, F_, H_, W_ = x.shape
        x = x.float().reshape(C, -1)
        count_ = x.shape[1]
        s1 += x.sum(dim=1).numpy()
        s2 += (x**2).sum(dim=1).numpy()
        count += count_
    mean = s1 / count
    var = (s2 / count) - mean**2
    std = np.sqrt(np.clip(var, a_min=1e-12, a_max=None))
    return mean.astype(np.float32), std.astype(np.float32)

def bootstrap(a, b, func, samples=10000):
    a = np.array(a)
    b = np.array(b)
    boots = []
    for _ in range(samples):
        idx = np.random.choice(len(a), len(a))
        boots.append(func(a[idx], b[idx]))
    boots.sort()
    val = func(a, b)
    l5 = boots[ round(0.05*len(boots)) ]
    u95 = boots[ round(0.95*len(boots)) ]
    return (val, l5, u95)

def latexify():
    import matplotlib
    params = {
        'backend':'pdf',
        'axes.titlesize':8,
        'axes.labelsize':8,
        'font.size':8,
        'legend.fontsize':8,
        'xtick.labelsize':8,
        'ytick.labelsize':8,
        'font.family':'DejaVu Serif',
        'font.serif':'Computer Modern',
    }
    matplotlib.rcParams.update(params)

# -----------------------------
# 資料集定義：EchoDataset
# -----------------------------
class EchoDataset(Dataset):
    """
    基於 CSV 檔案讀取影片資料，並根據輸入遮罩進行加權增強。
    此處不再使用獨立的 with_mask 參數，
    只以 alpha 決定是否執行遮罩加權（alpha<1 表示執行加權）。
    
    參數:
      - alpha: 控制加權程度，alpha = 1 表示不加權，alpha < 1 表示加權；
      - mask_dir: 遮罩所在的目錄，遮罩檔名格式為 {video_basename}_masks.npy，
                  每個遮罩檔案形狀一般為 (frames, H, W)，值域為 0 或 255。
    """
    def __init__(self, csv_file: str, videos_dir: str,
                 split="TRAIN", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 alpha=1.0,
                 mask_dir=None):
        super().__init__()
        self.csv_file   = csv_file
        self.videos_dir = videos_dir
        self.split      = split.upper()
        if not isinstance(target_type, list):
            target_type= [target_type]
        self.target_type= target_type
        self.mean = mean
        self.std  = std
        self.length = length
        self.period = period
        self.max_length = max_length
        self.clips = clips
        self.pad = pad

        self.alpha = alpha
        self.mask_dir = mask_dir

        df = pd.read_csv(csv_file)
        df["Split"] = df["Split"].str.upper()
        if self.split != "ALL":
            df = df[df["Split"] == self.split].reset_index(drop=True)
        self.df = df
        self.header = df.columns.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = str(row["FileName"])
        if not fn.lower().endswith(".avi"):
            fn += ".avi"
        vpath = os.path.join(self.videos_dir, fn)

        vid = loadvideo(vpath).astype(np.float32)  # shape=(3, F, H, W)
        c, f, h, w = vid.shape

        # 正規化
        if isinstance(self.mean, (float, int)):
            vid -= self.mean
        else:
            vid -= self.mean.reshape((3, 1, 1, 1))
        if isinstance(self.std, (float, int)):
            vid /= self.std
        else:
            vid /= self.std.reshape((3, 1, 1, 1))

        # 設定 clip 長度
        if self.length is None:
            length = f // self.period
        else:
            length = self.length
        if self.max_length is not None:
            length = min(length, self.max_length)

        if f < length * self.period:
            need = length * self.period - f
            z = np.zeros((c, need, h, w), dtype=vid.dtype)
            vid = np.concatenate((vid, z), axis=1)
            c, f, h, w = vid.shape

        # -----------------------------
        # 遮罩讀取與處理：僅當 mask_dir 存在且 alpha < 1 時啟動
        # -----------------------------
        masks_np = None
        if (self.mask_dir is not None) and (self.alpha < 1.0):
            base = os.path.splitext(os.path.basename(fn))[0]
            mask_path = os.path.join(self.mask_dir, f"{base}_masks.npy")
            if os.path.exists(mask_path):
                masks_np = np.load(mask_path)  
                new_masks = []
                for i in range(masks_np.shape[0]):
                    # 將遮罩調整到 112×112，可根據需求調整
                    mask_resized = cv2.resize(masks_np[i], (112, 112), interpolation=cv2.INTER_NEAREST)
                    new_masks.append(mask_resized)
                masks_np = np.stack(new_masks, axis=0)
            else:
                masks_np = None

        # -----------------------------
        # 選取 clip 起始點
        # -----------------------------
        if self.clips == "all":
            starts = np.arange(f - (length - 1) * self.period)
        else:
            possible = f - (length - 1) * self.period
            if possible < 1:
                possible = 1
            st_ = np.random.choice(possible, int(self.clips))
            starts = np.atleast_1d(st_)

        # 取得 target 數值
        target = []
        for t in self.target_type:
            if t.upper() == "FILENAME":
                target.append(fn)
            else:
                if t in row:
                    target.append(np.float32(row[t]))
                else:
                    target.append(np.float32(0))
        if len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        out_clips = []
        for s in starts:
            idxs = s + self.period * np.arange(length)
            # 取出對應幀的 clip，shape=(3, length, h, w)
            clip_arr = vid[:, idxs, :, :]
            if masks_np is not None:
                # 每幀遮罩直接用二值資訊加權（alpha 控制加權強度）
                for i, fidx in enumerate(idxs):
                    if fidx < masks_np.shape[0]:
                        lambda_factor = (1 - self.alpha) * 0.5
                        mask_norm = masks_np[fidx].astype(np.float32) / 255.0
                        mask_3d = np.stack([mask_norm]*3, axis=0)
                        clip_arr[:, i, :, :] += lambda_factor * mask_3d
            out_clips.append(clip_arr)

        if len(out_clips) == 1:
            out_ = out_clips[0]
        else:
            out_ = np.stack(out_clips, axis=0)

        # -----------------------------
        # 隨機 Padding 與裁切（資料增強）
        # -----------------------------
        if self.pad is not None:
            if out_.ndim == 4:
                c_, l_, hh_, ww_ = out_.shape
                tmp = np.zeros((c_, l_, hh_ + 2*self.pad, ww_ + 2*self.pad), out_.dtype)
                tmp[:, :, self.pad:self.pad+hh_, self.pad:self.pad+ww_] = out_
                i = np.random.randint(0, 2*self.pad)
                j = np.random.randint(0, 2*self.pad)
                out_ = tmp[:, :, i:i+hh_, j:j+ww_]
            else:
                nclip, c_, l_, hh_, ww_ = out_.shape
                i = np.random.randint(0, 2*self.pad)
                j = np.random.randint(0, 2*self.pad)
                tmp_list = []
                for cidx in range(nclip):
                    arr = out_[cidx]
                    ttmp = np.zeros((c_, l_, hh_+2*self.pad, ww_+2*self.pad), arr.dtype)
                    ttmp[:, :, self.pad:self.pad+hh_, self.pad:self.pad+ww_] = arr
                    patch = ttmp[:, :, i:i+hh_, j:j+ww_]
                    tmp_list.append(patch)
                out_ = np.stack(tmp_list, axis=0)

        return out_, target

# -----------------------------
# 單 epoch 執行函式
# -----------------------------
def run_epoch(model, loader, train, optimizer, device, save_all=False, block_size=None):
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    count = 0
    s1 = 0
    s2 = 0
    yhat = []
    y = []
    from tqdm import tqdm
    with torch.set_grad_enabled(train):
        with tqdm(total=len(loader)) as pbar:
            for (X, outcome) in loader:
                outcome = outcome.to(device).float()
                y.append(outcome.detach().cpu().numpy())

                X = X.to(device)
                average = (X.ndim == 6)
                if average:
                    B, clips, c_, f_, hh_, ww_ = X.shape
                    X = X.view(-1, c_, f_, hh_, ww_)

                if block_size is None:
                    outputs = model(X)
                else:
                    outs = []
                    for j in range(0, X.shape[0], block_size):
                        outs.append(model(X[j:j+block_size]))
                    outputs = torch.cat(outs, dim=0)

                if save_all:
                    yhat.append(outputs.view(-1).detach().cpu().numpy())

                if average:
                    outputs = outputs.view(B, clips, -1).mean(dim=1)

                if not save_all:
                    yhat.append(outputs.view(-1).detach().cpu().numpy())

                loss = nn.functional.mse_loss(outputs.view(-1), outcome)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total += loss.item() * X.size(0)
                count += X.size(0)
                s1 += outcome.sum().item()
                s2 += (outcome ** 2).sum().item()

                pbar.set_postfix_str(f"{(total/count):.2f} ({loss.item():.2f}) / {(s2/count - (s1/count)**2):.2f}")
                pbar.update()

    y = np.concatenate(y)
    if not save_all:
        yhat = np.concatenate(yhat)
    return total/count, yhat, y

# -----------------------------
# 訓練函式
# -----------------------------
def train(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # (1) 利用訓練集計算正規化所需的 mean/std
    print("Reading train dataset to compute mean/std ...")
    train_for_stat = EchoDataset(
        csv_file=args.csv_file,
        videos_dir=args.videos_dir,
        split="TRAIN",
        target_type=args.task,
        length=args.frames,
        period=args.period,
        pad=12,
        alpha=args.alpha,
        mask_dir=args.mask_dir
    )
    mean, std = get_mean_and_std(train_for_stat, samples=128, batch_size=8, num_workers=args.num_workers)
    print("Computed mean=", mean, "std=", std)

    # (2) 建立訓練與驗證資料集
    ds_train = EchoDataset(
        csv_file=args.csv_file,
        videos_dir=args.videos_dir,
        split="TRAIN",
        target_type=args.task,
        mean=mean, std=std,
        length=args.frames, period=args.period,
        max_length=250,
        clips=1,
        pad=12,
        alpha=args.alpha,
        mask_dir=args.mask_dir
    )
    ds_val = EchoDataset(
        csv_file=args.csv_file,
        videos_dir=args.videos_dir,
        split="VAL",
        target_type=args.task,
        mean=mean, std=std,
        length=args.frames, period=args.period,
        max_length=250,
        clips=1,
        pad=None,
        alpha=args.alpha,
        mask_dir=args.mask_dir
    )
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type.startswith("cuda")), drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device.type.startswith("cuda")), drop_last=False)

    # (3) 模型建立：使用 torchvision 提供的 3D CNN (例如 r2plus1d_18)
    model = torchvision.models.video.__dict__[args.model_name](pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, 1)
    with torch.no_grad():
        model.fc.bias.fill_(55.0)
    if device.type.startswith("cuda"):
        model = nn.DataParallel(model)
    model = model.to(device)

    # (4) 載入預訓練權重 (如有)
    if args.weights is not None and os.path.exists(args.weights):
        ckp = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckp["state_dict"])
        print("Loaded pretrain weights from", args.weights)

    # (5) 設定 optimizer 與 learning rate scheduler
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    step_size = args.lr_step_period if args.lr_step_period is not None else math.inf
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.1)

    best_loss = float("inf")
    start_epoch = 0
    logf = open(os.path.join(args.output_dir, "log.csv"), "a")
    ckp_path = os.path.join(args.output_dir, "checkpoint.pt")
    if args.resume and os.path.exists(ckp_path):
        c = torch.load(ckp_path, map_location=device)
        model.load_state_dict(c["state_dict"])
        opt.load_state_dict(c["opt_dict"])
        sched.load_state_dict(c["scheduler_dict"])
        start_epoch = c["epoch"] + 1
        best_loss = c["best_loss"]
        logf.write(f"Resuming from epoch {start_epoch}\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        st = time.time()
        train_loss, train_pred, train_y = run_epoch(model, train_loader, True, opt, device)
        r2_train = sklearn.metrics.r2_score(train_y, train_pred)
        used = time.time() - st
        logf.write(f"{epoch},train,{train_loss},{r2_train},{used},{len(train_y)}\n")

        st = time.time()
        val_loss, val_pred, val_y = run_epoch(model, val_loader, False, None, device)
        r2_val = sklearn.metrics.r2_score(val_y, val_pred)
        used = time.time() - st
        logf.write(f"{epoch},val,{val_loss},{r2_val},{used},{len(val_y)}\n")
        logf.flush()

        sched.step()
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "opt_dict": opt.state_dict(),
            "scheduler_dict": sched.state_dict(),
            "best_loss": best_loss
        }
        torch.save(state, ckp_path)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(state, os.path.join(args.output_dir, "best.pt"))
    logf.close()
    print("Train finished, best val loss=", best_loss)

# -----------------------------
# 測試函式
# -----------------------------
def test(args):
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found => {args.weights}")

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 先從訓練集計算正規化參數
    print("Reading train dataset to compute mean/std for test ...")
    ds_train_stat = EchoDataset(
        csv_file=args.csv_file,
        videos_dir=args.videos_dir,
        split="TRAIN",
        target_type=args.task,
        length=args.frames,
        period=args.period,
        pad=12,
        alpha=args.alpha,
        mask_dir=args.mask_dir
    )
    mean, std = get_mean_and_std(ds_train_stat, samples=128, batch_size=8, num_workers=args.num_workers)
    print("Test: computed mean=", mean, "std=", std)

    model = torchvision.models.video.__dict__[args.model_name](pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, 1)
    with torch.no_grad():
        model.fc.bias.fill_(55.0)
    if device.type.startswith("cuda"):
        model = nn.DataParallel(model)
    model = model.to(device)

    ckp = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckp["state_dict"])
    print("Loaded best state from", args.weights)

    ds_test_single = EchoDataset(
        csv_file=args.csv_file,
        videos_dir=args.videos_dir,
        split="TEST",
        target_type=args.task,
        mean=mean, std=std,
        length=args.frames, period=args.period,
        max_length=250, clips=1,
        pad=None,
        alpha=args.alpha,
        mask_dir=args.mask_dir
    )
    loader_single = DataLoader(ds_test_single, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=(device.type.startswith("cuda")), drop_last=False)
    tloss, tpred, tY = run_epoch(model, loader_single, False, None, device, save_all=False)
    from math import sqrt
    r2_ = bootstrap(tY, tpred, sklearn.metrics.r2_score)
    mae_ = bootstrap(tY, tpred, sklearn.metrics.mean_absolute_error)
    rmse_ = bootstrap(tY, tpred, lambda a, b: sqrt(sklearn.metrics.mean_squared_error(a, b)))
    print(f"Test (one clip) R2= {r2_[0]:.3f} ({r2_[1]:.3f}-{r2_[2]:.3f}), MAE= {mae_[0]:.2f} ({mae_[1]:.2f}-{mae_[2]:.2f}), RMSE= {rmse_[0]:.2f} ({rmse_[1]:.2f}-{rmse_[2]:.2f})")

    ds_test_all = EchoDataset(
        csv_file=args.csv_file,
        videos_dir=args.videos_dir,
        split="TEST",
        target_type=args.task,
        mean=mean, std=std,
        length=args.frames, period=args.period,
        max_length=250, clips="all",
        pad=None,
        alpha=args.alpha,
        mask_dir=args.mask_dir
    )
    loader_all = DataLoader(ds_test_all, batch_size=1, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device.type.startswith("cuda")), drop_last=False)
    tloss2, tpred2, tY2 = run_epoch(model, loader_all, False, None, device, save_all=True, block_size=args.batch_size)
    merged = []
    for arr in tpred2:
        merged.append(arr)
    merged_means = np.array([x.mean() for x in merged])
    r2b = bootstrap(tY2, merged_means, sklearn.metrics.r2_score)
    maeb = bootstrap(tY2, merged_means, sklearn.metrics.mean_absolute_error)
    rmseb = bootstrap(tY2, merged_means, lambda a, b: sqrt(sklearn.metrics.mean_squared_error(a, b)))
    print(f"Test (all clips) R2= {r2b[0]:.3f} ({r2b[1]:.3f}-{r2b[2]:.3f}), MAE= {maeb[0]:.2f} ({maeb[1]:.2f}-{maeb[2]:.2f}), RMSE= {rmseb[0]:.2f} ({rmseb[1]:.2f}-{rmseb[2]:.2f})")

    # 寫入預測結果
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    with open(pred_path, "w") as g:
        for i, fn in enumerate(ds_test_all.df["FileName"].tolist()):
            arr = tpred2[i]  # shape=(nclips,)
            for cidx, val in enumerate(arr):
                g.write(f"{fn},{cidx},{val.mean():.4f}\n")
    print("Saved test predictions =>", pred_path)

    latexify()
    lo = min(tY2.min(), merged_means.min())
    hi = max(tY2.max(), merged_means.max())
    fig = plt.figure(figsize=(3, 3))
    plt.scatter(tY2, merged_means, color="k", s=1, zorder=2)
    plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
    plt.axis([lo-3, hi+3, lo-3, hi+3])
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("Actual EF (%)")
    plt.ylabel("Predicted EF (%)")
    plt.xticks([10,20,30,40,50,60,70,80])
    plt.yticks([10,20,30,40,50,60,70,80])
    plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
    plt.tight_layout()
    scatf = os.path.join(args.output_dir, "test_scatter.pdf")
    plt.savefig(scatf)
    plt.close(fig)
    print("Saved scatter =>", scatf)

def main():
    parser = argparse.ArgumentParser("EF Prediction Training/Test Script with Mask-based Enhancement")
    subparsers = parser.add_subparsers(dest="mode")

    # train 子命令
    p_train = subparsers.add_parser("train", help="Train EF model")
    p_train.add_argument("--csv_file", type=str, default=CSV_FILE)
    p_train.add_argument("--videos_dir", type=str, default=VIDEOS_ROOT)
    p_train.add_argument("--mask_dir", type=str, default=MASK_DIR, help="Path to precomputed masks")
    p_train.add_argument("--alpha", type=float, default=1.0, help="Background scale factor; lower means stronger enhancement")
    p_train.add_argument("--output_dir", type=str, default="./output_train")
    p_train.add_argument("--model_name", type=str, default="r2plus1d_18")
    p_train.add_argument("--pretrained", action="store_true", default=True)
    p_train.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    p_train.add_argument("--resume", action="store_true", help="Resume from checkpoint.pt if exists")
    p_train.add_argument("--task", type=str, default="EF")
    p_train.add_argument("--frames", type=int, default=32)
    p_train.add_argument("--period", type=int, default=2)
    p_train.add_argument("--batch_size", type=int, default=20)
    p_train.add_argument("--epochs", type=int, default=45)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--lr_step_period", type=int, default=15)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--seed", type=int, default=0)

    # test 子命令
    p_test = subparsers.add_parser("test", help="Test EF model")
    p_test.add_argument("--csv_file", type=str, default=CSV_FILE)
    p_test.add_argument("--videos_dir", type=str, default=VIDEOS_ROOT)
    p_test.add_argument("--mask_dir", type=str, default=MASK_DIR)
    p_test.add_argument("--alpha", type=float, default=1.0, help="Background scale factor; lower means stronger enhancement")
    p_test.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    p_test.add_argument("--output_dir", type=str, default="./output_test")
    p_test.add_argument("--model_name", type=str, default="r2plus1d_18")
    p_test.add_argument("--pretrained", action="store_true", default=True)
    p_test.add_argument("--task", type=str, default="EF")
    p_test.add_argument("--frames", type=int, default=32)
    p_test.add_argument("--period", type=int, default=2)
    p_test.add_argument("--batch_size", type=int, default=10)
    p_test.add_argument("--num_workers", type=int, default=4)
    p_test.add_argument("--device", type=str, default=None)
    p_test.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

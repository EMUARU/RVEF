
## convert_dicom_to_avi.ipynb

**檔案功能：**  
　本 Notebook 遍歷指定資料夾（包含子資料夾）中的 DICOM 檔案，  
　利用 pydicom 讀取 DICOM 影像後，對影像數據進行處理與裁剪（例如裁切掉不需要的邊界及應用遮罩篩選），  
　接著使用 OpenCV 將每筆 DICOM 影像序列轉換成 AVI 格式的影片，且影片解析度固定為 112 × 112（或依需求設定的尺寸）。

**主要參數：**  
　- **AllA4cNames**          ： 輸入的 DICOM 檔案所在資料夾路徑（預設為 "dicom_data/"）。  
　- **destinationFolder**    ： 轉換後 AVI 檔案儲存的位置（預設為 "des/"）。  
　- **cropSize**             ： 影片輸出的解析度，預設為 (112, 112)。

**dependencies (執行時版本)：**  
　- pydicom=2.3.1

---

## echonet_mask_gen.ipynb

**檔案功能：**  
　本 Notebook 主要用於從 Echonet 相關的視頻資料中提取指定幀影像，  
　並根據 CSV 檔案中提供的追蹤座標資訊生成對應的遮罩圖像。  
　具體流程包括：  
　　- 從 CSV 檔案 (VolumeTracings.csv) 中讀取追蹤座標，並根據文件內容建立每個視頻文件的幀索引與對應追蹤點。  
　　- 透過 OpenCV 讀取視頻檔案，根據索引提取出指定的幀。  
　　- 利用提供的追蹤點數據，藉由 skimage 的 polygon 函數生成二值遮罩。  
　　- 將提取的原始幀與生成的遮罩分別存入指定的資料夾，並在處理過程中記錄 log 檔案。

**主要參數：**  
　- **video_dir**           ： 存放 Echonet 視頻檔案的資料夾路徑。  
　- **volume_tracings_csv** ： 含有追蹤座標資訊的 CSV 檔案路徑。  
　- **output_dir**          ： 影片幀與遮罩圖像輸出的資料夾路徑。

---

## unet_lv.ipynb

**檔案功能：**  
　本 Notebook 利用 Echonet 資料集與 U-Net 模型架構執行心臟影像分割任務，  
　提供兩種訓練模式：  
　　1. **基本**：直接使用 Echonet 資料進行左心室 (LV) 分割訓練，適用於原始的左心室標註數據。  
　　2. **水平翻轉**：將 Echonet 資料進行完全水平翻轉，並以左心室分割標註模擬右心室分割訓練。

**主要參數：**  
　- **frames_dir** : 原始影像所在資料夾。  
　- **masks_dir**  : 遮罩影像所在資料夾。

---

## unet_rv.ipynb (版本 1)

**檔案功能：**  
　　使用以 LV 水平翻轉訓練得到的 U-Net 模型，在 RV 資料上測試分割推論與後處理。  
　　流程包括：  
　　- 從指定影片中隨機抽取幀進行測試。  
　　- 執行影像預處理（調整大小、轉為 Tensor）並進行模型推論。  
　　- 利用後處理（邊緣先驗、距離轉換）改善分割結果，展示原始預測、二值遮罩和後處理結果疊加圖。

**主要參數：**  
　- **model_path**     ： 模型權重的檔案路徑（例如：unet_best_model_LV_HorizontalFlip_112.pth）。  
　- **transform**      ： 影像轉換設定（Resize 至 112×112，再轉為 Tensor）。  
　- **csv_file**       ： 影片列表 CSV 檔案路徑（例如：FileList.csv）。  
　- **root_dir**       ： 影片檔案所在資料夾路徑。  
　- **user_threshold** ： 二值化閾值（由使用者輸入，例如 0.5）。  
　- **edge_width**     ： 邊緣寬度（由使用者輸入，例如 5）。

---

## unet_rv.ipynb (版本 2)

**檔案功能：**  
　　使用已經以 LV 水平翻轉訓練得到的 U-Net 模型，對 AVI 格式影片進行分割，模擬 RV 二元分割任務。  
　　主要流程包括：  
　　- 讀取影片中的每一幀並轉為灰階後依據預處理設定 (Resize 至 112×112, ToTensor) 進行轉換。  
　　- 利用 U-Net 模型產生 soft mask，依閾值（預設 0.5）二值化，再透過後處理（取最大連通區 + 凸包）精修遮罩。  
　　- 將每部影片的所有幀遮罩合併存成 numpy 檔案，放置於預設資料夾 (precomputed_masks)。

**主要參數：**  
　- **seg_model_path** : 分割模型權重檔案 (unet_best_model_LV_HorizontalFlip_112.pth)。  
　- **transform**      ： 影像預處理設定（Resize 至 112×112、轉為 Tensor）。  
　- **threshold**      ： 二值化閾值（預設 0.5）。  
　- **mask_dir**       ： 遮罩輸出資料夾（預設 precomputed_masks）。  
　- **csv_file**       ： 影片清單 CSV 檔（a4c-video-dir/FileList.csv）。  
　- **videos_dir**     ： 影片所在資料夾（a4c-video-dir/Videos）。

---

## rvef_test

**檔案功能：**  
　　用於比較有無使用 RV 區域加權通道對 2+1D CNN 預測 RVEF 表現的影響。  
　　主要流程包括：  
　　- 根據 CSV 清單讀取影片資料，再利用預先計算的遮罩 (npy 檔) 對影片資料進行局部區域加權增強（alpha < 1 時啟用）。  
　　- 採用 torchvision 提供的 2+1D CNN 模型（例如 r2plus1d_18）進行訓練與測試。  
　　- 分別計算並輸出模型在訓練/測試資料上的 R2、MAE 與 RMSE 指標，同時生成預測結果 CSV 及散點圖。

**主要參數：**  
　- **CSV_FILE**    ： 影片清單 CSV 路徑（預設 "/a4c-video-dir/FileList.csv"）。  
　- **VIDEOS_ROOT** ： 影片所在資料夾路徑（預設 "/a4c-video-dir/Videos"）。  
　- **MASK_DIR**    ： 預先計算遮罩所在資料夾（預設 "precomputed_masks"）。  
　- **alpha**       ： 加權參數 (alpha = 1 表示不加權，alpha < 1 則啟用 RV 區域加權增強)。  
　- **model_name**  ： 使用的 2+1D CNN 模型名稱（預設 "r2plus1d_18"）。  
　- **frames**      ： 每支影片選取的幀數（預設 32）。  
　- **period**      ： 幀取樣間隔（預設 2）。  
　- **batch_size**  ： 批次大小（訓練/測試各自可調整）。  
　- **epochs**      ： 訓練 epoch 數（例如 45）。  
　- **lr**          ： 學習率（預設 1e-4）。  
　- **weight_decay**： 權重衰減參數（預設 1e-4）。  
　- **output_dir**  ： 輸出結果目錄（用以儲存日誌、模型權重與預測結果）。

---

## 使用RV資料來源

[https://rvenet.github.io/dataset/](https://rvenet.github.io/dataset/)

---

+++ 醫院分割標註使用 +++

## view_classifier

**檔案功能：**  
　　本腳本用於對 "masklab_jpg_only" 資料夾中的影像進行視角分類，流程包括：  
　　- 使用 tensorflow.keras 的 ImageDataGenerator 進行影像加載，  
　　- 載入預先訓練好的 CNN 模型 (例如 "mymodel_echocv_500-500-8_adam_16_0.9394.h5")，  
　　- 對資料夾中的影像進行分類預測，  
　　- 輸出包含檔案名稱、預測結果與置信度的 CSV 檔 (例如 "masklab_results.csv")。

**主要參數：**  
　- **dir**          ： 輸入影像所在資料夾，預設為 "masklab_jpg_only"。  
　- **model_name**   ： 預先訓練的模型檔案名稱 (例如 "mymodel_echocv_500-500-8_adam_16_0.9394.h5")。  
　- **input_shape**  ： 輸入影像尺寸 (例如 (224,224,3))。  
　- **batch_size**   ： 分類預測使用的批次大小（預設 2）。  
　- **results_file** ： 輸出結果 CSV 檔案名稱，預設為 "masklab_results.csv"。

 
## 分類器參考來源

[https://github.com/raventan95/echo-view-classifier](https://github.com/raventan95/echo-view-classifier)

---

## filtered_a4c_view.ipynb

**檔案功能：**  
　　整理醫院標註資料，挑選出 A4C 視角的影像與對應遮罩。  
　　依據以下 CSV 進行篩選：  
　　- **masklab_results.csv**：原始視角分類結果（含 Confidence 與 Prediction 等欄位）。  
　　- **reviewed_data_results.csv**：A4C和A5C分類中可信度 0.8 以上資料經人工篩選後較清晰的資料（Keep 標記）。  
　　程式會依據 CSV 內容與檔案存在性，進行以下步驟：  
　　　1. 讀取並前處理 JPG 影像與 TIF 遮罩（灰階化、中心裁切、調整尺寸、套用掃描區遮罩）。  
　　　2. 將處理後檔案儲存至指定資料夾。  
　　　3. 產生更新後的新 CSV，記錄處理後影像與遮罩的相對路徑。

**主要參數：**  
　- **csv_file**      ： 原始標註資料 CSV（masklab_results.csv）。  
　- **root_dir**      ： 原始影像與遮罩所在資料夾（包含 JPG 與 TIF 檔案）。  
　- **reviewed_csv**  ： 人工篩選後的 CSV（reviewed_data_results.csv）。  
　- **output_folder** ： 處理後檔案儲存資料夾（例如 processed_masklab）。  
　- **new_csv_file**  ： 處理後 CSV 檔案（例如 processed_masklab_results.csv）。

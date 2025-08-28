
# LRF-Unet

這是我的畢業論文的程式專案，請照著以下流程安裝並執行

本專案也有同步推上github，網址在: https://github.com/nick70422/LRF-Unet

## 1. Structure

因為沒有成功把SAR-U-Net整合進去nnUNet的程式框架，所以只能獨立於其他模型一包，以下是程式架構

```bash
└─ 專案位置/
    ├─ proposed/
    |   ├─ nnUNet/
    |   ├─ UNetPP/
    |   ├─ UNet3P/
    |   ├─ SwinUnet/
    |   ├─ SwinUnetV2/
    |   └─ LRFUnet/
    └─ SAR-U-Net-liver-segmentation-master/
```
## 2. Installation

首先，如果是在windows底下執行，請先安裝wsl以及Linux(本來就是Linux系統直接跳過這條)

安裝cuda以及cudnn，cuda版本為11.8

在你的Linux系統底下安裝python，版本為3.10.16

### 虛擬環境以及相關package安裝
安裝一個虛擬環境
```bash
# 1. 確認 venv 可用
python3 -m venv --help     # 若沒報錯就代表已內建

# 2. 建立虛擬環境（範例放在 ~/venvs/myproj）
python3 -m venv ~/venvs/虛擬環境名

# 3. 啟用
source ~/venvs/虛擬環境名/bin/activate
```

從requirements.txt安裝pip當中的包(記得確認你是否有在虛擬環境中)
```bash
pip install -r requirements.txt
```

安裝nnUNet，特別注意是舊版的v1而不是最新版的v2

```bash
pip install git+https://github.com/MIC-DKFZ/nnUNet.git@nnunetv1
```

### nnUNet環境配置

首先，在你想要裝nnUNet程式包所使用的raw dataset、preprocessed dataset以及trained model的位置創一個資料夾(筆者是直接裝在專案資料夾/proposed/底下，並命名為data)並cd進去這個資料夾裡面
```bash
cd 專案路徑/proposed
mkdir ./data
cd ./data
```

再來，在這個資料夾裡面創建以下三個資料夾
```bash
mkdir ./nnUNet_raw
mkdir ./nnUNet_prerpocessed
mkdir ./nnUNet_trained_models
```

在nnUNet_raw的底下再創建一個nnUNet_raw_data資料夾

接下來編輯系統設定(這步操作類似windows裡面的環境變數設定)
```bash
nano ~/.bashrc
```

在檔案末尾加入下面這行方便未來呼叫虛擬環境(非必須，如果要生效的話需要離開wsl環境重新進入)
```bash
alias activate_LRFUnet='source ~/venvs/虛擬環境名/bin/activate'
```

在檔案末尾加入下面三行（改成你自己的路徑）
```bash
export nnUNet_raw_data_base="nnUNet_raw的路徑"
export nnUNet_preprocessed="nnUNet_preprocessed的路徑"
export RESULTS_FOLDER="nnUNet_trained_models的路徑"
```

存檔並離開（nano：Ctrl-O Enter、Ctrl-X）。

執行以立即生效
```bash
source ~/.bashrc
```

驗證資料夾是否有被正確抓到
```bash
echo $nnUNet_raw_data_base
echo $nnUNet_preprocessed
echo $RESULTS_FOLDER
```
## 3. 資料準備
首先請確認自己是在前面步驟創建好的虛擬環境裡執行接下來的操作!!

到老師的NAS裡面把所使用的raw dataset下載下來(目前data是放在"詮智/dataset/LRF-Unet dataset"底下)
下載下來會是這樣子的結構
```bash
└─ LRF-Unet dataset/
    ├─ Task581_CSMAbdominalSegFold1/
    ├─ Task582_CSMAbdominalSegFold2/
    ├─ Task583_CSMAbdominalSegFold3/
    ├─ Task584_CSMAbdominalSegFold4/
    ├─ Task585_CSMAbdominalSegFold5/
    └─ renumbered_files.xlsx

```

把Task581_CSMAbdominalSegFold1~Task585_CSMAbdominalSegFold5分別複製到以下路徑

```bash
專案路徑/proposed/data/nnUNet_raw/nnUNet_raw_data(就是nnUNet_raw的路徑)
專案路徑/SAR-U-Net-liver-segmentation-master/raw_data
```

接下來回到老師的NAS裡面的"詮智/dataset/SAR-U-Net dataset"下載並放到以下路徑

```bash
專案路徑/SAR-U-Net-liver-segmentation-master/DCM_PNG
```

### SAR-U-Net的Pre-processing
依序執行以下程式來執行validation資料的劃分以及data pre-processing:
```bash
python3 preprocessing_CSM_1.py
python3 preprocessing_CSM_2.py
python3 preprocessing_CSM_3.py
```

### nnUNet的Pre-processing
使用以下指令
```bash
cd 專案路徑/proposed/準備data
./preprocessing.sh
```
## 4. 跑實驗
首先請確認自己是在前面步驟創建好的虛擬環境裡執行接下來的操作!!

### 事前準備
從程式碼到.sh腳本，有許多路徑是需要更改為自己電腦中的路徑的，以下列出nnUNet包底下的模型(以LRF-Unet中的為例)，以及SAR-U-Net要修改的地方

#### LRF-Unet
run_training.py
```bash
# 原本的
workdir = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/'

# 更改過後
workdir = '專案路徑/proposed/data/'
```
run_testing.py
```bash
# 原本的
base_dir = '/mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/lightweight_UNetPP'

# 更改過後
workdir = '專案路徑/proposed/LRFUnet/'
```
train_test_5fold_proposed.sh
```bash
# 原本的
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+1))_${dataset_task_name}Fold1/imagesTs -o ./results/fold1 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+1)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+2))_${dataset_task_name}Fold2/imagesTs -o ./results/fold2 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+2)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+3))_${dataset_task_name}Fold3/imagesTs -o ./results/fold3 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+3)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+4))_${dataset_task_name}Fold4/imagesTs -o ./results/fold4 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+4)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /mnt/c/Nick/研究/medical/code/DL_models_benchmark-master/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+5))_${dataset_task_name}Fold5/imagesTs -o ./results/fold5 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+5)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos

#更改過後
python3 predict_simple.py -i /專案路徑/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+1))_${dataset_task_name}Fold1/imagesTs -o ./results/fold1 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+1)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /專案路徑/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+2))_${dataset_task_name}Fold2/imagesTs -o ./results/fold2 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+2)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /專案路徑/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+3))_${dataset_task_name}Fold3/imagesTs -o ./results/fold3 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+3)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /專案路徑/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+4))_${dataset_task_name}Fold4/imagesTs -o ./results/fold4 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+4)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
python3 predict_simple.py -i /專案路徑/data/nnUNet_raw/nnUNet_raw_data/Task$(($dataset+5))_${dataset_task_name}Fold5/imagesTs -o ./results/fold5 -tr nnUNetPlusPlusTrainerV2 -m 2d -p nnUNetPlansv2.1 -t $(($dataset+5)) -chk model_best -f 0 --factorization LRF --low_rank_conv $LRF_type --LRF_r $LRF_r --start_pos $LRF_start_pos
```

### 模型的比較實驗

#### 切割表現
LRF-Unet
```bash
cd 專案位置/proposed/LRFUnet
./train_test_5fold_proposed.sh
```

nnUNet程式包底下的其他模型
```bash
cd 專案位置/proposed/模型資料夾名
./train_test_5fold.sh
```

SAR-U-Net
```bash
cd 專案位置/SAR-U-Net-liver-segmentation-master

# training
./train_5fold.sh

# testing
python3 predict.py
```

#### 參數量&FLOPs
執行專案位置/propsoed的cal_Flops.py來查看模型的參數量跟FLOPs
使用到的模型都有備註在該檔案裡面的main()底下，要使用的就解除mark並把其他模型都mark起來
```bash
cd 專案位置/proposed/
python3 cal_Flops.py
```

#### 訓練時間(平均每個fold)
將所有模型都訓練過後，對模型做以下動作來計算訓練跟推論時間:

nnUNet底下的模型

```bash
cd 專案位置/data/nnUNet_trained_models/nnUNet/2d
mkdir ./時間計算
```

接下來把訓練好的log檔案複製到這個資料夾底下(模型檔可以不用)，架構如下
```bash
└─ 時間計算/
    ├─ proposed/
    |   ├─ Task581_CSMAbdominalSegFold1/
    |   ├─ Task582_CSMAbdominalSegFold2/
    |   ├─ Task583_CSMAbdominalSegFold3/
    |   ├─ Task584_CSMAbdominalSegFold4/
    |   └─ Task585_CSMAbdominalSegFold5/
    ├─ nnUNet/
    ├─ UNetPP/
    ├─ UNet3P/
    ├─ SwinUnet/
    └─ SwinV2Unet/
```

再來執行以下指令以計算訓練時間
```bash
cd 專案位置/proposed
python3 ./training_time_cal/training_time_cal.py
```

SAR-U-Net

請把SAR-U-Net的5-fold訓練過程的所有print出來的訊息複製進去一個SAR-U-Net_log.txt(如果有能力的話最好寫進程式自動輸出log)，並且將其放到以下目錄:
```bash
專案位置/proposed/training_time_cal
```

在相同位置會有一個同名的檔案，請確保自己訓練的輸出結果跟它格式一致，並取代掉它

接下來執行訓練時間的計算
```bash
cd 專案位置/proposed
python3 ./training_time_cal/training_time_cal.py
```

#### 推論時間(平均每張圖)
這邊的比較麻煩一些，都是程式統計並且手動輸入進另一個小程式計算的

nnUNet底下的模型(以LRF-Unet為例)
執行完訓練後到以下路徑
```bash
專案路徑/proposed/LRFUnet/results/fold1
```

在這邊找到一個inference_time.txt並打開，看起來格式會像是這樣子
```bash
total_seconds	58.011662
cases	228
seconds_per_case	0.254437
```

這樣你就可以獲得LRF-Unet的fold1中的inference時間以及case數量

接下來重複五個fold獲得inference時間總和，再除以5fold的所有case數量就可以得到該模型的inference time

SAR-U-Net

執行以下指令來產生inference_time.txt
```bash
cd SAR-U-Net-liver-segmentation-master
python3 predict_NIfTI.py
```

接著到以下位置找到inference_time.txt，並重複相同的計算方式來獲得inference time

```bash
SAR-U-Net-liver-segmentation-master\results\NIfTI\fold1
```

### Ablation study
baseline (同上面模型比較實驗的UNet++)
```bash
cd 專案位置/proposed/UNetPP
./train_test_5fold.sh
```

MV3 only
```bash
cd 專案位置/proposed/LRFUnet
./train_test_5fold_MV3_only.sh
```

LRF-Conv only
```bash
cd 專案位置/proposed/LRFUnet
./train_test_5fold_LRFConv_only.sh
```

完整的LRF-Unet (同上面模型比較實驗的LRF-Unet)
```bash
cd 專案位置/proposed/LRFUnet
./train_test_5fold_proposed.sh
```

雖然論文裡面沒有陳列ablation study模型的模型效能部分，但是有興趣試試看的話，它們跟其他模型一樣放在cal_Flops.py裡面

### r值的比較
#### 切割表現
這個實驗用的跟前面用的propose LRF-Unet完全是相同檔案，不過要在train_test_5fold_proposed.sh裡面修改LRF_r這個參數
```bash
# 原本
LRF_r="128"
# 修改後
LRF_r="欲實驗的r值"
```

訓練及測試LRF-Unet (同上面模型比較實驗的LRF-Unet)
```bash
cd 專案位置/proposed/LRFUnet
./train_test_5fold_proposed.sh
```

#### 模型效能
這邊跟前面用的LRF-Unet也一樣步驟
執行專案位置/propsoed的cal_Flops.py來查看模型的參數量跟FLOPs
使用到的模型都有備註在該檔案裡面的main()底下，要使用的就解除mark並把其他模型都mark起來
```bash
cd 專案位置/proposed/
python3 cal_Flops.py
```

並且函數get_MobileNetV3UNetPP_LRF_AB()裡面要做如下的修改，這邊以r值改用32進行演示:
```bash
# 原本的
def get_MobileNetV4UNetPP_LRF_AB():
    return Generic_MobileV4UNetPlusPlus_LRF(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='AB', LRF_r=128, LRF_start_pos=0, use_CBAM=False, use_ASPP=False)

# 修改後
def get_MobileNetV4UNetPP_LRF_AB():
    return Generic_MobileV4UNetPlusPlus_LRF(num_input_channels, base_num_features, num_classes,
                                            len(net_num_pool_op_kernel_sizes),
                                            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs,
                                            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
                                            LRF_type='AB', LRF_r=32, LRF_start_pos=0, use_CBAM=False, use_ASPP=False)
```

### 注意事項
在所有nnUNet程式包底下的模型都需要注意以下事項:

(1) 每次訓練都會在專案位置/proposed/data/nnUNet_trained_models/nnUNet/2d產生Task581_CSMAbdominalSegFold1~Task585_CSMAbdominalSegFold5五個資料夾，為訓練好的模型檔以及訓練參數還有過程的紀錄

(2) inference完之後在./results底下會產生fold1~fold5的資料夾

(3) testing之後再./results底下會產生visualize以及
testing_results.xlsx

在SAR-U-Net也有類似情況:

(1) 訓練完的模型會放在/SAR-U-Net-liver-segmentation-master/final checkpoints/Se_PPP_ResUNet_ce_NoS底下的fold1~fold5資料夾

(2) SAR-U-Net這包程式裡面有tensoarboard紀錄，所以訓練過程記錄的內容會儲存在/SAR-U-Net-liver-segmentation-master/runs裡面

(3) 跑完測試的圖像會放在/SAR-U-Net-liver-segmentation-master/results底下的fold1~fold5資料夾並且會有相同格式的testing_results.xlsx

上述提到的所有資料都建議在開始下一輪訓練之前刪除或是移動到其他位置留存，以免影響後續訓練!!影響後續訓練!!

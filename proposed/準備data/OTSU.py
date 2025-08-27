import nibabel as nib
from skimage import io, filters, morphology, measure, draw, img_as_ubyte
import numpy as np
import cv2


if __name__ == "__main__":
    img = nib.load("CSMAbdominalSegFold1_0001_0000.nii.gz")
    data = img.get_fdata()[:, :, 0]

    data += -1024
    data[data > 240] = 240
    data[data < -160] = -160
    
    normalized_data = (data - data.min()) / (data.max() - data.min())
    image_u8 = (normalized_data)          # 0/255 單通道
    cv2.imshow("original image", normalized_data)   # 灰階顯示
    cv2.waitKey(0)

    # ② Otsu 門檻 → 二值遮罩 (True=前景)
    th = filters.threshold_otsu(data)
    mask = data >= th

    # ③ 可選前處理：剔除雜點、補洞
    mask = morphology.remove_small_objects(mask, min_size=64)   # <64 px 直接丟
    mask = morphology.remove_small_holes(mask, area_threshold=64)

    mask_u8 = (mask.astype(np.uint8) * 255)          # 0/255 單通道
    cv2.imshow("Otsu Mask (thresholded)", mask_u8)   # 灰階顯示
    cv2.waitKey(0)

    # ④ 連通元件標記
    label, num = measure.label(mask, connectivity=2, return_num=True)
    props = measure.regionprops(label)

    # ⑤ 擷取每個物件的「外接矩形」(min_row, min_col, max_row, max_col)
    bboxes = [p.bbox for p in props]

    # ⑥ 找到面積最大的物件
    largest = max(props, key=lambda p: p.area)
    largest_bbox = largest.bbox            # 同樣是 (r0, c0, r1, c1)

    # ⑦（可選）把所有物件畫框；最大物件用不同顏色突出
    # 1) 把 mask 轉成可顯示的 3-channel 圖 (0/255, BGR)
    mask_u8   = (mask.astype(np.uint8) * 255)          # 0/255 單通道
    mask_bgr  = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    # 2) 對所有物件畫紅框；最大物件畫綠框
    for p in props:
        r0, c0, r1, c1 = p.bbox                       # (row, col) 座標
        cv2.rectangle(                                 # OpenCV 用 (x, y) = (col, row)
            mask_bgr, (c0, r0), (c1-1, r1-1),
            color=(0, 0, 255), thickness=1             # BGR: 紅色
        )

    r0, c0, r1, c1 = largest.bbox
    cv2.rectangle(
        mask_bgr, (c0, r0), (c1-1, r1-1),
        color=(0, 255, 0), thickness=2                 # BGR: 綠色 (較粗)
    )

    # 3) 顯示（不存檔）
    window_name = f"框物件圖"
    cv2.imshow(window_name, mask_bgr)
    cv2.waitKey(0)          # 等待任意鍵
    cv2.destroyAllWindows() # 關閉視窗
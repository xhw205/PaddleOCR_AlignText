from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from interval import Interval
from PIL import Image

det_model_dir = "./inference/ch_ppocr_server_v2.0_det_infer/"
rec_model_dir = "./inference/ch_ppocr_server_v2.0_rec_infer/"
cls_model_dir = "./inference/ch_ppocr_mobile_v2.0_cls_infer/"

class OCR(PaddleOCR):
    def __init__(self, **kwargs):
        super(OCR, self).__init__(**kwargs)

    def ocr_new(self, img, det=True, rec=True, cls=True):
        res = self.ocr(img, det=det, rec=rec, cls=cls)
        if res!=None:
            return res
        else:
            img_v = Image.open(img).convert("RGB")
            img_v = np.asanyarray(img_v)[:,:,[2,1,0]]
            dt_boxes, rec_res = self.__call__(img_v)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]

def align_text(res, threshold=0):
    res.sort(key=lambda i: (i[0][0][0]))  # 按照x排
    already_IN, line_list = [], []
    for i in range(len(res)):  # i当前
        if res[i][0][0] in already_IN:
            continue
        line_txt = res[i][1][0]
        already_IN.append(res[i][0][0])
        y_i_points = [res[i][0][0][1], res[i][0][1][1], res[i][0][3][1], res[i][0][2][1]]
        min_I_y, max_I_y = min(y_i_points), max(y_i_points)
        curr = Interval(min_I_y + (max_I_y - min_I_y) // 3, max_I_y)
        curr_mid = min_I_y + (max_I_y - min_I_y) // 2

        for j in range(i + 1, len(res)):  # j下一个
            if res[j][0][0] in already_IN:
                continue
            y_j_points = [res[j][0][0][1], res[j][0][1][1], res[j][0][3][1], res[j][0][2][1]]
            min_J_y, max_J_y = min(y_j_points), max(y_j_points)
            next_j = Interval(min_J_y, max_J_y - (max_J_y - min_J_y) // 3)

            if next_j.overlaps(curr) and curr_mid in Interval(min_J_y, max_J_y):
                line_txt += (res[j][1][0] + "  ")
                already_IN.append(res[j][0][0])
                curr = Interval(min_J_y + (max_J_y - min_J_y) // 3, max_J_y)
                curr_mid = min_J_y + (max_J_y - min_J_y) // 2
        line_list.append((res[i][0][0][1], line_txt))
    line_list.sort(key=lambda x: x[0])
    txt = '\n'.join([i[1] for i in line_list])
    return txt

paddle_ocr_engin = OCR(det_model_dir= det_model_dir,rec_model_dir= rec_model_dir,cls_model_dir= cls_model_dir,use_angle_cls=True)
res = paddle_ocr_engin.ocr_new('./test.jpg')
print(res)
print(align_text(res))

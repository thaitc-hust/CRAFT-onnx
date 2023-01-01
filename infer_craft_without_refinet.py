import torch
import cv2
import onnxruntime as rt
import craft_utils
import imgproc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--craftonnxpath', type=str, default='onnx_model/craftmlt25k.onnx', help='path craft mlt 25k onnx model') 
parser.add_argument('--device', type=str, default='cuda', help='device')  
parser.add_argument('--image', type=str, default='images/16.jpg', help='image path inference') 
opt = parser.parse_args()

sess = rt.InferenceSession(opt.craftonnxpath)
input_name = sess.get_inputs()[0].name
img = imgproc.loadImage(opt.image)
img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
ratio_h = ratio_w = 1 / target_ratio
x = imgproc.normalizeMeanVariance(img_resized)
x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
x = x.unsqueeze(0)                # [c, h, w] to [b, c, h, w]

y, feature = sess.run(None, {input_name: x.numpy()})
score_text = y[0,:,:,0]
score_link = y[0,:,:,1]
boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.5, 0.4, 0.4, False)
boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

for k in range(len(polys)):
    if polys[k] is None: polys[k] = boxes[k]

bboxes_xxyy = []
h,w,c = img.shape
ratios = []

for box in boxes:
    x_min = max(int(min(box, key=lambda x: x[0])[0]),1)
    x_max = min(int(max(box, key=lambda x: x[0])[0]),w-1)
    y_min = max(int(min(box, key=lambda x: x[1])[1]),3)
    y_max = min(int(max(box, key=lambda x: x[1])[1]),h-2)    
    bboxes_xxyy.append([x_min-1,x_max,y_min-1,y_max])

if len(bboxes_xxyy) >0:
    for idx, text_box in enumerate(bboxes_xxyy):
        # text_in_cell = img[text_box[2]:text_box[3], text_box[0]:text_box[1]]
        # cv2.imwrite('result/'+str(idx)+'.jpg', text_in_cell)
        img = cv2.rectangle(img,(text_box[0],text_box[2]), (text_box[1],text_box[3]), (0,0,255), 2)

        # text_in_cell = Image.fromarray(text_in_cell)
        # text_result.append(self.module_text_recognition.predict_text(text_in_cell))
    cv2.imwrite('result/result_without_refinet.jpg', img)
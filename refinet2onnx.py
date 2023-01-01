import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image, ImageOps
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
from craft import CRAFT
from collections import OrderedDict
from refinenet import RefineNet

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class TextDetection:
    def __init__(self, device: torch.device, trained_model:str, refinenet_model: str, save_refine_onnx:str):
        self.trained_model = trained_model
        self.refiner_model = refinenet_model
        self.save_refine_onnx = save_refine_onnx
        self.device = device
        self.refine = True
        self.text_threshold = 0.7
        self.canvas_size = 1280
        self.link_threshold = 0.4
        self.low_text = 0.4
        self.mag_ratio = 1.5
        self.cuda = False
        self.poly = False
        self.setup()
        self.ratio_h = 0
        self.ratio_w = 0
        self.usingrefine = True

    def setup(self):
        self.net = CRAFT()
        self.net.load_state_dict(copyStateDict(torch.load(self.trained_model, map_location = self.device)))
        self.net.eval()
        print('-- LOADING NET --')
        self.refinenet = RefineNet()
        self.refinenet.load_state_dict(copyStateDict(torch.load(self.refiner_model, map_location = self.device)))
        self.refinenet.eval()
        print('-- LOADING REFINENET --')
        
    def preprocessing(self, img):
        #resize
        self.img_resized, self.target_ratio, self.size_heatmap = \
                imgproc.resize_aspect_ratio(img, self.canvas_size, \
                interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        self.ratio_h = self.ratio_w = 1 / self.target_ratio
        # preprocessing
        print(self.img_resized)
        x = imgproc.normalizeMeanVariance(self.img_resized)
        print(x.shape)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        return x

    def refinet2onnx(self, img):
        with torch.no_grad():
            y, feature =self.net(self.preprocessing(img))
        torch.onnx.export(self.refinenet,
                (y, feature),
                self.save_refine_onnx,
                export_params=True,
                verbose=True,
                input_names = ['y', 'feature'],   # the model's input names
                output_names = ['y_refiner'], # the model's output names
                dynamic_axes={'y' : {0 : 'Transposey_dim_0', 1: 'y_dynamic_axes_1', 2:'y_dynamic_axes_2'},    # variable length axes
                            'feature' : {0: 'Transposey_dim_0', 2: 'feature_dynamic_axes_1', 3: 'feature_dynamic_axes_2'}, 
                            })
        print('[INFO] Done convert refine pytorch to onnx !')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--craftmlt25kpthpath', type=str, default='weights/craft_mlt_25k.pth', help='path model craft mlt 25k pytorch') 
    parser.add_argument('--refinetpthpath', type=str, default='weights/craft_refiner_CTW1500.pth.pth', help='path model refine pytorch') 
    parser.add_argument('--device', type=str, default='cuda', help='device') 
    parser.add_argument('--refinetonnxpath', type=str, default='onnx_model/refine.onnx', help='path save refine onnx model')  
    opt = parser.parse_args()
    print('*' *10)
    print(opt)  
    print('*' *10)
    img = imgproc.loadImage('./images/16.jpg')
    module = TextDetection(device=opt.device, trained_model=opt.craftmlt25kpthpath, refinenet_model=opt.refinetpthpath, save_refine_onnx=opt.refinetonnxpath)
    module.refinet2onnx(img)
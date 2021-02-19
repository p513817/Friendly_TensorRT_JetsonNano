import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models.alexnet import alexnet

from torch2trt import torch2trt
from torch2trt import TRTModule

import os
import cv2
import PIL.Image as Image
import time

# Use to print info and timing
from print_log import log


def load_model():

    model_log = log('Load {} ... '.format('alexnet & tensorrt'))
    
    model = alexnet().eval().cuda()
    model.load_state_dict(torch.load('alexnet.pth'))
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
    
    model_log.end()

    return (model, model_trt)

def load_data(img_path):
    
    data_log = log('Load data ...')
    
    img_pil = Image.open(img_path)
    trans = T.Compose([T.Resize(256),T.CenterCrop(224), T.ToTensor()])
    
    data_log.end()

    return trans(img_pil).unsqueeze(0).cuda()

def load_label(label_path):
    f = open( label_path, 'r')
    return f.readlines()    

def infer(trg_model, trg_label, trg_tensor, info = 'Normal Model'):
    
    softmax = nn.Softmax(dim=0)
    infer_log = log('[{}] Start Inference ...'.format(info))

    with torch.no_grad():
        predict = trg_model(trg_tensor)[0]
        predict_softmax = softmax(predict)
    
    infer_log.end()
    label = trg_label[torch.argmax(predict_softmax)].replace('\n',' ')
    value = torch.max(predict_softmax)
    return ( label, value)

if __name__ == "__main__":

    # Load Model
    model, model_trt = load_model()

    # Input Data
    img_path = 'test_photo.jpg'
    img_tensor = load_data(img_path)

    # Label
    label_path = 'imagenet_classes.txt'
    labels = load_label(label_path)

    # Predict : Normal
    label, val = infer(model, labels, img_tensor, "Normal AlexNet")
    print('\nResult: {}  {}\n'.format(label, val))

    # Predict : TensorRT
    label_trt, val_trt = infer(model_trt, labels, img_tensor, "TensorRT AlexNet")
    print('\nResult: {}  {}\n'.format(label_trt, val_trt))

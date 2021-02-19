import torch
from torch2trt import torch2trt
from torchvision import transforms as T
from torchvision.models.alexnet import alexnet
import time

# Use to print info and timing
from print_log import log

# Load Model
alexnet_pth = 'alexnet.pth'
load_model = log("Load Model...")
model = alexnet(pretrained=True).eval().cuda()
torch.save(model.state_dict(), alexnet_pth, _use_new_zipfile_serialization=False)
load_model.end()

# TRT Model
convert_model = log("Convert Model...")
x = torch.ones((1, 3, 224, 224)).cuda()
model_trt = torch2trt(model, [x])
convert_model.end()

# Save Model
alexnet_trt_pth = 'alexnet_trt.pth'
save_model = log("Saving TRT...")
torch.save(model_trt.state_dict(), alexnet_trt_pth)
save_model.end()

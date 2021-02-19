import tensorrt as trt
from PIL import Image
import torchvision.transforms as T
import numpy as np

import common
from engine import load_engine
from log import timer, logger

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def load_data(path):
    trans = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor()
    ])

    img = Image.open(path)
    img_tensor = trans(img).unsqueeze(0)
    return np.array(img_tensor)


# load trt engine
load_trt = timer("Load TRT Engine")
trt_path = 'alexnet.trt'
engine = load_engine(trt_runtime, trt_path)
load_trt.end()

# allocate buffers
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# load data
inputs[0].host = load_data('../test_photo.jpg')

# inference
infer_trt = timer("TRT Inference")
with engine.create_execution_context() as context:
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
preds = trt_outputs[0]
infer_trt.end()

# Get Labels
f = open('../imagenet_classes.txt')
t = [ i.replace('\n','') for i in f.readlines()]
logger(f"Result : {t[np.argmax(preds)]}")
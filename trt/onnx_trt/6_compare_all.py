
import torch
import torchvision
import onnxruntime as ort
import tensorrt as trt

import torchvision.transforms as T
import time
import numpy as np
from PIL import Image
import numpy as np

import common
from engine import load_engine
from log import timer, logger

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def load_data():

    trans = T.Compose([ T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    img = Image.open('../test_photo.jpg')
    img_pil = trans(img).unsqueeze(0)
    img_np = np.array(img_pil)

    return img_pil.cuda(), img_np

def load_model():
    # Load Model
    load_model = timer('Load Normal Model')
    model = torchvision.models.alexnet(pretrained=True).eval().cuda()
    load_model.end()
    return model

def load_ort():
    # Load ONNX
    load_onnx = timer('Load ONNX Model')
    ort_session = ort.InferenceSession('alexnet.onnx')
    load_onnx.end()

    return ort_session, ort_session.get_inputs()[0].name

def load_trt():
    # load trt engine
    load_tensorrt = timer("Load TRT Engine")
    trt_path = 'alexnet.trt'
    engine = load_engine(trt_runtime, trt_path)
    load_tensorrt.end()

    return engine

def get_buffer(engine, img_np):
    # allocate buffers
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # load data
    inputs[0].host = img_np

    return inputs, outputs, bindings, stream

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_res(predict):
    # Get Labels
    f = open('../imagenet_classes.txt')
    t = [ i.replace('\n','') for i in f.readlines()]
    
    predict = softmax(np.array(predict))
    print(f"Result : {t[np.argmax(predict)]} , {np.max(predict)}\n")

if __name__ == "__main__":
    
    ### Prepare
    img_pil, img_np = load_data()
    model = load_model()
    ort, in_name = load_ort()
    engine = load_trt()

    ### Normal Model Infer
    infer_torch = timer("Run Torch Infer")
    with torch.no_grad():   
        out_torch = model(img_pil)[0]
    infer_torch.end()
    get_res(out_torch.cpu())

    ### ORT Infer
    infer_onnx = timer('Run ORT Infer')
    out_ort = ort.run(None, {in_name: img_np})[0]
    infer_onnx.end()
    get_res(out_ort)

    ### TRT Infer
    infer_trt = timer("Run TRT Infer")
    inputs, outputs, bindings, stream = get_buffer(engine, img_np)
    with engine.create_execution_context() as context:
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    infer_trt.end()
    get_res(trt_outputs[0])
    


import onnxruntime as ort
import time
from PIL import Image
import numpy as np
from torchvision import transforms as T

# Custom 
from log import timer, logger

trans = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor()
])

img = Image.open('../test_photo.jpg')
img_tensor = trans(img).unsqueeze(0)
img_np = np.array(img_tensor)

logger('Image : {} >>> {}'.format(np.shape(img) , np.shape(img_tensor)))

# ONNX Run Time
load_onnx = timer('Load ONNX Model')
ort_session = ort.InferenceSession('alexnet.onnx')
load_onnx.end()

# run( out_feed, in_feed, opt )
input_name = ort_session.get_inputs()[0].name

infer_onnx = timer('Run Infer')
outputs = ort_session.run(None, {input_name: img_np})[0]
infer_onnx.end()

# Get Labels
f = open('../imagenet_classes.txt')
t = [ i.replace('\n','') for i in f.readlines()]
logger("Result : {}".format(t[np.argmax(outputs)]))
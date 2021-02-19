import tensorrt as trt
from log import timer, logger

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_path, shape = [1,224,224,3]):
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = (256 << 20)    # 256MiB
        builder.fp16_mode = True # fp32_mode -> False
        
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())

        engine = builder.build_cuda_engine(network)

        return engine

def save_engine(engine, engine_path):
    
    buf = engine.serialize()
    with open(engine_path, 'wb') as f:
        f.write(buf)

def load_engine(trt_runtime, engine_path):
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)

    return engine

if __name__ == "__main__":
    
    onnx_path = 'alexnet.onnx'
    trt_path = 'alexnet.trt'
    input_shape = [1, 224, 224, 3]
    
    build_trt = timer('Parser ONNX & Build TensorRT Engine')
    engine = build_engine(onnx_path, input_shape)
    build_trt.end()
    
    save_trt = timer('Save TensorRT Engine')
    save_engine(engine, trt_path)
    save_trt.end()
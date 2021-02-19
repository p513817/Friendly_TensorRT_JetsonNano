import tensorrt as trt
from log import PrintLog

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_path, shape = [1,224,224,3]):
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = (256 << 20)    # 256MiB
        builder.fp16_mode = True # fp32_mode -> False
        
        parser_onnx = PrintLog('Parser ONNX Model')
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        parser_onnx.end()

        build_trt = PrintLog('Build TensorRT Engine')
        engine = builder.build_cuda_engine(network)
        build_trt.end()
        return engine

def save_engine(engine, engine_path):
    save_engine = PrintLog('Save TensorRT Engine')
    buf = engine.serialize()
    with open(engine_path, 'wb') as f:
        f.write(buf)
    save_engine.end()

def load_engine(trt_runtime, engine_path):
    load_engine = PrintLog('Load TensorRT Engine')
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    load_engine.end()
    return engine

if __name__ == "__main__":
    
    onnx_path = 'alexnet.onnx'
    trt_path = 'alexnet.trt'
    input_shape = [1, 224, 224, 3]
    
    engine = build_engine(onnx_path, input_shape)
    
    save_engine(engine, trt_path)
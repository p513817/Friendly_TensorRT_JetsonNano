import onnx

# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the IR is well formed
print(onnx.checker.check_model(model))

print('-'*50)

# Print a human readable representation of the graph
print('Model :\n\n{}'.format(onnx.helper.printable_graph(model.graph)))


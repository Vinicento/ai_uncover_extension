import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("../extension_files/model.onnx")

# Create some input data (adjust the shape as per your model's requirements)
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)  # Example shape

# Run the model (adjust the input name as per your model's input)
inputs = {session.get_inputs()[0].name: input_data}
outputs = session.run(None, inputs)

# Print the output
print(outputs)

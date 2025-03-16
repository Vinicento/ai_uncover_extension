async function runModel() {
    try {
        // Create an ONNX session with WebGL backend.
        const session = new onnx.InferenceSession({ backendHint: 'webgl' });

        // Load the ONNX model. Adjust the path to where your model is located.
        await session.loadModel('path_to_your_model.onnx');

        // Generate random input data.
        // The shape and type should match what your model expects.
        // For example, a model with an input shape of [1, 3, 32, 32] (batch size, channels, height, width).
        const inputData = new Float32Array(1 * 3 * 32 * 32).fill().map(() => Math.random());
        const tensor = new onnx.Tensor(inputData, 'float32', [1, 3, 32, 32]);

        // Run the model with the generated input.
        const outputMap = await session.run([tensor]);

        // Assuming model has one output.
        const outputData = outputMap.values().next().value;

        // Print the prediction results.
        console.log('Model prediction:', outputData.data);
    } catch (error) {
        console.error('Failed to run model:', error);
    }
}

// Call the function to run the model.
runModel();

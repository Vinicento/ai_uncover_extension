document.addEventListener('DOMContentLoaded', async function () {
    const params = new URLSearchParams(window.location.search);
    const imageUrl = params.get('image');
    let session;
    let modelLoaded = false;

    document.getElementById('image').src = imageUrl; // Display the image

    async function loadModel() {
        try {
            session = new onnx.InferenceSession();
            await session.loadModel("model.onnx");
            modelLoaded = true;
        } catch (error) {
            console.error("Model Loading Error:", error);
            document.getElementById('result').textContent = 'Model loading failed.';
        }
    }

    function normalize(data, mean, std) {
        return data.map((value, index) => {
            const channel = index % 3;
            return (value - mean[channel]) / std[channel];
        });
    }

    if (imageUrl) {
        const img = new Image();
        img.crossOrigin = "Anonymous"; // Prevent CORS errors
        img.onload = async () => {
            try {
                if (!modelLoaded) {
                    await loadModel();
                }

                // Resize the image to 256x256 pixels
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 256;
                canvas.height = 256;
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                let data = Float32Array.from(imageData.data, value => value / 255).filter((_, index) => index % 4 !== 3);

                // Normalize the data
                const mean = [0.485, 0.456, 0.406];
                const std = [0.229, 0.224, 0.225];
                data = normalize(data, mean, std);

                // Prepare the tensor
                const tensor = new onnx.Tensor(new Float32Array(data), 'float32', [1, 3, 256, 256]);

                const output = await session.run([tensor]);
                const outputTensor = output.values().next().value;
                const manipulatedProbability = outputTensor.data[1];
                document.getElementById('result').textContent = `Probability of Image Being Ai Generated: ${(manipulatedProbability * 100).toFixed(2)}%`;

            } catch (error) {
                document.getElementById('result').textContent = 'Error processing image.';
            }
        };

        img.onerror = () => {
            document.getElementById('result').textContent = 'Failed to load image.';
        };

        img.src = imageUrl;
    } else {
        document.getElementById('result').textContent = 'Invalid or no image URL provided.';
    }
});

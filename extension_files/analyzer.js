
const polishHints = [
    "Obraz może zawierać nienaturalne szczegóły, np. zniekształcone dłonie.",
    "Wygenerowane twarze mogą mieć asymetryczne oczy lub usta.",
    "Tekst na obrazach bywa rozmazany lub nielogiczny.",
    "Tło może wyglądać sztucznie lub być niespójne.",
    "Niektóre elementy mogą wyglądać jak przypadkowo sklejone.",
    "Skóra bywa nienaturalnie gładka, bez struktury.",
    "Cienie i odbicia mogą nie zgadzać się z oświetleniem.",
    "Biżuteria i akcesoria mogą mieć nierealne kształty.",
    "Ubrania mogą mieć nielogiczne wzory lub zniekształcenia.",
    "Oczy mogą mieć nieprawidłowe odbicia lub kształt.",
    "Jeśli cokolwiek budzi Twoje wątpliwości, uruchom wyszukiwanie wsteczne (reverse image search).",
    "Porównaj obraz z relacjami w zaufanych mediach lub na stronach fact-checkingowych.",
    "Sprawdź metadane pliku – brak informacji lub dziwne daty to sygnał ostrzegawczy.",
    "Zwróć uwagę, czy wszystkie obiekty rzucają cień zgodny z kierunkiem oświetlenia.",
    "Przybliż fragmenty kadru: nierówne krawędzie i artefakty kompresji mogą zdradzić montaż.",
    "Zweryfikuj, kto pierwszy udostępnił materiał i jaki może mieć interes w jego rozpowszechnianiu.",
    "Gdy to wideo, posłuchaj ścieżki audio – często jest zniekształcona lub niespójna z obrazem.",
    "Poproś niezależną osobę (np. znajomego lub eksperta) o ocenę materiału, zanim go udostępnisz.",
    "Wykorzystaj dostępne narzędzia do wykrywania deepfake’ów (np. od ośrodków badawczych).",
    "Nigdy nie udostępniaj podejrzanych treści bez wyraźnego zaznaczenia, że mogą być sfałszowane."

];

function getRandomHint() {
    return polishHints[Math.floor(Math.random() * polishHints.length)];
}

function startRotatingHints() {
    const hintElement = document.getElementById('hint');
    let index = Math.floor(Math.random() * polishHints.length);

    function showNextHint() {
        hintElement.textContent = `🔍 Wskazówka: ${polishHints[index]}`;
        index = (index + 1) % polishHints.length;
    }

    showNextHint(); // initial
    setInterval(showNextHint, 5000); // rotate every 5 seconds
}

function startRotatingHints() {
    const hintElement = document.getElementById('hint');
    let index = Math.floor(Math.random() * polishHints.length);

    function showNextHint() {
        hintElement.textContent = `🔍 Wskazówka: ${polishHints[index]}`;
        index = (index + 1) % polishHints.length;
    }

    showNextHint(); // initial
    setInterval(showNextHint, 5000); // rotate every 5 seconds
}


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
                const percent = (manipulatedProbability * 100).toFixed(2);
                document.getElementById('result').textContent = `Prawdopodobieństwo, że obraz został wygenerowany przez AI: ${percent}%`;

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
    startRotatingHints();

});

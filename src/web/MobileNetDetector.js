class MobileNetDetector {
    constructor() {
        this.MODEL_PATH = 'model/mobilenet/model.json';
        this.model = null;

        this.emotions = ['angry', 'contempt', 'happy', 'neutral', 'sad', 'suprise'];

        this.emotionMap = {
            'angry': 'anger',
            'contempt': 'contempt',
            'happy': 'happiness',
            'neutral': 'neutrality',
            'sad': 'sadness',
            'suprise': 'surprise'
        };
    }

    async loadModels() {
        console.log('[MobileNet] Loading MobileNetV2 model...');

        const LOAD_TIMEOUT = 60000;

        try {
            console.log('[MobileNet] Step 1: Checking TensorFlow.js...');
            console.log('[MobileNet] TF.js version:', tf.version.tfjs);
            console.log('[MobileNet] Current backend:', tf.getBackend());

            console.log('[MobileNet] Step 2: Attempting WebGL backend...');
            let backend = 'cpu';

            try {
                await tf.setBackend('webgl');
                await tf.ready();
                backend = tf.getBackend();

                if (backend === 'webgl') {
                    console.log('[MobileNet] ✓ WebGL active - GPU acceleration enabled!');
                } else {
                    throw new Error('WebGL initialization failed');
                }
            } catch (e) {
                console.warn('[MobileNet] WebGL unavailable, using CPU fallback');
                console.warn('[MobileNet] Reason:', e.message);
                await tf.setBackend('cpu');
                await tf.ready();
                backend = 'cpu';
                console.log('[MobileNet] ✓ CPU backend active');
            }

            console.log('[MobileNet] Step 3: Running on:', backend);

            console.log('[MobileNet] Step 4: Loading model...');
            console.log('[MobileNet] Path:', this.MODEL_PATH);

            const loadPromise = tf.loadGraphModel(this.MODEL_PATH);
            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Timeout: Model loading took >30s')), LOAD_TIMEOUT)
            );

            this.model = await Promise.race([loadPromise, timeoutPromise]);

            console.log('[MobileNet] ✓ Model loaded!');
            console.log('[MobileNet] Inputs:', this.model.inputs.map(i => i.name));
            console.log('[MobileNet] Outputs:', this.model.outputs.map(o => o.name));

            console.log('[MobileNet] Warming up...');
            const dummy = tf.zeros([1, 224, 224, 3]);
            const result = this.model.predict(dummy);
            result.dispose();
            dummy.dispose();
            console.log('[MobileNet] ✓ Ready!');

        } catch (error) {
            console.error('[MobileNet] ❌ Load failed:', error.message);
            console.error('[MobileNet] Stack:', error.stack);
            throw new Error(`Model load failed: ${error.message}`);
        }
    }

    async detectEmotion(videoElement) {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModels() first.');
        }

        try {
            console.log('[MobileNet] Starting detection...');

            const tensor = tf.tidy(() => {
                const img = tf.browser.fromPixels(videoElement);
                console.log('[MobileNet] Input image shape:', img.shape);

                const resized = tf.image.resizeBilinear(img, [224, 224]);
                console.log('[MobileNet] Resized shape:', resized.shape);

                const normalized = resized.div(127.5).sub(1.0);

                const batched = normalized.expandDims(0);
                console.log('[MobileNet] Final tensor shape:', batched.shape);
                return batched;
            });

            console.log('[MobileNet] Running inference...');

            const predictionTensor = this.model.execute(tensor);
            console.log('[MobileNet] Prediction tensor type:', predictionTensor.constructor.name);

            const predictions = await predictionTensor.data();
            console.log('[MobileNet] Raw predictions:', Array.from(predictions));
            console.log('[MobileNet] Predictions sum:', predictions.reduce((a, b) => a + b, 0));

            tensor.dispose();
            predictionTensor.dispose();

            let maxConfidence = 0;
            let maxIndex = 0;
            for (let i = 0; i < predictions.length; i++) {
                if (predictions[i] > maxConfidence) {
                    maxConfidence = predictions[i];
                    maxIndex = i;
                }
            }

            const emotion = this.emotions[maxIndex];
            const mappedEmotion = this.emotionMap[emotion];

            console.log(`[MobileNet] Detected: ${emotion} -> ${mappedEmotion} (${(maxConfidence * 100).toFixed(2)}%)`);
            console.log('[MobileNet] All scores:', this.emotions.map((e, i) => `${e}: ${(predictions[i] * 100).toFixed(2)}%`).join(', '));

            return {
                emotion: mappedEmotion,
                confidence: maxConfidence,
                allScores: Object.fromEntries(
                    this.emotions.map((e, i) => [this.emotionMap[e], predictions[i]])
                )
            };

        } catch (error) {
            console.error('[MobileNet] Error during emotion detection:', error);
            return null;
        }
    }
}

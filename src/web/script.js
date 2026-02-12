if ('caches' in window) {
    caches.keys().then(names => {
        names.forEach(name => caches.delete(name));
    });
}

const CONFIG = {
    model: 'mobilenet',
    inferenceRate: 8
};

const START_BTN = document.getElementById('start-btn');
const STATUS_OVERLAY = document.getElementById('status-overlay');
const EMOTION_LABEL = document.getElementById('emotion-label');
const CONFIDENCE_TEXT = document.getElementById('confidence-text');
const CONFIDENCE_BAR = document.getElementById('confidence-bar');
const VIDEO_ELEMENT = document.getElementById('webcam');
const FPS_COUNTER = document.getElementById('fps-counter');

const EMOTION_COLORS = {
    anger: '#FF4B2B',
    contempt: '#9B59B6',
    happiness: '#FDC830',
    neutrality: '#E0E0E0',
    sadness: '#0083B0',
    surprise: '#00f2fe',
    fear: '#8E2DE2'
};

let isRunning = false;
let emotionDetector = null;
let detectionInterval = null;
let lastInferenceTime = 0;

let fpsFrames = [];
let lastFpsUpdate = 0;

const SMOOTHING_CONFIG = {
    confidenceThreshold: 0.60,
    debounceFrames: 3,
    historySize: 5
};

let emotionHistory = [];
let currentEmotion = null;
let emotionCounter = {};

async function loadModels() {
    try {
        EMOTION_LABEL.innerText = `Loading ${CONFIG.model} models...`;

        emotionDetector = await EmotionDetectorFactory.create(CONFIG);

        console.log(`✓ ${CONFIG.model} detector loaded successfully`);
        return true;
    } catch (error) {
        console.error("Model loading error:", error);
        EMOTION_LABEL.innerText = "Model Load Failed";

        console.error("Full error details:", {
            message: error.message,
            stack: error.stack,
            backend: typeof tf !== 'undefined' ? tf.getBackend() : 'tf not loaded'
        });

        alert(`Failed to load ${CONFIG.model} models: ` + error.message);
        return false;
    }
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        VIDEO_ELEMENT.srcObject = stream;
        await VIDEO_ELEMENT.play();

        console.log("✓ Camera started");
        return true;
    } catch (error) {
        console.error("Camera error:", error);
        alert("Camera access failed: " + error.message);
        return false;
    }
}

async function detectEmotions() {
    if (!isRunning) {
        console.log('[Script] Detection stopped - isRunning=false');
        return;
    }

    const now = performance.now();
    const timeSinceLastInference = now - lastInferenceTime;
    const minInterval = 1000 / CONFIG.inferenceRate;

    if (timeSinceLastInference < minInterval) {
        return;
    }

    try {
        console.log('[Script] Running detection cycle...');
        lastInferenceTime = now;

        const result = await emotionDetector.detectEmotion(VIDEO_ELEMENT);

        if (result) {
            console.log('[Script] Detection result:', result);

            const smoothed = smoothEmotion(result.emotion, result.confidence);

            if (smoothed.emotion) {
                updateUI(smoothed.emotion, smoothed.confidence, smoothed.changed);
            }
        } else {
            console.warn('[Script] Detection returned null');
        }

        const inferenceTime = performance.now() - now;
        console.log(`[Script] Inference took ${inferenceTime.toFixed(1)}ms`);
        if (inferenceTime > minInterval * 0.8) {
            console.warn(`[Script] Inference slow: ${inferenceTime.toFixed(1)}ms (target: ${minInterval}ms)`);
        }

        updateFPS(inferenceTime);

    } catch (error) {
        console.error("[Script] Detection error:", error);
    }
}

function smoothEmotion(rawEmotion, rawConfidence) {
    emotionHistory.push({ emotion: rawEmotion, confidence: rawConfidence });

    if (emotionHistory.length > SMOOTHING_CONFIG.historySize) {
        emotionHistory.shift();
    }

    if (rawConfidence < SMOOTHING_CONFIG.confidenceThreshold) {
        return { emotion: currentEmotion, confidence: rawConfidence, changed: false };
    }

    if (!emotionCounter[rawEmotion]) {
        emotionCounter[rawEmotion] = 0;
    }
    emotionCounter[rawEmotion]++;

    for (let key in emotionCounter) {
        if (key !== rawEmotion) {
            emotionCounter[key] = 0;
        }
    }

    if (emotionCounter[rawEmotion] >= SMOOTHING_CONFIG.debounceFrames) {
        if (currentEmotion !== rawEmotion) {
            currentEmotion = rawEmotion;
            return { emotion: rawEmotion, confidence: rawConfidence, changed: true };
        }
    }

    return { emotion: currentEmotion || rawEmotion, confidence: rawConfidence, changed: false };
}

function updateFPS(inferenceTime) {
    const now = performance.now();
    fpsFrames.push(now);

    fpsFrames = fpsFrames.filter(time => now - time < 1000);

    if (now - lastFpsUpdate > 500) {
        const fps = fpsFrames.length;
        const avgInferenceTime = inferenceTime.toFixed(0);
        FPS_COUNTER.textContent = `FPS: ${fps} | Inference: ${avgInferenceTime}ms`;
        lastFpsUpdate = now;
    }
}

function updateUI(emotion, confidence, changed = false) {
    if (changed) {
        EMOTION_LABEL.style.opacity = '0';
        setTimeout(() => {
            EMOTION_LABEL.innerText = emotion.toUpperCase();
            EMOTION_LABEL.style.opacity = '1';
        }, 150);
    } else {
        EMOTION_LABEL.innerText = emotion.toUpperCase();
    }

    const confidencePercent = Math.round(confidence * 100);
    CONFIDENCE_TEXT.innerText = `Confidence: ${confidencePercent}%`;
    CONFIDENCE_BAR.style.width = `${confidencePercent}%`;

    const color = EMOTION_COLORS[emotion] || EMOTION_COLORS.neutrality;
    const darkerColor = adjustColorBrightness(color, -20);
    document.body.style.background = `linear-gradient(135deg, ${color} 0%, ${darkerColor} 100%)`;
}

function adjustColorBrightness(hex, percent) {
    hex = hex.replace('#', '');
    let r = parseInt(hex.substring(0, 2), 16);
    let g = parseInt(hex.substring(2, 4), 16);
    let b = parseInt(hex.substring(4, 6), 16);
    r = Math.max(0, Math.min(255, r + (r * percent / 100)));
    g = Math.max(0, Math.min(255, g + (g * percent / 100)));
    b = Math.max(0, Math.min(255, b + (b * percent / 100)));
    return `#${Math.round(r).toString(16).padStart(2, '0')}${Math.round(g).toString(16).padStart(2, '0')}${Math.round(b).toString(16).padStart(2, '0')}`;
}

START_BTN.addEventListener('click', async () => {
    START_BTN.disabled = true;
    START_BTN.innerText = "Loading...";

    const modelsLoaded = await loadModels();
    if (!modelsLoaded) {
        START_BTN.disabled = false;
        START_BTN.innerText = "Retry";
        return;
    }

    const cameraStarted = await startCamera();
    if (!cameraStarted) {
        START_BTN.disabled = false;
        START_BTN.innerText = "Retry";
        return;
    }

    document.getElementById('section-controls').style.display = 'none';
    STATUS_OVERLAY.style.display = 'block';

    isRunning = true;
    lastInferenceTime = 0;
    EMOTION_LABEL.innerText = "READY";

    const intervalMs = 1000 / CONFIG.inferenceRate;
    detectionInterval = setInterval(detectEmotions, intervalMs);

    console.log(`Detection started at ${CONFIG.inferenceRate} FPS (${intervalMs}ms interval)`);
});

function stopDetection() {
    isRunning = false;
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    console.log("Detection stopped");
}

window.addEventListener('beforeunload', stopDetection);

console.log("Emotion Detection App Initialized");
console.log(`Using detector: ${CONFIG.model}`);
console.log(`Inference rate: ${CONFIG.inferenceRate} FPS (${1000 / CONFIG.inferenceRate}ms interval)`);
console.log("Click 'Start Camera' to begin");

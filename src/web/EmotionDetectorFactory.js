class EmotionDetectorFactory {
    static async create(config) {
        console.log(`Creating detector: ${config.model}`);

        if (config.model === 'mobilenet') {
            const detector = new MobileNetDetector();
            await detector.loadModels();
            return detector;
        } else {
            throw new Error(`Unknown model type: ${config.model}`);
        }
    }
}

(async function () {
    console.log('[Init] Forcing WebGL backend for MobileNetV3...');

    let attempts = 0;
    while (typeof tf === 'undefined') {
        await new Promise(resolve => setTimeout(resolve, 100));
        attempts++;
        if (attempts > 50) {
            console.error('[Init] TensorFlow.js failed to load after 5 seconds');
            return;
        }
    }

    console.log('[Init] TensorFlow.js loaded after', attempts * 100, 'ms');
    console.log('[Init] TF.js version:', tf.version.tfjs);
    console.log('[Init] Current backend:', tf.getBackend());

    try {
        console.log('[Init] Attempting to set WebGL backend...');
        await tf.setBackend('webgl');
        console.log('[Init] setBackend called, waiting for ready...');
        await tf.ready();
        console.log('[Init] ✓ WebGL backend initialized:', tf.getBackend());
    } catch (error) {
        console.error('[Init] ❌ Failed to initialize WebGL:');
        console.error('[Init] Error message:', error.message);
        console.error('[Init] Error stack:', error.stack);
        console.error('[Init] Current backend after failure:', tf.getBackend());

        console.error('[Init] WebGL initialization failed - check webgl-diagnostic.js output');
    }
})();

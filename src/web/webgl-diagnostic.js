function checkWebGLSupport() {
    console.log('=== WebGL Diagnostic ===');

    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

    if (gl) {
        console.log('✓ WebGL context created successfully');
        console.log('WebGL Vendor:', gl.getParameter(gl.VENDOR));
        console.log('WebGL Renderer:', gl.getParameter(gl.RENDERER));
        console.log('WebGL Version:', gl.getParameter(gl.VERSION));
        console.log('GLSL Version:', gl.getParameter(gl.SHADING_LANGUAGE_VERSION));

        console.log('Max Texture Size:', gl.getParameter(gl.MAX_TEXTURE_SIZE));
        console.log('Max Viewport Dims:', gl.getParameter(gl.MAX_VIEWPORT_DIMS));

        return true;
    } else {
        console.error('✗ WebGL context creation failed');
        console.log('Browser:', navigator.userAgent);
        return false;
    }
}

function checkWebGL2Support() {
    const canvas = document.createElement('canvas');
    const gl2 = canvas.getContext('webgl2');

    if (gl2) {
        console.log('✓ WebGL2 also available');
        return true;
    } else {
        console.log('✗ WebGL2 not available (WebGL1 only)');
        return false;
    }
}

console.log('Running WebGL diagnostics...');
const hasWebGL = checkWebGLSupport();
const hasWebGL2 = checkWebGL2Support();

console.log('\\n=== TensorFlow.js Backend Info ===');
if (typeof tf !== 'undefined') {
    console.log('TensorFlow.js version:', tf.version.tfjs);
    console.log('Available backends:', tf.engine().registryFactory);
} else {
    console.log('TensorFlow.js not loaded yet');
}

console.log('========================\\n');

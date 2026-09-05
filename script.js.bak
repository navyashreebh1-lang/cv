/*
=============================================================================
AGRIVISION - Plant Detection System - Stage 1 Prototype
=============================================================================

ARCHITECTURE OVERVIEW:

This script implements a modular computer vision pipeline:

1. CAMERA ABSTRACTION
   - getUserMedia API wrapper for laptop webcam
   - Frame provider interface
   - Future: Can be replaced with ESP32-CAM stream without UI changes

2. DETECTION ENGINE ABSTRACTION
   - Unified detection interface
   - Real mode: TensorFlow.js + COCO-SSD for plant detection
   - Simulation mode: Realistic mock detector for testing/demo
   - Separate FPS from camera FPS

3. VISUALIZATION LAYER
   - Canvas-based bounding box rendering
   - Confidence score display
   - Multiple detection support
   - Real-time statistics

4. UI MANAGEMENT
   - Status updates
   - Control state management
   - Performance metrics

DESIGN PRINCIPLES:
- Clean separation of concerns
- Abstracted camera/detection for future hardware replacement
- Graceful error handling
- No global state pollution
- Comments explain "WHY" not "WHAT"

FUTURE EXTENSIBILITY:
- Plant species classification (Stage 2)
- Disease detection (Stage 3)
- GPS coordinate association (Stage 4)
- Digital twin integration (Stage 5)

=============================================================================
*/

// ============================================================================
// 1. CAMERA ABSTRACTION LAYER
// ============================================================================

const CameraManager = {
    video: null,
    stream: null,
    isActive: false,

    async init() {
        this.video = document.getElementById('cameraFeed');

        // Prefer a rear camera (for a future handheld/rover mount) but never
        // require one — most laptops only expose a front-facing webcam, and
        // a hard 'environment' constraint throws OverconstrainedError there.
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: { ideal: 'environment' }
                },
                audio: false
            });
        } catch (err) {
            console.warn('Preferred camera constraints failed, retrying with a basic video request:', err);
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            } catch (fallbackErr) {
                console.error('Camera initialization failed:', fallbackErr);
                UIManager.showError(`Camera Error: ${fallbackErr.message}`);
                return false;
            }
        }

        this.video.srcObject = this.stream;
        return true;
    },

    async start() {
        if (this.isActive) return;
        try {
            await this.video.play();
            this.isActive = true;
            return true;
        } catch (err) {
            console.error('Failed to start camera:', err);
            return false;
        }
    },

    stop() {
        if (!this.isActive) return;
        this.video.pause();
        this.isActive = false;
    },

    release() {
        this.stop();
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    },

    getFrame() {
        if (!this.isActive || !this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
            return null;
        }
        return this.video;
    },

    getCanvasFrame() {
        const frame = this.getFrame();
        if (!frame) return null;

        const canvas = document.createElement('canvas');
        canvas.width = frame.videoWidth;
        canvas.height = frame.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(frame, 0, 0);
        return canvas;
    }
};

// ============================================================================
// 2. DETECTION ENGINE ABSTRACTION LAYER
// ============================================================================

// Custom plant detector (YOLOv8, trained via training/train_plant_detector.ipynb
// and exported to TF.js). Loaded from model/model.json when present; the app
// falls back to COCO-SSD when it isn't (e.g. before training has been run).
const CustomModel = {
    model: null,
    classNames: ['plant'],
    imgsz: 640,
    scoreThreshold: 0.4,
    iouThreshold: 0.45,

    async load() {
        try {
            const metaRes = await fetch('model/metadata.json');
            if (metaRes.ok) {
                const meta = await metaRes.json();
                if (meta.names) this.classNames = meta.names;
                if (meta.imgsz) this.imgsz = meta.imgsz;
            }
            this.model = await tf.loadGraphModel('model/model.json');
            return true;
        } catch (err) {
            console.log('No custom model found (model/model.json) — will use COCO-SSD:', err.message);
            return false;
        }
    },

    async detect(frame) {
        const srcW = frame.videoWidth || frame.width;
        const srcH = frame.videoHeight || frame.height;

        const input = tf.tidy(() => tf.browser.fromPixels(frame)
            .resizeBilinear([this.imgsz, this.imgsz])
            .div(255.0)
            .expandDims(0));

        let output = await this.model.executeAsync(input);
        input.dispose();
        if (Array.isArray(output)) output = output[0];

        const { boxesXYXY, scores, classIds } = tf.tidy(() => {
            // YOLOv8 output: [1, 4 + numClasses, numAnchors] -> [numAnchors, 4 + numClasses]
            const transposed = output.transpose([0, 2, 1]).squeeze([0]);
            const numClasses = this.classNames.length;
            const boxesXYWH = transposed.slice([0, 0], [-1, 4]);
            const classScores = transposed.slice([0, 4], [-1, numClasses]);

            const x = boxesXYWH.slice([0, 0], [-1, 1]);
            const y = boxesXYWH.slice([0, 1], [-1, 1]);
            const w = boxesXYWH.slice([0, 2], [-1, 1]);
            const h = boxesXYWH.slice([0, 3], [-1, 1]);

            return {
                boxesXYXY: tf.concat([
                    y.sub(h.div(2)), x.sub(w.div(2)),
                    y.add(h.div(2)), x.add(w.div(2))
                ], 1),
                scores: classScores.max(1),
                classIds: classScores.argMax(1)
            };
        });

        const nmsIndices = await tf.image.nonMaxSuppressionAsync(
            boxesXYXY, scores, 50, this.iouThreshold, this.scoreThreshold
        );

        const [boxesData, scoresData, classIdsData, keepIndices] = await Promise.all([
            boxesXYXY.array(), scores.array(), classIds.array(), nmsIndices.array()
        ]);

        const scaleX = srcW / this.imgsz;
        const scaleY = srcH / this.imgsz;

        const detections = keepIndices.map(i => {
            const [y1, x1, y2, x2] = boxesData[i];
            return {
                className: this.classNames[classIdsData[i]] || 'plant',
                confidence: (scoresData[i] * 100).toFixed(1),
                boundingBox: {
                    x: x1 * scaleX,
                    y: y1 * scaleY,
                    width: (x2 - x1) * scaleX,
                    height: (y2 - y1) * scaleY
                }
            };
        });

        tf.dispose([output, boxesXYXY, scores, classIds, nmsIndices]);
        return detections;
    }
};

const DetectionEngine = {
    mode: 'real', // 'real' or 'simulation'
    engineType: null, // 'custom' or 'coco-ssd'
    model: null,
    isLoading: false,
    isReady: false,
    lastFrameTime: 0,
    inferenceTime: 0,

    async init() {
        this.isLoading = true;
        try {
            if (typeof tf !== 'undefined' && await CustomModel.load()) {
                this.engineType = 'custom';
                this.isReady = true;
                console.log('Custom plant detector loaded successfully');
                return true;
            }

            // Fall back to the generic COCO-SSD model
            if (typeof cocoSsd === 'undefined') {
                throw new Error('TensorFlow.js models not loaded');
            }
            this.model = await cocoSsd.load();
            this.engineType = 'coco-ssd';
            this.isReady = true;
            console.log('Detection model loaded successfully');
            return true;
        } catch (err) {
            console.error('Model loading failed:', err);
            console.log('Falling back to simulation mode');
            this.isReady = false;
            return false;
        } finally {
            this.isLoading = false;
        }
    },

    setMode(simulationMode) {
        this.mode = simulationMode ? 'simulation' : 'real';
    },

    async detectFrame(frame) {
        if (this.mode === 'simulation') {
            return this.simulateDetection(frame);
        }
        return this.realDetection(frame);
    },

    async realDetection(frame) {
        if (!this.isReady) {
            return [];
        }

        const startTime = performance.now();

        if (this.engineType === 'custom') {
            try {
                const results = await CustomModel.detect(frame);
                this.inferenceTime = performance.now() - startTime;
                return results;
            } catch (err) {
                console.error('Detection error:', err);
                return [];
            }
        }

        if (!this.model) {
            return [];
        }

        try {
            // Use COCO-SSD for plant detection
            const predictions = await this.model.detect(frame);

            // Filter for plant-like objects
            // COCO classes: 58=potted plant, 59=plant, 60=flowers, etc.
            const plantClasses = ['potted plant', 'plant', 'flowers', 'vase'];
            const plantDetections = predictions.filter(pred => {
                const className = pred.class.toLowerCase();
                return plantClasses.some(pc => className.includes(pc));
            });

            // If no specific plant classes, accept confidence > 0.6 for any object
            // (plant detection is Stage 1, so we're lenient here)
            const allWithHighConfidence = predictions.filter(pred => pred.score > 0.6);
            const results = plantDetections.length > 0 ? plantDetections : allWithHighConfidence;

            this.inferenceTime = performance.now() - startTime;

            return results.map(detection => ({
                className: detection.class || 'Plant',
                confidence: (detection.score * 100).toFixed(1),
                boundingBox: {
                    x: detection.bbox[0],
                    y: detection.bbox[1],
                    width: detection.bbox[2],
                    height: detection.bbox[3]
                }
            }));
        } catch (err) {
            console.error('Detection error:', err);
            return [];
        }
    },

    // Simulation mode - generates realistic mock detections
    // Used when real model isn't available or for development/demo
    simulateDetection(frame) {
        const startTime = performance.now();

        const width = frame.videoWidth || frame.width || 640;
        const height = frame.videoHeight || frame.height || 480;

        // Generate 1-3 random plants per frame for realistic testing
        const plantCount = Math.floor(Math.random() * 3);
        const detections = [];

        for (let i = 0; i < plantCount; i++) {
            // Random position in frame
            const plantWidth = 80 + Math.random() * 120;
            const plantHeight = 100 + Math.random() * 150;
            const x = Math.random() * (width - plantWidth);
            const y = Math.random() * (height - plantHeight);

            // High confidence for simulation
            const confidence = 85 + Math.random() * 15;

            detections.push({
                className: 'Plant',
                confidence: confidence.toFixed(1),
                boundingBox: {
                    x: x,
                    y: y,
                    width: plantWidth,
                    height: plantHeight
                }
            });
        }

        // Simulate realistic inference time (30-50ms)
        this.inferenceTime = 30 + Math.random() * 20;

        return detections;
    }
};

// ============================================================================
// 3. VISUALIZATION LAYER
// ============================================================================

const Visualizer = {
    canvas: null,
    ctx: null,
    scale: 1.0,

    init() {
        this.canvas = document.getElementById('detectionCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    },

    resizeCanvas() {
        const video = document.getElementById('cameraFeed');
        if (video.videoWidth && video.videoHeight) {
            this.canvas.width = video.videoWidth;
            this.canvas.height = video.videoHeight;

            const wrapper = document.querySelector('.camera-wrapper');
            const wrapperWidth = wrapper.offsetWidth;
            const wrapperHeight = wrapper.offsetHeight;

            this.scale = Math.min(
                wrapperWidth / this.canvas.width,
                wrapperHeight / this.canvas.height
            );
        }
    },

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    },

    drawDetections(detections) {
        this.clear();

        detections.forEach((detection, index) => {
            const box = detection.boundingBox;

            // Draw bounding box
            this.ctx.strokeStyle = '#10b981'; // Primary green
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(box.x, box.y, box.width, box.height);

            // Draw background for text
            const labelText = `${detection.className} ${detection.confidence}%`;
            this.ctx.font = 'bold 14px Arial';
            const textWidth = this.ctx.measureText(labelText).width;
            const textHeight = 24;

            this.ctx.fillStyle = 'rgba(16, 185, 129, 0.9)';
            this.ctx.fillRect(box.x, box.y - textHeight - 5, textWidth + 10, textHeight);

            // Draw text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(labelText, box.x + 5, box.y - textHeight / 2 - 5);

            // Draw corner markers for aesthetic
            const cornerSize = 10;
            this.ctx.strokeStyle = '#06b6d4';
            this.ctx.lineWidth = 2;

            // Top-left
            this.ctx.strokeRect(box.x, box.y, cornerSize, cornerSize);
            // Top-right
            this.ctx.strokeRect(box.x + box.width - cornerSize, box.y, cornerSize, cornerSize);
            // Bottom-left
            this.ctx.strokeRect(box.x, box.y + box.height - cornerSize, cornerSize, cornerSize);
            // Bottom-right
            this.ctx.strokeRect(box.x + box.width - cornerSize, box.y + box.height - cornerSize, cornerSize, cornerSize);
        });
    }
};

// ============================================================================
// 4. PERFORMANCE MONITORING
// ============================================================================

const PerformanceMonitor = {
    cameraFrames: 0,
    cameraLastTime: performance.now(),
    inferenceFrames: 0,
    inferenceLastTime: performance.now(),

    recordCameraFrame() {
        this.cameraFrames++;
        const now = performance.now();
        if (now - this.cameraLastTime >= 1000) {
            StatsManager.cameraFps = this.cameraFrames;
            this.cameraFrames = 0;
            this.cameraLastTime = now;
        }
    },

    recordInferenceFrame() {
        this.inferenceFrames++;
        const now = performance.now();
        if (now - this.inferenceLastTime >= 1000) {
            StatsManager.inferenceFps = this.inferenceFrames;
            this.inferenceFrames = 0;
            this.inferenceLastTime = now;
        }
    }
};

// ============================================================================
// 5. STATISTICS & STATE MANAGEMENT
// ============================================================================

const StatsManager = {
    detections: [],
    cameraFps: 0,
    inferenceFps: 0,
    inferenceTime: 0,

    updateDetections(detections) {
        this.detections = detections;
        this.inferenceTime = DetectionEngine.inferenceTime;
    },

    getStats() {
        if (this.detections.length === 0) {
            return {
                plantCount: 0,
                highestConfidence: '--',
                averageConfidence: '--'
            };
        }

        const confidences = this.detections.map(d => parseFloat(d.confidence));
        const highest = Math.max(...confidences);
        const average = (confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(1);

        return {
            plantCount: this.detections.length,
            highestConfidence: highest.toFixed(1),
            averageConfidence: average
        };
    }
};

// ============================================================================
// 6. UI MANAGEMENT
// ============================================================================

const UIManager = {
    state: {
        cameraRunning: false,
        detectionActive: false,
        simulationMode: false,
        cameraReady: false,
        modelReady: false
    },

    init() {
        this.setupEventListeners();
        this.updateAllStatus();
    },

    setupEventListeners() {
        document.getElementById('startCameraBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopCameraBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('startDetectionBtn').addEventListener('click', () => this.startDetection());
        document.getElementById('stopDetectionBtn').addEventListener('click', () => this.stopDetection());
        document.getElementById('simulationModeToggle').addEventListener('change', (e) => this.toggleSimulation(e));
    },

    async startCamera() {
        if (!this.state.cameraReady) {
            const success = await CameraManager.init();
            if (!success) return;
            this.state.cameraReady = true;
        }

        const started = await CameraManager.start();
        if (started) {
            this.state.cameraRunning = true;
            this.updateAllStatus();
            this.hideOverlay();

            // Start animation loop
            this.animationLoop();
        }
    },

    stopCamera() {
        CameraManager.stop();
        this.state.cameraRunning = false;
        this.state.detectionActive = false;
        this.updateAllStatus();
        Visualizer.clear();
        this.showOverlay('Camera Stopped');
    },

    async startDetection() {
        if (!this.state.cameraRunning) {
            this.showError('Start camera first');
            return;
        }

        if (!this.state.modelReady && !DetectionEngine.mode === 'simulation') {
            this.showError('AI model not ready. Enable Simulation Mode.');
            return;
        }

        this.state.detectionActive = true;
        this.updateAllStatus();
    },

    stopDetection() {
        this.state.detectionActive = false;
        this.updateAllStatus();
        Visualizer.clear();
        StatsManager.detections = [];
        this.updateDetectionUI();
    },

    toggleSimulation(e) {
        this.state.simulationMode = e.target.checked;
        DetectionEngine.setMode(this.state.simulationMode);
        this.updateAllStatus();
    },

    animationLoop() {
        if (!this.state.cameraRunning) return;

        PerformanceMonitor.recordCameraFrame();

        if (this.state.detectionActive) {
            const frame = CameraManager.getFrame();
            if (frame && frame.readyState === frame.HAVE_ENOUGH_DATA) {
                this.runDetection(frame);
            }
        }

        requestAnimationFrame(() => this.animationLoop());
    },

    async runDetection(frame) {
        const detections = await DetectionEngine.detectFrame(frame);
        StatsManager.updateDetections(detections);
        PerformanceMonitor.recordInferenceFrame();

        Visualizer.drawDetections(detections);
        this.updateDetectionUI();
        this.updateStats();
    },

    updateAllStatus() {
        // Header mode badge
        const modeBadge = document.getElementById('modeBadge');
        if (this.state.simulationMode) {
            modeBadge.textContent = 'SIMULATION MODE';
            modeBadge.classList.add('simulation');
        } else {
            modeBadge.textContent = 'LIVE MODE';
            modeBadge.classList.remove('simulation');
        }

        // Status indicator
        const indicator = document.getElementById('statusIndicator');
        const statusText = indicator.querySelector('.status-text');
        const statusDot = indicator.querySelector('.status-dot');

        if (this.state.detectionActive) {
            statusText.textContent = 'DETECTING';
            statusDot.classList.add('online');
        } else if (this.state.cameraRunning) {
            statusText.textContent = 'CAMERA READY';
            statusDot.classList.add('online');
        } else {
            statusText.textContent = 'STANDBY';
            statusDot.classList.remove('online');
        }

        // Button states
        const startCameraBtn = document.getElementById('startCameraBtn');
        const stopCameraBtn = document.getElementById('stopCameraBtn');
        const startDetectionBtn = document.getElementById('startDetectionBtn');
        const stopDetectionBtn = document.getElementById('stopDetectionBtn');

        startCameraBtn.disabled = this.state.cameraRunning;
        stopCameraBtn.disabled = !this.state.cameraRunning;
        startDetectionBtn.disabled = !this.state.cameraRunning || this.state.detectionActive;
        stopDetectionBtn.disabled = !this.state.detectionActive;

        // System status panel
        document.getElementById('cameraStatus').textContent = this.state.cameraRunning ? 'ONLINE' : 'OFFLINE';
        document.getElementById('cameraStatus').className = this.state.cameraRunning ? 'status-badge online' : 'status-badge';

        document.getElementById('aiStatus').textContent = this.state.simulationMode ? 'SIMULATION' : (this.state.modelReady ? 'READY' : 'LOADING');
        document.getElementById('aiStatus').className = this.state.modelReady || this.state.simulationMode ? 'status-badge ready' : 'status-badge loading';

        document.getElementById('detectionStatus').textContent = this.state.detectionActive ? 'ACTIVE' : 'PAUSED';
        document.getElementById('detectionStatus').className = this.state.detectionActive ? 'status-badge active' : 'status-badge paused';

        document.getElementById('modeStatus').textContent = this.state.simulationMode ? 'SIMULATION' : 'LIVE';
        document.getElementById('modeStatus').className = this.state.simulationMode ? 'status-badge simulation' : 'status-badge';

        const modelLabel = DetectionEngine.engineType === 'custom' ? 'CUSTOM' : 'COCO-SSD';
        document.getElementById('modelValue').textContent = this.state.simulationMode ? 'MOCK' : (this.state.modelReady ? modelLabel : '--');
    },

    updateDetectionUI() {
        const stats = StatsManager.getStats();
        const detectionSummary = document.getElementById('detectionSummary');
        const detectionsList = document.getElementById('detectionsList');

        // Update summary indicator
        const indicator = detectionSummary.querySelector('.summary-indicator');
        if (stats.plantCount > 0) {
            indicator.classList.remove('offline');
            indicator.classList.add('detected');
            indicator.querySelector('.indicator-text').textContent = `🟢 ${stats.plantCount} PLANT${stats.plantCount > 1 ? 'S' : ''} DETECTED`;
        } else {
            indicator.classList.add('offline');
            indicator.classList.remove('detected');
            indicator.querySelector('.indicator-text').textContent = '⚪ NO PLANT DETECTED';
        }

        // Update statistics
        document.getElementById('plantCountValue').textContent = stats.plantCount;
        document.getElementById('highestConfValue').textContent = stats.highestConfidence + '%';
        document.getElementById('avgConfValue').textContent = stats.averageConfidence + '%';

        // Update detections list
        if (stats.plantCount === 0) {
            detectionsList.innerHTML = '<div class="empty-state">No plants detected</div>';
        } else {
            detectionsList.innerHTML = StatsManager.detections.map((det, idx) => `
                <div class="detection-item">
                    <span class="detection-name">Plant ${idx + 1}</span>
                    <span class="detection-confidence">${det.confidence}%</span>
                </div>
            `).join('');
        }
    },

    updateStats() {
        document.getElementById('cameraFpsValue').textContent = PerformanceMonitor.cameraFrames + StatsManager.cameraFps;
        document.getElementById('inferenceFpsValue').textContent = StatsManager.inferenceFps;
        document.getElementById('inferenceTimeValue').textContent = StatsManager.inferenceTime.toFixed(1) + ' ms';
    },

    hideOverlay() {
        const overlay = document.getElementById('cameraOverlay');
        overlay.classList.add('hidden');
    },

    showOverlay(message = '') {
        const overlay = document.getElementById('cameraOverlay');
        if (message) {
            overlay.querySelector('.overlay-message p').textContent = message;
        }
        overlay.classList.remove('hidden');
    },

    showError(message) {
        console.error(message);
        // A blocking alert() would freeze the whole tab (including any
        // automated/devtools session watching it) until manually dismissed —
        // show the error inline in the camera overlay instead.
        this.showOverlay(message);
        const spinner = document.querySelector('#cameraOverlay .spinner');
        if (spinner) spinner.style.display = 'none';
    }
};

// ============================================================================
// 7. INITIALIZATION
// ============================================================================

async function initializeApp() {
    console.log('Initializing AGRIVISION Plant Detection System...');

    // Initialize UI
    UIManager.init();
    Visualizer.init();

    // Initialize detection model
    console.log('Loading AI model...');
    const modelReady = await DetectionEngine.init();
    UIManager.state.modelReady = modelReady;
    UIManager.updateAllStatus();

    console.log('AGRIVISION Plant Detection System initialized');
    console.log('Ready for plant detection - click START CAMERA to begin');
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

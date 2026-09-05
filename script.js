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
// 0. DEBUG LOGGING + TIMEOUT UTILITY
// ============================================================================

// Every step of model load / inference logs through here with a consistent
// prefix so it's easy to filter in devtools (Console -> filter "[AGRIVISION]")
// while diagnosing why detection isn't producing results.
function debugLog(...args) {
    console.log('[AGRIVISION]', ...args);
}

// tfjs-tflite's WASM backend has a known failure mode where, without
// cross-origin isolation (COOP/COEP headers - which a plain `python -m
// http.server` never sends), the multithreaded WASM path can hang instead of
// rejecting cleanly. An unresolved promise there would leave the whole app
// stuck on "LOADING" forever with no error, which is worse than a fast,
// visible failure. Race every model load against a hard timeout so the app
// always reaches a definite READY/ERROR state.
function withTimeout(promise, ms, label) {
    return Promise.race([
        promise,
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
        )
    ]);
}

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
        if (!this.isActive || this.video.readyState !== this.video.HAVE_ENOUGH_DATA) {
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
            debugLog('Loading custom model metadata (model/metadata.json)...');
            const metaRes = await fetch('model/metadata.json');
            if (metaRes.ok) {
                const meta = await metaRes.json();
                if (meta.names) this.classNames = meta.names;
                if (meta.imgsz) this.imgsz = meta.imgsz;
            }
            if (typeof tflite === 'undefined') {
                throw new Error('tfjs-tflite not loaded');
            }

            // Multithreaded WASM requires cross-origin isolation (COOP/COEP
            // response headers) to use SharedArrayBuffer. `python -m
            // http.server` never sends those, so requesting multiple threads
            // here would silently degrade to a hang in some browsers rather
            // than a clean error. Only ask for >1 thread when the page is
            // actually cross-origin-isolated.
            const isolated = typeof self !== 'undefined' && self.crossOriginIsolated === true;
            const numThreads = isolated ? (navigator.hardwareConcurrency || 4) : 1;
            debugLog(`crossOriginIsolated=${isolated}, using numThreads=${numThreads}`);

            // NOTE: verified by direct request that jsDelivr's exact-version path
            // (.../tfjs-tflite@0.0.1-alpha.10/dist/*.wasm) 404s for every WASM
            // binary in this package - the npm tarball's version-pinned file
            // listing on jsDelivr doesn't include them, even though the same JS
            // bundle IS correctly served at that pinned version. The *unversioned*
            // path, confirmed by inspecting response headers/magic bytes, serves
            // real WASM binaries. So the <script> tag is pinned (safe - same JS
            // either way) but the WASM base path deliberately isn't.
            tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');
            debugLog('Loading model/plant_detector.tflite ...');
            this.model = await withTimeout(
                tflite.loadTFLiteModel('model/plant_detector.tflite', { numThreads }),
                10000,
                'Custom TFLite model load'
            );
            debugLog('Custom model loaded. Input imgsz =', this.imgsz, 'classes =', this.classNames);
            return true;
        } catch (err) {
            console.error('[AGRIVISION] Custom model load failed - falling back to COCO-SSD.');
            console.error('[AGRIVISION] error.name:', err.name);
            console.error('[AGRIVISION] error.message:', err.message);
            console.error('[AGRIVISION] error.stack:', err.stack);
            return false;
        }
    },

    async detect(frame) {
        const srcW = frame.videoWidth || frame.width;
        const srcH = frame.videoHeight || frame.height;

        // Letterbox (resize preserving aspect ratio, pad the rest with gray)
        // instead of a naive stretch to square. Ultralytics' own train/val/export
        // pipeline always letterboxes — a plain stretch-resize of a 16:9 webcam
        // frame into a square 640x640 distorts plant shapes in a way the model
        // never saw during training, which costs real recall and box accuracy.
        const scale = Math.min(this.imgsz / srcW, this.imgsz / srcH);
        const newW = Math.round(srcW * scale);
        const newH = Math.round(srcH * scale);
        const padX = Math.floor((this.imgsz - newW) / 2);
        const padY = Math.floor((this.imgsz - newH) / 2);
        const LETTERBOX_FILL = 114 / 255; // matches Ultralytics' gray (114,114,114) pad value

        // The exported model expects channels-first (NCHW) input, matching
        // Ultralytics' native TFLite export layout — [1, 3, imgsz, imgsz].
        const input = tf.tidy(() => tf.browser.fromPixels(frame)
            .resizeBilinear([newH, newW])
            .div(255.0)
            .pad([[padY, this.imgsz - newH - padY], [padX, this.imgsz - newW - padX], [0, 0]], LETTERBOX_FILL)
            .transpose([2, 0, 1])
            .expandDims(0));

        const output = this.model.predict(input);
        input.dispose();

        const { boxesXYXY, scores, classIds } = tf.tidy(() => {
            const numClasses = this.classNames.length;
            const featDim = 4 + numClasses;
            // YOLOv8 export output is either [1, featDim, numAnchors] or
            // [1, numAnchors, featDim] depending on the export path — detect
            // which axis holds the (box + class) features and normalize to
            // [numAnchors, featDim].
            const feats = output.shape[1] === featDim
                ? output.transpose([0, 2, 1]).squeeze([0])
                : output.squeeze([0]);

            const boxesXYWH = feats.slice([0, 0], [-1, 4]);
            const classScores = feats.slice([0, 4], [-1, numClasses]);

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

        // Undo the letterbox: strip the pad offset, then rescale from the
        // resized-but-unpadded region back to the original frame dimensions.
        const toSrcX = px => Math.max(0, Math.min(srcW, (px - padX) / scale));
        const toSrcY = px => Math.max(0, Math.min(srcH, (px - padY) / scale));

        const detections = keepIndices.map(i => {
            const [y1, x1, y2, x2] = boxesData[i];
            const left = toSrcX(x1);
            const top = toSrcY(y1);
            const right = toSrcX(x2);
            const bottom = toSrcY(y2);
            return {
                className: this.classNames[classIdsData[i]] || 'plant',
                confidence: (scoresData[i] * 100).toFixed(1),
                boundingBox: {
                    x: left,
                    y: top,
                    width: right - left,
                    height: bottom - top
                }
            };
        });

        debugLog(`Custom model: raw anchors=${boxesData.length}, kept after NMS=${keepIndices.length}`,
            keepIndices.length ? `confidences=[${keepIndices.map(i => scoresData[i].toFixed(2)).join(', ')}]` : '');

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
    status: 'loading', // 'loading' | 'ready' | 'error'
    errorMessage: null,
    lastFrameTime: 0,
    inferenceTime: 0,

    async logEnvironmentDiagnostics() {
        debugLog('--- MODEL DIAGNOSTICS ---');
        debugLog('TensorFlow.js version:', typeof tf !== 'undefined' ? tf.version.tfjs : 'NOT LOADED');
        if (typeof tf !== 'undefined') {
            try {
                await tf.ready();
                debugLog('Backend:', tf.getBackend());
            } catch (err) {
                debugLog('Backend: FAILED TO INITIALIZE -', err.message);
            }
        }
        debugLog('WebGL available:', (() => {
            try { return !!document.createElement('canvas').getContext('webgl2'); } catch { return false; }
        })());
        debugLog('WASM available:', typeof WebAssembly !== 'undefined');
        debugLog('crossOriginIsolated:', typeof self !== 'undefined' ? self.crossOriginIsolated : 'unknown');
        debugLog('SharedArrayBuffer available:', typeof SharedArrayBuffer !== 'undefined');
        debugLog('tf global defined:', typeof tf !== 'undefined');
        debugLog('cocoSsd global defined:', typeof cocoSsd !== 'undefined');
        debugLog('tflite global defined:', typeof tflite !== 'undefined');
        debugLog('Custom model URL: model/plant_detector.tflite');
        debugLog('COCO-SSD default model URL: https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json',
            '(NOTE: this dataset\'s hosted weight shards were confirmed 404/NoSuchKey as of 2026-09 - Google appears to have pruned them; COCO-SSD may fail through no fault of this app\'s code)');
        debugLog('-------------------------');
    },

    async init() {
        this.isLoading = true;
        this.status = 'loading';
        this.errorMessage = null;
        await this.logEnvironmentDiagnostics();
        try {
            if (typeof tf !== 'undefined' && await CustomModel.load()) {
                this.engineType = 'custom';
                this.isReady = true;
                this.status = 'ready';
                debugLog('Ready. Engine = custom plant detector (YOLOv8n/TFLite)');
                return true;
            }

            // Fall back to the generic COCO-SSD model. NOTE: as of this session,
            // COCO-SSD's default hosted weights (all 3 base-model variants -
            // lite_mobilenet_v2, mobilenet_v1, mobilenet_v2) were verified dead
            // (storage.googleapis.com returns 404 NoSuchKey for every weight
            // shard, even though model.json itself still resolves). This is an
            // external breakage in Google's hosting, not something fixable from
            // this app. We still attempt it (in case it's restored) but fail
            // fast rather than waiting the full timeout on a near-certain dead end.
            if (typeof cocoSsd === 'undefined') {
                throw new Error('COCO-SSD library did not load from CDN (check network/adblock)');
            }
            debugLog('Custom model unavailable - trying COCO-SSD fallback (known to have dead default weight hosting, see above)...');
            this.model = await withTimeout(cocoSsd.load(), 8000, 'COCO-SSD model load');
            this.engineType = 'coco-ssd';
            this.isReady = true;
            this.status = 'ready';
            debugLog('Ready. Engine = COCO-SSD (generic, filtered to plant-adjacent classes)');
            return true;
        } catch (err) {
            console.error('[AGRIVISION] MODEL LOAD ERROR');
            console.error('[AGRIVISION] error.name:', err.name);
            console.error('[AGRIVISION] error.message:', err.message);
            console.error('[AGRIVISION] error.stack:', err.stack);
            this.isReady = false;
            this.status = 'error';
            this.errorMessage = err.message;
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
                debugLog(`Inference: ${this.inferenceTime.toFixed(0)}ms, plant detections: ${results.length}`);
                return results;
            } catch (err) {
                console.error('[AGRIVISION] Detection error:', err);
                return [];
            }
        }

        if (!this.model) {
            return [];
        }

        try {
            // Use COCO-SSD for plant detection. Pass a lower raw minScore
            // (default is 0.5) so weaker plant-class candidates aren't
            // discarded before we even get to look at them — COCO-SSD's
            // "potted plant" class is narrow (trained mostly on full houseplants
            // with a visible pot) and often scores lower on close-ups, seedling
            // trays, or plants partly occluded by a hand. This is safe because
            // the class filter below still enforces precision: only real
            // plant-adjacent classes survive, regardless of how low the
            // threshold is set here.
            const predictions = await this.model.detect(frame, 20, 0.25);

            // Debug visibility: log everything COCO-SSD actually saw in this
            // frame (class + confidence), not just what passed our plant
            // filter. Open devtools console to check whether "no plant
            // detected" means the model saw nothing plant-like at all, or
            // saw something plant-like below the class-match filter below.
            debugLog(`COCO-SSD raw detections: ${predictions.length}`,
                predictions.length ? predictions.map(p => `${p.class} ${(p.score * 100).toFixed(1)}%`).join(', ') : '(nothing above 25%)');

            // Filter for plant-like objects only. COCO-SSD's 80 classes only
            // include "potted plant" and "vase" as plant-adjacent categories
            // (there is no generic "plant"/"flowers" class in the stock model,
            // but we keep those strings in case a future model version adds
            // them). We deliberately do NOT fall back to "any confident
            // object" here — that was the bug: a person, chair, etc. detected
            // with high confidence was being mislabeled and counted as a
            // plant whenever no real plant was in frame.
            const plantClasses = ['potted plant', 'plant', 'flowers', 'vase'];
            const results = predictions.filter(pred => {
                const className = pred.class.toLowerCase();
                return plantClasses.some(pc => className.includes(pc));
            });

            this.inferenceTime = performance.now() - startTime;
            debugLog(`Inference: ${this.inferenceTime.toFixed(0)}ms, raw: ${predictions.length}, plant detections: ${results.length}`);

            return results.map(detection => ({
                className: detection.class,
                confidence: (detection.score * 100).toFixed(1),
                boundingBox: {
                    x: detection.bbox[0],
                    y: detection.bbox[1],
                    width: detection.bbox[2],
                    height: detection.bbox[3]
                }
            }));
        } catch (err) {
            console.error('[AGRIVISION] Detection error:', err);
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
        modelReady: false,
        modelStatus: 'loading', // 'loading' | 'ready' | 'error' — mirrors DetectionEngine.status
        modelError: null
    },

    // Cap inference well below camera FPS (Step 7): even if a frame infers in
    // a few ms, running the model on every camera frame (up to 30-60/s) would
    // fight the main thread for no benefit. ~8 FPS is plenty for a hand-held
    // plant to track visibly while keeping the tab responsive.
    minDetectionIntervalMs: 120,
    lastDetectionAt: 0,

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

        if (!this.state.modelReady && !this.state.simulationMode) {
            const reason = this.state.modelStatus === 'error'
                ? `AI model failed to load (${this.state.modelError || 'unknown error'}). Enable Simulation Mode to continue testing the UI.`
                : 'AI model is still loading. Wait a moment, or enable Simulation Mode.';
            this.showError(reason);
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

    isDetecting: false,

    animationLoop() {
        if (!this.state.cameraRunning) return;

        PerformanceMonitor.recordCameraFrame();

        const now = performance.now();
        if (this.state.detectionActive && !this.isDetecting && now - this.lastDetectionAt >= this.minDetectionIntervalMs) {
            const frame = CameraManager.getFrame();
            if (frame && frame.readyState === frame.HAVE_ENOUGH_DATA) {
                this.lastDetectionAt = now;
                this.runDetection(frame);
            }
        }

        requestAnimationFrame(() => this.animationLoop());
    },

    async runDetection(frame) {
        this.isDetecting = true;
        const detections = await DetectionEngine.detectFrame(frame);
        this.isDetecting = false;
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

        const aiStatusEl = document.getElementById('aiStatus');
        if (this.state.simulationMode) {
            aiStatusEl.textContent = 'SIMULATION';
            aiStatusEl.className = 'status-badge ready';
            aiStatusEl.title = '';
        } else if (this.state.modelReady) {
            aiStatusEl.textContent = 'READY';
            aiStatusEl.className = 'status-badge ready';
            aiStatusEl.title = '';
        } else if (this.state.modelStatus === 'error') {
            aiStatusEl.textContent = 'ERROR';
            aiStatusEl.className = 'status-badge error';
            aiStatusEl.title = this.state.modelError || 'Model failed to load';
        } else {
            aiStatusEl.textContent = 'LOADING';
            aiStatusEl.className = 'status-badge loading';
            aiStatusEl.title = '';
        }

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
            detectionsList.innerHTML = StatsManager.detections.map((det, idx) => {
                const label = (det.className || 'plant').replace(/\b\w/g, c => c.toUpperCase());
                return `
                <div class="detection-item">
                    <span class="detection-name">${label} ${idx + 1}</span>
                    <span class="detection-confidence">${det.confidence}%</span>
                </div>
            `;
            }).join('');
        }
    },

    updateStats() {
        document.getElementById('cameraFpsValue').textContent = StatsManager.cameraFps;
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
    debugLog('Initializing AGRIVISION Plant Detection System...');

    // Initialize UI
    UIManager.init();
    Visualizer.init();

    // Initialize detection model
    const modelReady = await DetectionEngine.init();
    UIManager.state.modelReady = modelReady;
    UIManager.state.modelStatus = DetectionEngine.status;
    UIManager.state.modelError = DetectionEngine.errorMessage;
    UIManager.updateAllStatus();

    if (modelReady) {
        debugLog(`AGRIVISION ready. Engine: ${DetectionEngine.engineType}. Click START CAMERA to begin.`);
    } else {
        debugLog('AGRIVISION model failed to load. Real detection unavailable until this is fixed:', DetectionEngine.errorMessage);
    }
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

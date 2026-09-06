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
   - TensorFlow.js + COCO-SSD for plant detection
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
// 0b. DETECTION PIPELINE CONFIG + GEOMETRY / DE-DUPLICATION HELPERS
// ============================================================================

// Single source of truth for every threshold in the
//   detect -> class-filter -> confidence-filter -> NMS -> dedupe -> count
// pipeline. Tuned for the rule "one physical plant = one detection".
const DETECTION_CONFIG = {
    // Which detector to try FIRST ('coco-ssd' | 'custom'); the other is the
    // fallback. Default is 'coco-ssd' deliberately.
    //
    // The shipped model/plant_detector.tflite is a single-class ("plant")
    // YOLOv8n trained on an Open Images slice in which EVERY image contains a
    // plant - it never saw a background/negative image, and Fruit/Flower/
    // Flowerpot were merged into `plant`. The result is that it reports plants
    // on a person-only frame at >90% confidence. Its inference path here is
    // correct (verified op-by-op against the .tflite flatbuffer); the model
    // itself is the defect.
    //
    // COCO-SSD is trained on all 80 COCO classes, so `person`, `chair`,
    // `bottle`, `laptop`, `dining table`, `tv` etc. are learned as their own
    // classes and get rejected by CLASS VALIDATION below - which is exactly the
    // negative signal the custom model lacks. Until the retrain lands (see
    // training/train_plant_detector.ipynb), this is the engine that satisfies
    // "person only -> 0 plants".
    //
    // Switch to 'custom' after installing a retrained plant_detector.tflite.
    engine: 'coco-ssd',

    // COCO-SSD base network. The library default is 'lite_mobilenet_v2' (17MB),
    // which is the SPEED variant and the weakest of the three - SSD is already
    // poor at small objects because it resizes the frame to 300x300, and the
    // lite variant is weaker still. A plant that occupies a small part of the
    // frame (on a phone screen, across the room, held at arm's length) is
    // exactly the case it misses. 'mobilenet_v2' (64MB) is the ACCURACY variant
    // and is the right default for a prototype whose inference is already
    // throttled to ~8 FPS. Drop back to 'lite_mobilenet_v2' if load time or
    // frame rate matters more than recall.
    cocoBase: 'mobilenet_v2',   // 'lite_mobilenet_v2' | 'mobilenet_v1' | 'mobilenet_v2'

    // OBSERVATION FLOOR - how low we ask COCO-SSD to report boxes.
    // This is NOT the decision gate. It exists purely so the console can show
    // near-misses: a plant box the model produced but scored weakly is the
    // difference between "the model never saw a plant" and "our filter threw
    // the plant away", and those two problems have completely different fixes.
    // Lowering this changes nothing about what COUNTS as a plant.
    observationFloor: 0.10,

    // Minimum class confidence for a box to count as a plant, PER ENGINE.
    //
    // These differ for a real reason, not as a tuning knob:
    //
    //   coco-ssd (0.25) - precision is guaranteed by CLASS VALIDATION, not by
    //     confidence. A `person` at 0.99 is rejected because of its class, so a
    //     low score threshold cannot admit a person, chair, bottle or laptop -
    //     it can only recover weak PLANT boxes. And COCO's only plant class,
    //     `potted plant`, was trained mostly on unoccluded houseplants filling
    //     the frame: a plant held by a person is smaller, often partly occluded
    //     by a hand, and at an odd angle, so it routinely scores 0.25-0.40. A
    //     0.40 gate silently discarded exactly those, which made a person in
    //     frame LOOK like it was cancelling plant detection.
    //
    //   custom (0.40) - the custom YOLO detector is single-class, so class
    //     validation is a no-op and confidence is the ONLY precision control
    //     it has. It keeps the stricter gate.
    // NOTE: with temporal confirmation in place (below) this is now the
    // CANDIDATE gate, not the final decision. A box only has to be plausible
    // here; it still has to survive across frames to be counted. That is what
    // lets this sit lower than a single-frame system could safely allow -
    // evidence accumulated over time replaces evidence in one frame, which
    // raises recall on weak/occluded plants WITHOUT inflating any confidence
    // value and WITHOUT admitting a non-plant class.
    scoreThreshold: {
        'coco-ssd': 0.20,
        'custom': 0.40
    },

    // ---- TEMPORAL CONFIRMATION (two-level result) -------------------------
    // A live webcam produces unstable per-frame output: a real plant drops out
    // for a frame, and a spurious box flickers in for a frame. Deciding from a
    // single frame therefore costs BOTH precision and recall.
    //
    //   PLANT CANDIDATE - class is "potted plant", clears the candidate gate,
    //                     and has a sane box. Not counted, not drawn.
    //   CONFIRMED PLANT - a candidate tracked across frames with enough
    //                     supporting evidence. Counted, drawn, sent to Model 2.
    //
    // A clearly-visible plant still confirms on its FIRST frame via
    // instantConfirmScore, so there is no added latency in the easy case. The
    // window only matters for weak/borderline detections, which is exactly
    // where a single frame is not trustworthy.
    //
    // Frames here are DETECTION frames (~8/s, see minDetectionIntervalMs), so
    // 1 frame is roughly 125 ms.
    temporal: {
        enabled: true,
        windowFrames: 6,            // ~750ms of history per track
        minHits: 3,                 // seen in >= 3 of the last 6 frames
        maxMisses: 4,               // ~500ms grace before a confirmed plant drops
        // IoU to associate this frame's box with an existing track. 0.30 was too
        // generous: two separate plants whose boxes overlap ~0.3-0.4 could both
        // get pulled onto ONE track, so only one confirmed and the count read 1
        // instead of 2. 0.45 requires real overlap to associate; a single
        // plant's frame-to-frame jitter is normally IoU > 0.6 so it still
        // tracks fine.
        matchIou: 0.45,
        emaAlpha: 0.5,              // smoothing for the confirmation decision only
        confirmScore: 0.30,         // sustained score needed alongside minHits
        instantConfirmScore: 0.55,  // strong enough to confirm from one frame
        // Persistence route. Without this there is a DEAD BAND between the
        // candidate gate (0.20) and confirmScore (0.30): a real but weak plant -
        // a plant half-hidden behind a person, say - would stay a candidate
        // forever and could never be counted no matter how long it sat there.
        // That defeats the point of accumulating evidence over time. A box that
        // holds the same position in 5 of the last 6 frames is not noise; noise
        // does not track. Class validation still applies, so this can only
        // promote a weak PLANT - never a person, chair or phone.
        persistentHits: 5,
        // Below this, a confirmed plant is flagged UNCERTAIN in the UI rather
        // than presented as a solid result. Nothing is hidden - it still counts.
        strongScore: 0.50
    },
    // TensorFlow NMS: a raw model box overlapping an already-kept box by more
    // than this IoU is treated as the same box and dropped.
    nmsIouThreshold: 0.45,
    // ---- Duplicate suppression, PER ENGINE (they fail differently) --------
    //
    // CUSTOM (YOLOv8): its decode can emit a tight foliage box AND a wide
    // whole-plant box for ONE plant. Their IoU is low (small box / big union),
    // so an IoU test alone keeps both and over-counts. The CONTAINMENT test
    // (intersection / area-of-smaller-box) catches the nested box. Keep it.
    //
    // COCO-SSD: `potted plant` boxes are whole-plant, one per instance, and
    // coco-ssd ALREADY runs its own NMS (IoU ~0.5) before we see them - so
    // genuine same-plant duplicates are rare here. The containment test is
    // actively HARMFUL for coco: two real plants placed close together often
    // have one box 60-80% inside the other, and containment then deletes the
    // second plant. THIS is why "2 plants" was becoming "1". So coco uses an
    // IoU-ONLY pass at a loose threshold: it only removes a box that is a near
    // copy of one already kept (similar size, heavy overlap), and never merges
    // two boxes just because one sits inside the other's area.
    dedupeIouThreshold: 0.50,        // custom
    dedupeOverlapThreshold: 0.60,    // custom containment
    cocoDedupeIouThreshold: 0.60,    // coco: IoU only, no containment
    // CLASS VALIDATION - a detection is only counted as a plant if its class
    // name matches one of these (case-insensitive substring). This is the ONLY
    // thing that rejects a person, and it rejects by class, never by presence:
    // a frame containing person + potted plant keeps the plant and drops the
    // person, so a person can never cancel a plant.
    //
    // Of COCO's 80 classes only `potted plant` matches (via the 'plant'
    // substring; 'airplane' does NOT contain 'plant'). The other entries are
    // for a future multi-class model and are inert today.
    //
    // `vase` is deliberately NOT here. A vase is a container, not a plant - an
    // empty one on a shelf would otherwise register as a plant and break the
    // person-only-means-zero guarantee. A plant standing in a vase is normally
    // also emitted as `potted plant`, so little recall is lost.
    plantClassNames: ['plant', 'potted plant', 'houseplant', 'flowers'],
    // Draw the RAW / CONFIDENCE / NMS / FINAL counts onto the canvas. Temporary
    // visual debugging for this fix - set false to hide.
    debugHud: true,

    // ---- MODEL 2: plant health / crop analysis -----------------------------
    // Runs only on the crops Model 1 already validated as plants. Disabled
    // automatically when model/plant_health_classifier.tflite is absent, so the
    // detection stage is unaffected until the model is trained and installed.
    analysis: {
        enabled: true,
        modelUrl: 'model/plant_health_classifier.tflite',
        metadataUrl: 'model/class_names.json',
        // Classification is far heavier than detection and a plant does not
        // change condition between frames, so run it on a slow cadence and
        // carry results forward by box overlap in between.
        intervalMs: 800,
        // Expand the detection box slightly before cropping - COCO-SSD boxes
        // sometimes clip leaf tips, and the classifier needs the lesion.
        cropMarginPct: 0.08,
        // Cap work per cycle so a crowded frame cannot stall the UI thread.
        maxPerCycle: 4,
        // Carry a previous analysis onto this frame's box when they overlap
        // at least this much (same physical plant, slightly moved).
        carryForwardIou: 0.40
    }
};

function boxIntersectionArea(a, b) {
    const ax2 = a.x + a.width, ay2 = a.y + a.height;
    const bx2 = b.x + b.width, by2 = b.y + b.height;
    const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
    const ih = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
    return iw * ih;
}

function boxIoU(a, b) {
    const inter = boxIntersectionArea(a, b);
    const union = a.width * a.height + b.width * b.height - inter;
    return union > 0 ? inter / union : 0;
}

// Intersection over the area of the SMALLER box: approaches 1.0 when the
// smaller box sits almost entirely inside the larger one, even when their IoU
// is low. This is the test that collapses "part of a plant" boxes.
function boxOverlapRatio(a, b) {
    const inter = boxIntersectionArea(a, b);
    const minArea = Math.min(a.width * a.height, b.width * b.height);
    return minArea > 0 ? inter / minArea : 0;
}

// Greedy duplicate suppression. Walks detections highest-confidence first and
// keeps a box only if it is not a duplicate of one already kept.
//
//   useContainment=false (COCO-SSD): duplicate == high IoU only. Two boxes are
//     the same detection only if they heavily overlap AND are similar in size.
//     A smaller box sitting inside a larger one is NOT merged - that is how two
//     nearby-but-separate plants both survive.
//
//   useContainment=true (custom YOLO): also merge a box that is mostly
//     contained inside a kept box (intersection / area-of-smaller-box). Needed
//     because that decode emits nested tight/loose boxes for one plant.
//
// Returns { kept, removed }, where each removed entry records WHY it was
// dropped and against which kept box - so the console can show exactly why a
// second plant disappeared, if it does.
function dedupeDetections(detections, { iouThresh = 0.5, overlapThresh = 0.6, useContainment = true } = {}) {
    const sorted = [...detections].sort(
        (a, b) => parseFloat(b.confidence) - parseFloat(a.confidence)
    );
    const kept = [];
    const removed = [];
    for (const det of sorted) {
        let killer = null, reason = null, metric = 0;
        for (const k of kept) {
            const iou = boxIoU(k.boundingBox, det.boundingBox);
            if (iou > iouThresh) { killer = k; reason = 'IoU'; metric = iou; break; }
            if (useContainment) {
                const ov = boxOverlapRatio(k.boundingBox, det.boundingBox);
                if (ov > overlapThresh) { killer = k; reason = 'containment'; metric = ov; break; }
            }
        }
        if (killer) {
            removed.push({ det, against: killer, reason, metric });
        } else {
            kept.push(det);
        }
    }
    return { kept, removed };
}

// True only for real plant class names. Used as the CLASS VALIDATION stage in
// both detection engines - this is what keeps a person / face / background
// object from ever reaching the plant count, without a "if face -> no plant"
// hack (a person holding a plant still yields the plant).
function isPlantClass(name) {
    const n = String(name || '').toLowerCase();
    return DETECTION_CONFIG.plantClassNames.some(c => n.includes(c));
}

// PLANT VALIDATION - the last gate before a box is counted as a real plant.
// Three checks, all evidence-based; nothing here rejects a box for where it
// sits in the frame or how green it is:
//   1. class    - must be a plant class (rejects person/chair/bottle/laptop...)
//   2. score    - must clear the confidence threshold
//   3. geometry - must be a non-degenerate box. A zero/negative/NaN-sized box
//                 means the coordinate decode produced garbage, and a garbage
//                 box must never be counted as a plant.
function isValidPlantDetection(det, minConfidence) {
    if (!det || !det.boundingBox) return false;
    if (!isPlantClass(det.className)) return false;

    const conf = parseFloat(det.confidence);
    if (!Number.isFinite(conf) || conf < minConfidence * 100) return false;

    const { x, y, width, height } = det.boundingBox;
    if (![x, y, width, height].every(Number.isFinite)) return false;
    if (width <= 1 || height <= 1) return false;

    return true;
}

// Confidence gate for the engine that is actually running. See the comment on
// DETECTION_CONFIG.scoreThreshold for why the two engines differ.
function scoreThresholdFor(engine) {
    const t = DETECTION_CONFIG.scoreThreshold;
    return (typeof t === 'number') ? t : (t[engine] ?? 0.40);
}

// ============================================================================
// 0c. TEMPORAL CONFIRMATION - candidate -> confirmed plant
// ============================================================================
//
// Associates this frame's plant CANDIDATES with plants seen in previous frames
// and promotes a track to CONFIRMED only once it has enough evidence. This is
// the piece that makes a live webcam stable:
//
//   * a one-frame spurious box never confirms          -> precision up
//   * a real plant that drops out for a frame or two    -> recall up
//     stays confirmed through the gap
//
// It works on the REAL scores throughout. The EMA is used only to decide
// whether to confirm; the confidence reported to the UI is the genuine
// COCO-SSD score from the last frame the plant was actually seen.
const PlantTracker = {
    tracks: [],
    nextId: 1,

    reset() {
        this.tracks = [];
        this.nextId = 1;
    },

    // Greedy IoU association: strongest candidate first, each track claimed once.
    update(candidates) {
        const cfg = DETECTION_CONFIG.temporal;
        if (!cfg.enabled) return candidates.map(c => ({ ...c, confirmed: true }));

        const claimed = new Set();
        const matchedTracks = new Set();

        const ordered = [...candidates].sort(
            (a, b) => parseFloat(b.confidence) - parseFloat(a.confidence));

        for (const cand of ordered) {
            let best = null, bestIou = cfg.matchIou;
            for (const track of this.tracks) {
                if (matchedTracks.has(track.id)) continue;
                const iou = boxIoU(track.boundingBox, cand.boundingBox);
                if (iou > bestIou) { bestIou = iou; best = track; }
            }
            const score = parseFloat(cand.confidence) / 100;
            if (best) {
                matchedTracks.add(best.id);
                claimed.add(cand);
                best.boundingBox = cand.boundingBox;
                best.className = cand.className;
                best.confidence = cand.confidence;      // real, latest score
                best.ema = cfg.emaAlpha * score + (1 - cfg.emaAlpha) * best.ema;
                best.history.push(true);
                best.misses = 0;
            } else {
                claimed.add(cand);
                const id = this.nextId++;
                // Mark the new track as matched THIS frame. Without this the
                // miss loop below immediately scores the brand-new track as a
                // miss, which costs it a frame of latency and corrupts its hit
                // history - a strong plant would confirm on frame 2 instead of
                // frame 1.
                matchedTracks.add(id);
                this.tracks.push({
                    id,
                    boundingBox: cand.boundingBox,
                    className: cand.className,
                    confidence: cand.confidence,
                    ema: score,
                    history: [true],
                    misses: 0,
                    confirmed: false
                });
            }
        }

        // Tracks with no candidate this frame record a miss.
        for (const track of this.tracks) {
            if (matchedTracks.has(track.id)) continue;
            track.history.push(false);
            track.misses++;
        }

        // Trim history, drop dead tracks, evaluate confirmation.
        this.tracks = this.tracks.filter(track => {
            if (track.history.length > cfg.windowFrames) {
                track.history = track.history.slice(-cfg.windowFrames);
            }
            if (track.misses > cfg.maxMisses) return false;

            const hits = track.history.reduce((n, seen) => n + (seen ? 1 : 0), 0);
            const sustained = hits >= cfg.minHits && track.ema >= cfg.confirmScore;
            const instant = track.ema >= cfg.instantConfirmScore;
            // Persistence: held position across nearly the whole window at any
            // candidate-level score. Closes the 0.20-0.30 dead band.
            const persistent = hits >= cfg.persistentHits;

            // Once confirmed, a track stays confirmed while it is still alive -
            // otherwise the UI would flicker every time the score dipped.
            track.confirmed = track.confirmed || sustained || instant || persistent;
            track.hits = hits;
            return true;
        });

        return this.tracks
            .filter(t => t.confirmed && t.misses === 0)
            .map(t => ({
                className: t.className,
                confidence: t.confidence,               // genuine COCO-SSD score
                boundingBox: t.boundingBox,
                trackId: t.id,
                confirmed: true,
                // Flagged, never hidden: a confirmed-but-weak plant is still
                // counted, but the UI says the evidence is thin.
                uncertain: t.ema < cfg.strongScore,
                smoothedScore: t.ema,
                hits: t.hits
            }));
    },

    // Human-readable state for the debug log.
    describe() {
        if (!this.tracks.length) return 'no active tracks';
        return this.tracks.map(t =>
            `#${t.id}[${t.confirmed ? 'CONFIRMED' : 'pending'} ` +
            `hits=${t.hits ?? 0}/${DETECTION_CONFIG.temporal.windowFrames} ` +
            `misses=${t.misses} ema=${t.ema.toFixed(2)}]`).join(' ');
    }
};

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
    // Thresholds now live in DETECTION_CONFIG (shared with the COCO-SSD
    // fallback). Exposed as getters so existing references / debug logs still
    // read the live value.
    get scoreThreshold() { return scoreThresholdFor('custom'); },
    get iouThreshold() { return DETECTION_CONFIG.nmsIouThreshold; },

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

        // --- Pipeline stage counts (diagnosing "1 plant -> 2 detections") ---
        //   RAW  : anchors scored above a low floor (every plausible box)
        //   CONF : anchors above the real confidence threshold, BEFORE any NMS
        //   NMS  : boxes left after TensorFlow's IoU-based non-max suppression
        //   FINAL: after the class-agnostic overlap dedupe below
        const RAW_FLOOR = 0.10;
        const rawCount = scoresData.reduce((n, s) => n + (s >= RAW_FLOOR ? 1 : 0), 0);
        const afterConfidence = scoresData.reduce((n, s) => n + (s >= this.scoreThreshold ? 1 : 0), 0);
        const afterNms = keepIndices.length;

        // --- Box coordinate space ---
        // This exported .tflite emits box coords NORMALIZED to [0,1]: the graph
        // ends with `xywh_pixels * (1/imgsz)` - verified by reading the
        // flatbuffer (final MUL by the constant 0.0015625 == 1/640, applied to
        // the 4 box channels only, before the class channel is concatenated).
        // The previous code treated these as 640-pixel-space values, so
        // `(0.4 - padX) / scale` went negative and every box collapsed into a
        // sub-pixel dot at the top-left corner.
        //
        // Auto-detect the space (instead of hard-assuming normalized) so a
        // future re-export that changes this doesn't silently break rendering:
        // if the largest coord among kept boxes is > 2.0 it must be pixel-space.
        const maxRawCoord = keepIndices.reduce((mx, i) => {
            const b = boxesData[i];
            return Math.max(mx, Math.abs(b[0]), Math.abs(b[1]), Math.abs(b[2]), Math.abs(b[3]));
        }, 0);
        const coordToImgsz = maxRawCoord > 2.0 ? 1 : this.imgsz; // normalized -> px
        debugLog(`coord decode: maxRawCoord=${maxRawCoord.toFixed(4)} => ` +
            `${coordToImgsz === 1 ? 'PIXEL space (x1)' : 'NORMALIZED (x' + this.imgsz + ')'}`);

        // Undo the letterbox: normalized|px -> 640-canvas px, strip the pad
        // offset, then rescale from the resized-but-unpadded region back to the
        // original frame dimensions.
        const toSrcX = c => Math.max(0, Math.min(srcW, (c * coordToImgsz - padX) / scale));
        const toSrcY = c => Math.max(0, Math.min(srcH, (c * coordToImgsz - padY) / scale));

        const nmsDetections = keepIndices.map(i => {
            const [y1, x1, y2, x2] = boxesData[i];
            const left = toSrcX(x1);
            const top = toSrcY(y1);
            const right = toSrcX(x2);
            const bottom = toSrcY(y2);
            return {
                classId: classIdsData[i],
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

        // CLASS VALIDATION: a box only survives if its class is an actual plant
        // class. This model only emits classId 0 ("plant"), so today every box
        // passes; it's the guard that keeps a future multi-class export - or a
        // mislabeled class - out of the count.
        const classValidated = nmsDetections.filter(
            d => isValidPlantDetection(d, scoreThresholdFor('custom')));

        // DUPLICATE / NMS FILTER: this decode emits nested tight/loose boxes for
        // one plant, so the containment test IS wanted here (useContainment
        // default true).
        const { kept: finalDetections, removed: dupsRemoved } = dedupeDetections(classValidated, {
            iouThresh: DETECTION_CONFIG.dedupeIouThreshold,
            overlapThresh: DETECTION_CONFIG.dedupeOverlapThreshold
        });
        dupsRemoved.forEach(r => debugLog(
            `DUPLICATE REMOVED: ${r.det.className} ${r.det.confidence}% ` +
            `(${r.reason} ${r.metric.toFixed(2)} vs kept box ${r.against.confidence}%)`));

        DetectionEngine.lastPipelineStats = {
            engine: 'custom',
            raw: rawCount,
            afterConfidence,
            afterNms,
            afterClass: classValidated.length,
            final: finalDetections.length
        };
        DetectionEngine.logDetectionSummary({
            raw: `${rawCount} anchors >= ${RAW_FLOOR} (of ${scoresData.length})`,
            afterConfidence, afterNms, classValidated, final: finalDetections
        });

        tf.dispose([output, boxesXYXY, scores, classIds, nmsIndices]);
        return finalDetections;
    }
};

// ============================================================================
// 2b. MODEL 2 - PLANT HEALTH / CROP ANALYSIS
// ============================================================================
//
// STAGE SEPARATION (deliberate):
//   MODEL 1 (COCO-SSD, DetectionEngine)  -> "is there a plant, and where?"
//   MODEL 2 (PlantAnalyzer, this module) -> "what crop, is it healthy, what
//                                            condition?" for ONE plant crop.
//
// Model 2 never sees the full frame. It is fed only the pixels inside a box
// Model 1 already class-validated as a plant, so a face or background cannot
// influence the health classification. Each plant is analysed independently -
// three plants produce three separate results, never one averaged verdict.
//
// The module self-disables if the model files are missing, leaving detection,
// counting, boxes, the popup and the debug overlay exactly as they are.
const PlantAnalyzer = {
    model: null,
    meta: null,
    available: false,
    status: 'not-loaded',  // 'not-loaded' | 'ready' | 'absent' | 'error'
    inputSize: 224,
    lastRunAt: 0,
    inferenceTime: 0,

    async load() {
        if (!DETECTION_CONFIG.analysis.enabled) {
            this.status = 'not-loaded';
            return false;
        }
        const cfg = DETECTION_CONFIG.analysis;
        try {
            // Probe the metadata first: it is small, and its absence is the
            // normal "model not trained yet" case rather than an error.
            const metaRes = await fetch(cfg.metadataUrl);
            if (!metaRes.ok) {
                this.status = 'absent';
                debugLog(`Model 2 (plant analysis) not installed - ${cfg.metadataUrl} ` +
                    `returned ${metaRes.status}. Detection continues normally. ` +
                    `Train it with training/plant_health/.`);
                return false;
            }
            this.meta = await metaRes.json();
            this.inputSize = this.meta?.input?.width || 224;

            if (typeof tflite === 'undefined') {
                throw new Error('tfjs-tflite not loaded');
            }
            tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');
            this.model = await withTimeout(
                tflite.loadTFLiteModel(cfg.modelUrl), 20000, 'Plant analysis model load');

            this.available = true;
            this.status = 'ready';
            debugLog(`Model 2 ready: ${this.meta.class_names.length} classes, ` +
                `${this.meta.crops.length} crops, input ${this.inputSize}px, ` +
                `range ${this.meta.input.range}`);
            return true;
        } catch (err) {
            this.status = 'error';
            console.error('[AGRIVISION] Model 2 (plant analysis) failed to load:', err.message);
            debugLog('Detection stage is unaffected; analysis disabled.');
            return false;
        }
    },

    // Crop one detection box out of the live frame and resize it to the
    // classifier's input. drawImage does the resampling; the result is fed as
    // RAW 0-255 floats because MobileNetV3's preprocessing is baked into the
    // exported graph (metadata.input.range says so - we do not guess).
    cropToTensor(frame, box) {
        const srcW = frame.videoWidth || frame.width;
        const srcH = frame.videoHeight || frame.height;
        const margin = DETECTION_CONFIG.analysis.cropMarginPct;

        const mx = box.width * margin;
        const my = box.height * margin;
        const sx = Math.max(0, box.x - mx);
        const sy = Math.max(0, box.y - my);
        const sw = Math.min(srcW - sx, box.width + mx * 2);
        const sh = Math.min(srcH - sy, box.height + my * 2);
        if (sw < 8 || sh < 8) return null;

        const canvas = document.createElement('canvas');
        canvas.width = this.inputSize;
        canvas.height = this.inputSize;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(frame, sx, sy, sw, sh, 0, 0, this.inputSize, this.inputSize);

        return tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().expandDims(0));
    },

    // Turn one softmax vector over "Crop___Condition" classes into crop /
    // health / condition, by exact marginalisation:
    //   P(crop)    = sum of that crop's class probabilities
    //   P(healthy) = sum of the healthy classes' probabilities
    //   condition  = the single most likely class, with its own probability
    // Three separate heads could contradict each other (crop=Tomato while
    // condition=Potato Early blight); marginalising one distribution cannot.
    interpret(probs) {
        const cm = this.meta.class_meta;
        const thresholds = this.meta.confidence || {};

        const cropScores = new Map();
        let healthyScore = 0;
        for (let i = 0; i < cm.length; i++) {
            const p = probs[i];
            cropScores.set(cm[i].crop, (cropScores.get(cm[i].crop) || 0) + p);
            if (cm[i].healthy) healthyScore += p;
        }

        let bestCrop = null, bestCropScore = 0;
        for (const [crop, score] of cropScores) {
            if (score > bestCropScore) { bestCropScore = score; bestCrop = crop; }
        }

        // Top-1 class overall, and the runner-up, for a stability margin.
        let top = 0, second = 1;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
        for (let i = 0; i < probs.length; i++) {
            if (i !== top && (second === top || probs[i] > probs[second])) second = i;
        }
        const margin = probs[top] - (probs[second] ?? 0);

        const unhealthyScore = 1 - healthyScore;
        const healthy = healthyScore >= unhealthyScore;
        const healthConfidence = healthy ? healthyScore : unhealthyScore;

        // Abstain rather than guess. There is no trained "unknown" class - a
        // reject option on the distribution is the honest way to handle input
        // the model was never trained for (a face, a wall, an unlisted crop).
        const uncertain =
            bestCropScore < (thresholds.crop_min ?? 0.6) ||
            probs[top] < (thresholds.condition_min ?? 0.55) ||
            healthConfidence < (thresholds.health_min ?? 0.6) ||
            margin < (thresholds.margin_min ?? 0.10);

        return {
            uncertain,
            crop: this.meta.crop_display_names?.[bestCrop] || bestCrop,
            cropConfidence: bestCropScore,
            health: healthy ? 'Healthy' : 'Unhealthy',
            healthConfidence,
            condition: cm[top].condition_display,
            conditionConfidence: probs[top],
            margin
        };
    },

    async analyzeOne(frame, box) {
        const input = this.cropToTensor(frame, box);
        if (!input) return null;
        let output = null;
        try {
            output = this.model.predict(input);
            const probs = await output.data();
            return this.interpret(Array.from(probs));
        } finally {
            input.dispose();
            if (output) tf.dispose(output);
        }
    },

    // Analyse every detection in this frame, independently. Returns after
    // mutating each detection with `.analysis`.
    async analyzeDetections(frame, detections) {
        if (!this.available || !detections.length) return;

        const now = performance.now();
        if (now - this.lastRunAt < DETECTION_CONFIG.analysis.intervalMs) return;
        this.lastRunAt = now;

        const start = performance.now();
        const limit = Math.min(detections.length, DETECTION_CONFIG.analysis.maxPerCycle);
        for (let i = 0; i < limit; i++) {
            try {
                const result = await this.analyzeOne(frame, detections[i].boundingBox);
                if (result) detections[i].analysis = result;
            } catch (err) {
                console.error('[AGRIVISION] Model 2 analysis error:', err);
            }
        }
        this.inferenceTime = performance.now() - start;

        detections.slice(0, limit).forEach((d, i) => {
            const a = d.analysis;
            if (!a) return;
            debugLog(a.uncertain
                ? `  plant[${i}] analysis: UNCERTAIN (crop ${(a.cropConfidence * 100).toFixed(0)}%, ` +
                  `cond ${(a.conditionConfidence * 100).toFixed(0)}%, margin ${(a.margin * 100).toFixed(0)}%)`
                : `  plant[${i}] analysis: ${a.crop} ${(a.cropConfidence * 100).toFixed(0)}% | ` +
                  `${a.health} ${(a.healthConfidence * 100).toFixed(0)}% | ` +
                  `${a.condition} ${(a.conditionConfidence * 100).toFixed(0)}%`);
        });
    },

    // Between analysis cycles, keep showing the previous verdict for a plant
    // that is still in roughly the same place, so the card does not blank out
    // at the detection frame rate.
    carryForward(previous, current) {
        if (!previous || !previous.length) return;
        const minIou = DETECTION_CONFIG.analysis.carryForwardIou;
        for (const det of current) {
            if (det.analysis) continue;
            let best = null, bestIou = minIou;
            for (const prev of previous) {
                if (!prev.analysis) continue;
                const iou = boxIoU(prev.boundingBox, det.boundingBox);
                if (iou > bestIou) { bestIou = iou; best = prev; }
            }
            if (best) det.analysis = best.analysis;
        }
    }
};

const DetectionEngine = {
    engineType: null, // 'custom' or 'coco-ssd'
    model: null,
    isLoading: false,
    isReady: false,
    status: 'loading', // 'loading' | 'ready' | 'error'
    errorMessage: null,
    lastFrameTime: 0,
    inferenceTime: 0,
    // Filled by whichever engine ran last: { engine, raw, afterConfidence,
    // afterNms, afterClass, final }. Drives the on-canvas debug HUD.
    lastPipelineStats: null,

    // The three named debug lines the spec requires, emitted identically by
    // both engines. Pipeline order: RAW -> CONFIDENCE -> NMS -> CLASS VALIDATION
    // -> DUPLICATE FILTER -> FINAL. `classValidated` and `final` are arrays.
    logDetectionSummary({ raw, afterConfidence, afterNms, classValidated, final }) {
        debugLog(`RAW DETECTIONS: ${raw}`);
        debugLog(`  confidence>=thr: ${afterConfidence}  ->  NMS: ${afterNms}  ->  ` +
            `class-valid: ${classValidated.length}  ->  dedupe: ${final.length}`);
        debugLog(`PLANT DETECTIONS: ${classValidated.length}` + (classValidated.length
            ? '  [' + classValidated.map(d =>
                `${d.className} ${d.confidence}% (x:${d.boundingBox.x.toFixed(0)} ` +
                `y:${d.boundingBox.y.toFixed(0)} w:${d.boundingBox.width.toFixed(0)} ` +
                `h:${d.boundingBox.height.toFixed(0)})`).join(' | ') + ']'
            : ''));
        if (final.length < classValidated.length) {
            debugLog(`  dedupe removed ${classValidated.length - final.length} ` +
                `box(es) overlapping a plant already counted`);
        }
        debugLog(`FINAL PLANT COUNT: ${final.length}`);
    },

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
        debugLog('COCO-SSD model URL: https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json',
            '(weight shards verified live: .../group1-shard1of5, HTTP 200, 4MB each - they carry NO .bin extension, which is why an earlier "*.bin" check wrongly reported them pruned)');
        debugLog('Configured engine preference:', DETECTION_CONFIG.engine);
        debugLog('-------------------------');
    },

    async loadCustomEngine() {
        if (typeof tf === 'undefined') return false;
        if (!await CustomModel.load()) return false;
        this.engineType = 'custom';
        debugLog('Engine = custom plant detector (YOLOv8n/TFLite, model/plant_detector.tflite)');
        return true;
    },

    async loadCocoEngine() {
        if (typeof cocoSsd === 'undefined') {
            throw new Error('COCO-SSD library did not load from CDN (check network/adblock)');
        }
        // Hosting note: the weight shards ARE live. They are served without a
        // .bin extension (.../ssdlite_mobilenet_v2/group1-shard1of5), which is
        // why an earlier check for "*.bin" concluded they were 404/pruned. All
        // three base variants verified HTTP 200 with 4MB octet-stream payloads.
        const base = DETECTION_CONFIG.cocoBase;
        debugLog(`Loading COCO-SSD base="${base}" ` +
            `(lite_mobilenet_v2 ~17MB / mobilenet_v1 ~26MB / mobilenet_v2 ~64MB)...`);
        this.model = await withTimeout(cocoSsd.load({ base }), 40000, 'COCO-SSD model load');
        this.engineType = 'coco-ssd';
        debugLog(`Engine = COCO-SSD base="${base}" (80 COCO classes, class-validated ` +
            `down to plant classes). NOTE: COCO's only plant class is "potted plant".`);
        return true;
    },

    async init() {
        this.isLoading = true;
        this.status = 'loading';
        this.errorMessage = null;
        await this.logEnvironmentDiagnostics();

        // Engine order. DETECTION_CONFIG.engine decides which detector is tried
        // first; the other is the fallback so the app still runs if one fails.
        const preferCoco = DETECTION_CONFIG.engine !== 'custom';
        const order = preferCoco
            ? [['coco-ssd', () => this.loadCocoEngine()], ['custom', () => this.loadCustomEngine()]]
            : [['custom', () => this.loadCustomEngine()], ['coco-ssd', () => this.loadCocoEngine()]];

        try {
            let lastErr = null;
            for (const [name, load] of order) {
                try {
                    debugLog(`Trying engine: ${name} ...`);
                    if (await load()) {
                        this.isReady = true;
                        this.status = 'ready';
                        return true;
                    }
                    debugLog(`Engine ${name} unavailable, trying next.`);
                } catch (err) {
                    lastErr = err;
                    console.warn(`[AGRIVISION] Engine ${name} failed to load:`, err.message);
                }
            }
            throw lastErr || new Error('No detection engine could be loaded');
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

    async detectFrame(frame) {
        // realDetection() returns PLANT CANDIDATES. The tracker decides which of
        // them are CONFIRMED PLANTS, and only those are counted, drawn and sent
        // on to Model 2.
        const candidates = await this.realDetection(frame);
        const confirmed = PlantTracker.update(candidates);

        if (this.lastPipelineStats) {
            this.lastPipelineStats.candidates = candidates.length;
            this.lastPipelineStats.final = confirmed.length;
        }

        debugLog(`TEMPORAL CONFIRMATION: ${PlantTracker.describe()}`);
        debugLog(`FINAL PLANT COUNT (confirmed): ${confirmed.length}`);
        if (confirmed.length === 0 && candidates.length > 0) {
            debugLog(`NO PLANT: ${candidates.length} candidate(s) present but not yet ` +
                `temporally confirmed (need ${DETECTION_CONFIG.temporal.minHits} hits in ` +
                `${DETECTION_CONFIG.temporal.windowFrames} frames at score ` +
                `>= ${DETECTION_CONFIG.temporal.confirmScore}, or one frame ` +
                `>= ${DETECTION_CONFIG.temporal.instantConfirmScore})`);
        }
        return confirmed;
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
            // Ask COCO-SSD for EVERY object it can see, down to the OBSERVATION
            // FLOOR (not the decision gate). The frame legitimately contains
            // many objects - person, phone, chair, table - and we want all of
            // them, because the decision is "does a plant exist among them",
            // never "is a plant the only thing here".
            const minConfidence = scoreThresholdFor('coco-ssd');
            const predictions = await this.model.detect(
                frame, 20, DETECTION_CONFIG.observationFloor);

            this.inferenceTime = performance.now() - startTime;

            const mapped = predictions.map(detection => ({
                classId: -1,
                className: detection.class,
                confidence: (detection.score * 100).toFixed(1),
                boundingBox: {
                    x: detection.bbox[0],
                    y: detection.bbox[1],
                    width: detection.bbox[2],
                    height: detection.bbox[3]
                }
            }));

            // CLASS VALIDATION: keep only plant classes. This rejects a person
            // by CLASS, never by presence - so `person + potted plant` keeps the
            // plant and drops the person. A person can never cancel a plant.
            const classValidated = mapped.filter(d => isValidPlantDetection(d, minConfidence));

            // DUPLICATE FILTER: IoU ONLY for coco-ssd. Two boxes are merged only
            // if they are near-copies (heavy overlap). A smaller plant's box
            // sitting inside a bigger plant's box is NOT merged - which is what
            // lets two nearby-but-separate plants both survive. coco-ssd already
            // ran its own NMS, so this only catches an occasional near-duplicate.
            const { kept: finalDetections, removed: dupsRemoved } = dedupeDetections(classValidated, {
                iouThresh: DETECTION_CONFIG.cocoDedupeIouThreshold,
                useContainment: false
            });

            // ---- Diagnostics -------------------------------------------------
            // PLANT CANDIDATES = every box whose CLASS could be a plant, at ANY
            // score. Comparing this against RAW and against the final count is
            // what distinguishes "the model never saw a plant" from "our filter
            // threw the plant away" - the two have completely different fixes,
            // so the log states which one happened instead of leaving it to
            // guesswork.
            const plantCandidates = mapped.filter(d => isPlantClass(d.className));
            const personDets = mapped.filter(d => d.className.toLowerCase() === 'person');
            const ignored = mapped.filter(d => !isPlantClass(d.className));
            const weakPlants = plantCandidates.filter(
                d => parseFloat(d.confidence) < minConfidence * 100);

            const boxStr = d => `[x:${d.boundingBox.x.toFixed(0)} y:${d.boundingBox.y.toFixed(0)} ` +
                `w:${d.boundingBox.width.toFixed(0)} h:${d.boundingBox.height.toFixed(0)}]`;

            debugLog(`RAW: ${mapped.length} detection(s) at/above the ` +
                `${(DETECTION_CONFIG.observationFloor * 100).toFixed(0)}% observation floor`);
            mapped.forEach(d => debugLog(
                `  class=${d.className} confidence=${(parseFloat(d.confidence) / 100).toFixed(3)} box=${boxStr(d)}`));

            debugLog(`PLANT CANDIDATES: ${plantCandidates.length}` + (plantCandidates.length
                ? '  [' + plantCandidates.map(d => `${d.className} ${d.confidence}%`).join(', ') + ']'
                : ''));
            debugLog(`PERSON DETECTIONS: ${personDets.length}` +
                (personDets.length ? '  (ignored - not a plant class, never cancels a plant)' : ''));
            if (ignored.length) {
                debugLog(`  non-plant objects ignored: ${ignored.length}  [` +
                    ignored.map(d => d.className).join(', ') + ']');
            }
            debugLog(`PLANT CANDIDATES (passed gate ${(minConfidence * 100).toFixed(0)}%): ` +
                `${classValidated.length}` + (classValidated.length
                    ? '  [' + classValidated.map(d => `${d.className} ${d.confidence}% ${boxStr(d)}`).join(' | ') + ']'
                    : ''));
            dupsRemoved.forEach(r => debugLog(
                `DUPLICATE REMOVED: ${r.det.className} ${r.det.confidence}% ${boxStr(r.det)} ` +
                `- ${r.reason} ${r.metric.toFixed(2)} with kept box ${r.against.confidence}% ` +
                `${boxStr(r.against)} (same physical plant)`));
            if (!dupsRemoved.length && classValidated.length > 1) {
                debugLog(`NMS/DEDUPE: ${classValidated.length} plant boxes, none merged ` +
                    `- treated as ${classValidated.length} separate plants`);
            }

            // The verdict, stated explicitly so it never has to be inferred.
            // (FINAL COUNT is logged after temporal confirmation, in detectFrame.)
            if (finalDetections.length === 0) {
                if (plantCandidates.length === 0) {
                    debugLog(`NO PLANT: no potted-plant detection in this frame.`);
                    debugLog(`DIAGNOSIS (A): COCO-SSD emitted NO plant-class box at all, ` +
                        `even down to the ${(DETECTION_CONFIG.observationFloor * 100).toFixed(0)}% ` +
                        `floor. The MODEL did not see a plant - filtering is not involved. ` +
                        `COCO's only plant class is "potted plant", trained on potted ` +
                        `houseplants; foliage on a screen, a cut stem, or a small/distant ` +
                        `plant often falls outside it. Try base="mobilenet_v2" (current: ` +
                        `"${DETECTION_CONFIG.cocoBase}") or move the plant closer/larger in frame.`);
                } else {
                    debugLog(`NO PLANT: candidate confidence below the ` +
                        `${(minConfidence * 100).toFixed(0)}% threshold.`);
                    debugLog(`DIAGNOSIS (B): ${plantCandidates.length} plant-class box(es) WERE ` +
                        `emitted but scored below the ${(minConfidence * 100).toFixed(0)}% gate ` +
                        `[` + weakPlants.map(d => `${d.className} ${d.confidence}%`).join(', ') +
                        `]. This IS a filtering/threshold issue - lower ` +
                        `DETECTION_CONFIG.scoreThreshold['coco-ssd'].`);
                }
            }

            DetectionEngine.lastPipelineStats = {
                engine: 'coco-ssd',
                raw: predictions.length,
                afterConfidence: predictions.length, // coco-ssd thresholds inside .detect()
                afterNms: predictions.length,        // coco-ssd runs its own NMS internally
                afterClass: classValidated.length,
                final: finalDetections.length
            };

            return finalDetections;
        } catch (err) {
            console.error('[AGRIVISION] Detection error:', err);
            return [];
        }
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

            // Confirmed plants are solid green. A confirmed-but-weak plant
            // (score below temporal.strongScore) is drawn dashed amber - it
            // still counts, but the box says the evidence is thin rather than
            // presenting a shaky detection as a solid one.
            const weak = detection.uncertain === true;
            this.ctx.save();
            this.ctx.strokeStyle = weak ? '#f59e0b' : '#10b981';
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash(weak ? [8, 6] : []);
            this.ctx.strokeRect(box.x, box.y, box.width, box.height);
            this.ctx.restore();

            // Draw background for text
            const labelText = `${detection.className} ${detection.confidence}%` +
                (weak ? ' (weak)' : '');
            this.ctx.font = 'bold 14px Arial';
            const textWidth = this.ctx.measureText(labelText).width;
            const textHeight = 24;

            this.ctx.fillStyle = weak ? 'rgba(245, 158, 11, 0.9)' : 'rgba(16, 185, 129, 0.9)';
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

        // NOTE: the diagnostics used to be painted onto this canvas, which meant
        // an opaque black panel sat on top of the video (and scaled with the
        // camera's native resolution, so it was tiny on a 1280x720 feed). They
        // now live in the #debugOverlay DOM element instead - see
        // UIManager.updateDebugOverlay().
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

    // Replaces (never appends) the detection list every inference frame, so the
    // count always reflects plants in the CURRENT frame and never accumulates
    // across frames. `detections` here is already the final, de-duplicated list
    // returned by the engine.
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
        this.updateDebugOverlay(StatsManager.getStats());
    },

    setupEventListeners() {
        document.getElementById('startCameraBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopCameraBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('startDetectionBtn').addEventListener('click', () => this.startDetection());
        document.getElementById('stopDetectionBtn').addEventListener('click', () => this.stopDetection());
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
        PlantTracker.reset();
        this.state.cameraRunning = false;
        this.state.detectionActive = false;
        this.updateAllStatus();
        Visualizer.clear();
        this.updateNoPlantPopup(0); // camera off -> hide the popup
        this.showOverlay('Camera Stopped');
    },

    async startDetection() {
        if (!this.state.cameraRunning) {
            this.showError('Start camera first');
            return;
        }

        if (!this.state.modelReady) {
            const reason = this.state.modelStatus === 'error'
                ? `AI model failed to load (${this.state.modelError || 'unknown error'}).`
                : 'AI model is still loading. Wait a moment and try again.';
            this.showError(reason);
            return;
        }

        // Start from a clean slate - stale tracks from a previous run must not
        // confirm a plant that is no longer in front of the camera.
        PlantTracker.reset();
        this.state.detectionActive = true;
        this.updateAllStatus();
    },

    stopDetection() {
        this.state.detectionActive = false;
        PlantTracker.reset();
        this.updateAllStatus();
        Visualizer.clear();
        StatsManager.detections = [];
        this.updateDetectionUI();          // recomputes count = 0
        this.updateNoPlantPopup(0);        // detection stopped -> hide the popup
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
        let detections = await DetectionEngine.detectFrame(frame);
        this.isDetecting = false;
        if (!Array.isArray(detections)) detections = [];

        // STAGE 2: for each plant Model 1 validated, crop that box out of the
        // frame and analyse it independently. Only runs when Model 2 is
        // installed; skipped entirely otherwise. Detections with no fresh
        // analysis inherit the previous frame's verdict by box overlap.
        if (PlantAnalyzer.available && detections.length) {
            const previous = StatsManager.detections;
            await PlantAnalyzer.analyzeDetections(frame, detections);
            PlantAnalyzer.carryForward(previous, detections);
        }

        // REPLACE (never merge) the detection list every frame. An empty array
        // here fully clears the previous frame's boxes, list, stats and
        // "N PLANTS DETECTED" message - so a no-plant frame right after a
        // plant frame reads "NO PLANT DETECTED", not the stale count.
        StatsManager.updateDetections(detections);
        PerformanceMonitor.recordInferenceFrame();

        Visualizer.drawDetections(detections); // clears the canvas first, then draws only these
        this.updateDetectionUI();
        this.updateStats();
    },

    updateAllStatus() {
        // Header mode badge - the app runs only in live camera mode.
        const modeBadge = document.getElementById('modeBadge');
        modeBadge.textContent = 'LIVE MODE';

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
        if (this.state.modelReady) {
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

        document.getElementById('modeStatus').textContent = 'LIVE';
        document.getElementById('modeStatus').className = 'status-badge';

        const modelLabel = DetectionEngine.engineType === 'custom' ? 'CUSTOM' : 'COCO-SSD';
        document.getElementById('modelValue').textContent = this.state.modelReady ? modelLabel : '--';
    },

    updateDetectionUI() {
        const stats = StatsManager.getStats();
        const detectionSummary = document.getElementById('detectionSummary');
        const detectionsList = document.getElementById('detectionsList');

        // Update summary indicator. The message is driven purely by
        // stats.plantCount, which is StatsManager.detections.length - the final
        // class-validated, de-duplicated plant list for the CURRENT frame. The
        // coloured dot is handled by the .detected / .offline class, so the text
        // is exactly the spec string with no decoration.
        const indicator = detectionSummary.querySelector('.summary-indicator');
        const indicatorText = indicator.querySelector('.indicator-text');
        if (stats.plantCount === 0) {
            indicator.classList.add('offline');
            indicator.classList.remove('detected');
            indicatorText.textContent = 'NO PLANT DETECTED';
        } else {
            indicator.classList.remove('offline');
            indicator.classList.add('detected');
            indicatorText.textContent =
                `${stats.plantCount} PLANT${stats.plantCount === 1 ? '' : 'S'} DETECTED`;
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
                    <div class="detection-head">
                        <span class="detection-name">PLANT ${idx + 1}${
                            det.uncertain ? ' <span class="weak-badge">WEAK</span>' : ''}</span>
                        <span class="detection-confidence">${det.confidence}%</span>
                    </div>
                    ${this.renderAnalysis(det.analysis)}
                </div>
            `).join('');
        }

        // Drive the "NO PLANT DETECTED" popup from the SAME final count.
        this.updateNoPlantPopup(stats.plantCount);
        this.updateDebugOverlay(stats);
    },

    // Model 2's verdict for one plant. Three states, and the module never
    // invents a diagnosis: if Model 2 is not installed we say so, and if it is
    // installed but not confident we say ANALYSIS UNCERTAIN rather than
    // reporting a crop or a disease it cannot stand behind.
    renderAnalysis(analysis) {
        if (!PlantAnalyzer.available) {
            return `<div class="analysis analysis-off">Analysis model not installed</div>`;
        }
        if (!analysis) {
            return `<div class="analysis analysis-off">Analysing…</div>`;
        }
        if (analysis.uncertain) {
            return `
                <div class="analysis analysis-uncertain">
                    <div class="analysis-title">⚠️ ANALYSIS UNCERTAIN</div>
                    <div class="analysis-note">Move closer or steady the camera.</div>
                </div>`;
        }
        const pct = v => `${(v * 100).toFixed(0)}%`;
        const unhealthy = analysis.health === 'Unhealthy';
        return `
            <div class="analysis">
                <div class="analysis-row">
                    <span class="analysis-key">Crop</span>
                    <span class="analysis-val">${analysis.crop}
                        <em>${pct(analysis.cropConfidence)}</em></span>
                </div>
                <div class="analysis-row">
                    <span class="analysis-key">Health</span>
                    <span class="analysis-val ${unhealthy ? 'is-unhealthy' : 'is-healthy'}">
                        ${unhealthy ? '⚠️ ' : ''}${analysis.health.toUpperCase()}
                        <em>${pct(analysis.healthConfidence)}</em></span>
                </div>
                <div class="analysis-row">
                    <span class="analysis-key">${unhealthy ? 'Possible condition' : 'Condition'}</span>
                    <span class="analysis-val">${analysis.condition}
                        <em>${pct(analysis.conditionConfidence)}</em></span>
                </div>
            </div>`;
    },

    // Live blue diagnostics over the camera feed. Read-only view of state that
    // already exists - it does not touch the detection pipeline in any way.
    updateDebugOverlay(stats) {
        const overlay = document.getElementById('debugOverlay');
        if (!overlay) return;

        overlay.hidden = !DETECTION_CONFIG.debugHud;
        if (!DETECTION_CONFIG.debugHud) return;

        const s = DetectionEngine.lastPipelineStats;
        const set = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        };

        const engine = s ? s.engine.toUpperCase() : (DetectionEngine.engineType || '--').toUpperCase();

        set('dbgEngine', engine);
        set('dbgRaw', s ? s.raw : 0);
        // "VALID" is the CANDIDATE level: class-valid, above the gate, deduped -
        // but not yet temporally confirmed. dbgFinal is the CONFIRMED count,
        // which is what the counter and the popup use.
        set('dbgValid', s ? (s.candidates ?? s.afterClass ?? 0) : 0);
        set('dbgFinal', stats.plantCount);
        set('dbgConf', stats.highestConfidence === '--' ? '--' : `${stats.highestConfidence}%`);
        set('dbgTime', `${StatsManager.inferenceTime.toFixed(0)} ms`);

        const statusEl = document.getElementById('dbgStatus');
        if (statusEl) {
            if (!this.state.detectionActive) {
                statusEl.textContent = 'IDLE';
                statusEl.className = 'debug-value';
            } else if (stats.plantCount === 0) {
                statusEl.textContent = 'NO PLANT DETECTED';
                statusEl.className = 'debug-value no-plants';
            } else {
                statusEl.textContent =
                    `${stats.plantCount} PLANT${stats.plantCount === 1 ? '' : 'S'} DETECTED`;
                statusEl.className = 'debug-value has-plants';
            }
        }
    },

    // Popup state - tracked so the debug lines only fire on a transition, not
    // every frame.
    noPlantPopupVisible: false,

    // Shows/hides the popup over the video. Condition (per spec):
    //   camera running  AND  detection active  AND  finalPlantCount === 0
    // It is NOT tied to face/person detection - a person holding a plant gives
    // finalPlantCount >= 1, so the popup stays hidden.
    updateNoPlantPopup(finalPlantCount) {
        const popup = document.getElementById('noPlantPopup');
        if (!popup) return;

        const shouldShow =
            this.state.cameraRunning &&
            this.state.detectionActive &&
            finalPlantCount === 0;

        if (shouldShow === this.noPlantPopupVisible) return; // no change

        this.noPlantPopupVisible = shouldShow;
        popup.hidden = !shouldShow;
        debugLog(`FINAL PLANT COUNT: ${finalPlantCount}`);
        debugLog(shouldShow
            ? 'NO PLANT DETECTED — showing popup'
            : 'Plant detected — hiding NO PLANT popup');
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

    // Model 1: plant detection. Model 2 loads after it and is optional - a
    // missing analysis model must never stop detection from running.
    const modelReady = await DetectionEngine.init();
    await PlantAnalyzer.load();
    UIManager.state.modelReady = modelReady;
    UIManager.state.modelStatus = DetectionEngine.status;
    UIManager.state.modelError = DetectionEngine.errorMessage;
    UIManager.updateAllStatus();
    // Engine name is only known once loading finishes - refresh the blue
    // diagnostics so it reads e.g. "ENGINE: COCO-SSD" before detection starts.
    UIManager.updateDebugOverlay(StatsManager.getStats());

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

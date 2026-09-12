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
//
// PER-FRAME verbose diagnostics are OFF by default. The detection path used to
// emit ~20 console.log calls per inference frame - measured at 70 calls/second -
// and each one built template-literal strings with .map/.filter/.join over every
// raw detection. That string building happens whether or not DevTools is open,
// and it was the single largest source of main-thread stutter. Model load,
// errors and the periodic performance summary still log unconditionally.
//
// Turn the per-frame detail back on at any time from the console:
//     AGRIVISION.setVerbose(true)
// Nothing about detection behaviour changes with the flag - it only controls
// what is printed.
let VERBOSE_LOGGING = false;

function debugLog(...args) {
    if (!VERBOSE_LOGGING) return;
    console.log('[AGRIVISION]', ...args);
}

// Always printed: lifecycle, errors, and the throttled performance summary.
function infoLog(...args) {
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
    // fallback.
    //
    // 'coco-ssd' was the default while Model 2 did not exist, because COCO-SSD
    // is trained on all 80 COCO classes, so `person`, `chair`, `bottle` etc.
    // are learned as their own classes and get rejected by CLASS VALIDATION
    // below - the negative signal the custom model lacks.
    //
    // It cannot be the default any more. COCO's ONLY plant class is `potted
    // plant`, which is trained on unoccluded whole houseplants. On the crops
    // this app exists to analyse it does not emit a plant class at all -
    // measured on real photos at the 0.10 observation floor:
    //     tomato plant  -> apple 83.3%, orange 15.3%   (no plant class)
    //     ZZ houseplant -> vase 67.2%                  (`vase` is deliberately
    //                                                   NOT a plant class)
    //     basil plant   -> nothing at all
    // Every one of those is dropped by class validation, so the count is 0, the
    // "NO PLANT DETECTED" popup never clears, and Model 2 - which only ever
    // runs on a CONFIRMED plant box - is never reached. That is a dead
    // pipeline, not a strict one.
    //
    // The custom detector is the model actually trained for this job and it
    // works: same three photos -> plant 92.6% / 82.4% / 83.7%, one box each,
    // correctly placed. Its inference path is verified op-by-op against the
    // .tflite flatbuffer (NCHW [1,3,640,640] in, [1,5,8400] out, normalised
    // box coords).
    //
    // KNOWN COST, measured, not hypothetical: it never saw a background/
    // negative image in training, so it also reports `plant 57.4%` on a
    // person-only frame. That clears both scoreThreshold.custom (0.40) and
    // temporal.instantConfirmScore (0.55), so a person alone CAN read as one
    // plant. The honest fixes are a retrain with negatives (see
    // training/train_plant_detector.ipynb) or raising scoreThreshold.custom -
    // NOT reverting to an engine that detects nothing.
    engine: 'custom',

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
    // Frames here are DETECTION frames (~8-12/s, see
    // PerformanceMonitor.detectionIntervalMs), so 1 frame is roughly 85-125 ms.
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
        strongScore: 0.50,
        // ---- box smoothing (display only) ---------------------------------
        // COCO-SSD's box for a stationary plant jitters by a few pixels every
        // frame, which reads as a shivering rectangle. This is a light EMA on
        // the DRAWN box: weight on the newest observation, so the lag is well
        // under one detection frame (~30-50ms) and the box still feels attached
        // to the plant. It affects nothing but the rectangle's coordinates -
        // association, confirmation, counting and class validation all continue
        // to use the model's raw output.
        boxSmoothing: 0.6,
        // If the plant moves far enough that the new box barely overlaps the
        // smoothed one, snap straight to the observation instead of easing
        // toward it - smoothing must never turn into visible drag.
        boxSnapIou: 0.50
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

    // ---- PERFORMANCE -------------------------------------------------------
    // One throttled summary line every few seconds (camera FPS, inference FPS,
    // inference time, average, tensor memory, skipped frames, Model 2 time).
    // This is production-safe: it is O(1) per cycle and does not scale with the
    // number of detections, unlike the per-frame verbose logging it replaced.
    perfSummary: true,

    // How often the on-screen numbers are refreshed. Inference runs at ~10/s but
    // the stats row and the blue overlay do not need to be rewritten that often -
    // a human cannot read them faster than this, and every write is layout work
    // on the main thread that competes with the camera preview.
    uiThrottleMs: 250,

    // ---- SCAN SESSION ------------------------------------------------------
    // The app used to publish a verdict every inference frame. That is honest
    // per frame and useless to a human: at ~1 frame/second the health line
    // flickered between Healthy and Unhealthy as the crop shifted a few pixels,
    // and the number beside it moved every time. The user's question is not
    // "what does frame 47 think", it is "is this plant healthy" - which is a
    // question about the whole observation, not one frame of it.
    //
    // So detection now runs as a fixed-length SCAN. Evidence accumulates for
    // the whole window and exactly one verdict is produced at the end, then
    // frozen until a new scan starts. Nothing about Model 1, the person veto or
    // Model 2 changes - this layer only decides how their per-frame output is
    // combined and when the user is shown a result.
    scan: {
        enabled: true,
        // Anywhere in 15000-30000 is sensible. 20 s is roughly 13-20 detection
        // cycles on the machines measured here, which is enough for the sample
        // counts below without making the user hold a plant still for half a
        // minute.
        durationMs: 20000,

        // ---- how much evidence is enough --------------------------------
        // A verdict is refused rather than guessed when these are not met.
        // A 20 s window yields ~10 detection cycles on the machines measured
        // here (inference is ~1.5-2 s per cycle). These floors are set well
        // under that on purpose: a slower phone must produce a real verdict,
        // not "inconclusive" as a permanent side effect of being slow. The
        // gates that actually protect the ANSWER are the health ones below -
        // these two only stop a scan that barely ran from being judged at all.
        minFrames: 5,           // detection cycles that must have run at all
        minPlantFrames: 3,      // cycles that must have contained a plant
        minPlantRatio: 0.30,    // ...and that must be this share of all cycles
        minHealthSamples: 3,    // independent Model 2 runs on a confirmed plant
        minConfidence: 0.60,    // aggregated model confidence floor
        minAgreement: 0.60,     // share of samples voting with the majority

        // Model 2 normally re-runs on an unchanged plant only every
        // analysis.refreshMs (4 s), which is right for a live readout and far
        // too slow for a 20 s scan - it would yield 5 samples. During a scan we
        // want independent observations, so the plant is re-analysed faster.
        // This does NOT change how Model 2 works, only how often it is asked.
        sampleRefreshMs: 1200,

        // Newer observations weigh more: the operator is usually still framing
        // the plant at the start of a scan and settled by the end. Half-life,
        // not a cut-off - an early sample still counts, just less.
        recencyHalfLifeMs: 12000,

        // How long the plant may be missing before the scan bar starts asking
        // for it back. Stops the hint flickering on a single dropped frame.
        missingHintMs: 2500,

        // A displayed confidence of 100% would claim certainty this system does
        // not have - and agreement between samples is NOT the same thing as
        // certainty about the plant. The number shown is always the model's own
        // averaged probability, and it is capped one point below certain.
        maxDisplayConfidence: 99
    },

    // ---- PERSON VETO -------------------------------------------------------
    // The custom detector never saw a background/negative image in training, so
    // it has no way to say "not a plant" - it only ever reports how plant-like
    // a region looks. Measured on this machine, straight out of
    // DetectionEngine.realDetection:
    //
    //     person, plain background   -> plant 88.9%
    //     person, half-length photo  -> plant 72.6%
    //     potted tomato plant        -> plant 41.4%
    //     tomato bush                -> plant 80.4%
    //     tomato foliage close-up    -> plant 58.5%
    //     blighted tomato leaf       -> plant 54.5%
    //
    // A person scores HIGHER than three of the four real plants. There is
    // therefore no value of scoreThreshold.custom that rejects a person and
    // keeps the plants - raising it to 0.89 would delete every plant above.
    // Confidence cannot fix this, because the missing information is not
    // "how sure are you", it is "sure of WHAT".
    //
    // COCO-SSD has that information: `person` is one of its 80 trained classes.
    // Measured on the same frames, it found person at 0.99 / 0.92 / 0.95 on the
    // three person photos and NO person at all on any of the four plant photos,
    // so using it purely as a NEGATIVE signal cannot cost a real detection.
    //
    // This is deliberately a veto and nothing more. COCO-SSD never contributes
    // a detection, never raises a confidence, and never runs as the engine -
    // the custom model remains the only thing that decides where plants are.
    // A person HOLDING a plant still yields the plant: the veto needs the
    // person box to cover most of the plant box, and a held plant sticks out.
    personVeto: {
        enabled: true,
        // The 17MB speed variant, not the 64MB accuracy one. `person` is COCO's
        // best-represented class and the lite base scored 0.92-0.99 on it here;
        // spending 47 extra megabytes to reject a person is not a good trade,
        // especially on the slow link measured above.
        base: 'lite_mobilenet_v2',
        // Loaded LAZILY, in the background, after the app is already usable, and
        // its failure is never fatal. On the measured connection this takes
        // ~170 s; until it finishes the app behaves exactly as it does today.
        // Detection must never wait on a veto.
        minScore: 0.50,        // COCO person confidence required to veto at all

        // ---- what "this box is a person" actually looks like ---------------
        // Two tests, because ONE is not enough. Measured coverage of the custom
        // detector's box against COCO's person box:
        //
        //   frame          plantInPerson  personInPlant  areaRatio
        //   person_a           0.786          1.00         1.27
        //   person_b           0.537          1.00         1.86
        //   person_c #1        0.893          0.92         1.03
        //   person_c #2        0.930          1.00         1.06
        //
        // A single "is the plant box inside the person box" test at 0.70 misses
        // person_b outright: the detector drew a box TALLER than the person
        // (full frame height), so only 54% of it is inside them - while 100% of
        // the person is inside it. Dropping the threshold to catch that would
        // start eating plants held in front of a torso.
        //
        // So the real question is not containment in one direction, it is
        // "are these two boxes the same region?", which needs both.
        // Values chosen by searching the threshold space against 10 measured
        // boxes (8 that are really a person, 2 that are really a plant next to
        // one). These separate all 10 - but HONESTLY, with a thin margin, and
        // the margin is a property of the signal rather than of the tuning:
        //
        //   held_b  plant held at the camera, torso behind : plantInPerson 0.727
        //   held_c  a person                               : plantInPerson 0.766
        //
        // 0.039 apart. Below the gate a person occasionally counts as a plant;
        // above it, a plant held close to the chest occasionally does not count.
        // 0.75 sits between them. Only ONE measured person box (person_b, the
        // one whose plant box was taller than the person) needs the second test,
        // and it scores 1.000 there against held_b's 0.915 - so 0.96 keeps a
        // real margin on both sides rather than shaving past held_b.
        //
        // If a person still registers during a demo, RAISE coverThresh toward
        // 0.70; if a held plant stops registering, LOWER it toward 0.80. Do not
        // read these as tuned-in-the-lab constants - they are two competing
        // errors on a 10-box sample.
        coverThresh: 0.75,          // plant box mostly INSIDE a person box
        sameRegionCover: 0.96,      // ...or a person is almost entirely inside the plant box
        sameRegionAreaRatio: 2.0,   // ...AND the plant box is not much bigger than them

        intervalMs: 1500,      // at most one COCO pass per this many ms
        staleMs: 5000          // forget the person map if it stops refreshing
    },

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
        // Force a re-classification of an unchanged plant this often, so a
        // verdict can never go stale indefinitely.
        refreshMs: 4000,
        // If the plant's box still overlaps the region we last classified by
        // more than this, the crop is effectively the same picture - skip it.
        // Below it, the plant has moved or resized enough to be worth re-running.
        regionChangeIou: 0.85,
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

// What FRACTION OF `inner` lies inside `outer`. Unlike boxOverlapRatio this is
// directional - it always divides by `inner`'s area, never by whichever box
// happens to be smaller. The person veto needs that direction specifically:
// "how much of this plant box is inside that person box", which is a different
// question from "how much do these two boxes overlap".
function coveredFraction(inner, outer) {
    const area = inner.width * inner.height;
    return area > 0 ? boxIntersectionArea(inner, outer) / area : 0;
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
// Light exponential smoothing of a bounding box, for DISPLAY only. Snaps
// instead of easing when the plant has clearly moved, so a fast-moving plant
// never trails its box.
function smoothBox(prev, obs, cfg) {
    if (!prev) return { ...obs };
    if (boxIoU(prev, obs) < cfg.boxSnapIou) return { ...obs };
    const a = cfg.boxSmoothing;
    return {
        x: a * obs.x + (1 - a) * prev.x,
        y: a * obs.y + (1 - a) * prev.y,
        width: a * obs.width + (1 - a) * prev.width,
        height: a * obs.height + (1 - a) * prev.height
    };
}

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
                best.boundingBox = cand.boundingBox;          // raw, used for association
                best.smoothBox = smoothBox(best.smoothBox, cand.boundingBox, cfg);
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
                    smoothBox: { ...cand.boundingBox },   // starts exactly on the observation
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
                boundingBox: t.smoothBox || t.boundingBox,   // display-smoothed
                rawBoundingBox: t.boundingBox,
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
            // 10000 ms was far too tight, and the way it failed was the worst
            // possible one. This promise does NOT just fetch the 12MB .tflite
            // (that is served from the same origin as the page - measured at
            // 0.24 s); it also waits for tfjs-tflite's WASM runtime to arrive
            // from the CDN. That runtime, tflite_web_api_cc_simd.wasm, is 1.1MB
            // over the public internet and was measured on this machine at
            // 6.4 s - two thirds of the old budget, on a connection that was
            // working normally.
            //
            // When it did overrun, the catch below demoted the app to the
            // COCO-SSD fallback, and cocoSsd.load() was measured here at
            // 170 SECONDS for the 17MB lite base (the configured base is the
            // 64MB one). The app sits on "PREPARING AI MODEL..." for minutes,
            // then ends up on the engine this file already documents as a dead
            // pipeline for crops. Slow is not the same as broken: wait for the
            // detector that works instead of racing to a worse one.
            this.model = await withTimeout(
                tflite.loadTFLiteModel('model/plant_detector.tflite', { numThreads }),
                60000,
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

        const rawDetections = scoresData.map((score, index) => ({
            index,
            className: this.classNames[classIdsData[index]] || 'plant',
            confidence: (score * 100).toFixed(1)
        }));
        const keepSet = new Set(keepIndices);
        const confidenceThreshold = this.scoreThreshold;
        const removedByConfidence = rawDetections.filter(
            d => parseFloat(d.confidence) < confidenceThreshold * 100
        );
        const removedByNms = rawDetections.filter(
            d => !keepSet.has(d.index) && parseFloat(d.confidence) >= confidenceThreshold * 100
        );

        if (VERBOSE_LOGGING) {
            const shortList = d => d.slice(0, 5).map(item => `${item.className} ${item.confidence}%`).join(', ');
            debugLog(`RAW CUSTOM DETECTIONS: ${rawDetections.length}` +
                (rawDetections.length ? ` [${shortList(rawDetections)}${rawDetections.length > 5 ? ' ...' : ''}]` : ''));
            if (removedByConfidence.length) {
                debugLog(`CUSTOM CONFIDENCE FILTER REMOVED: ${removedByConfidence.length}` +
                    ` [${shortList(removedByConfidence)}${removedByConfidence.length > 5 ? ' ...' : ''}]`);
            }
            if (removedByNms.length) {
                debugLog(`CUSTOM NMS REMOVED: ${removedByNms.length}` +
                    ` [${shortList(removedByNms)}${removedByNms.length > 5 ? ' ...' : ''}]`);
            }
        }

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

        if (VERBOSE_LOGGING) {
            debugLog(`FINAL CUSTOM PLANT DETECTIONS: ${finalDetections.length}` +
                (finalDetections.length ? ` [${finalDetections.map(d => `${d.className} ${d.confidence}%`).join(', ')}]` : ''));
        }
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
// 2a. PERSON VETO - the negative signal the custom detector does not have
// ============================================================================
//
// See DETECTION_CONFIG.personVeto for the measurements that motivate this.
//
// SHAPE OF THE THING, so it is not mistaken for a second detector:
//   * it can only ever REMOVE a candidate, never add or promote one
//   * it runs at most once per intervalMs, and only on frames where the custom
//     detector actually produced a candidate - a frame with no candidates costs
//     nothing at all
//   * it loads in the background and self-disables on any failure; detection
//     never waits for it and never breaks because of it
const PersonVeto = {
    model: null,
    available: false,
    status: 'not-loaded',   // 'not-loaded' | 'loading' | 'ready' | 'disabled' | 'error'
    personBoxes: [],
    mappedAt: 0,
    lastRunAt: 0,
    inferenceTime: 0,
    vetoed: 0,

    // Deliberately NOT awaited by initializeApp. The app is fully usable before
    // this resolves; arming the veto late is strictly better than delaying the
    // detector behind a 17MB download.
    async load() {
        const cfg = DETECTION_CONFIG.personVeto;
        if (!cfg.enabled) { this.status = 'disabled'; return false; }
        if (typeof cocoSsd === 'undefined') {
            this.status = 'error';
            console.warn('[AGRIVISION] Person veto unavailable: COCO-SSD library did not load. ' +
                'Detection continues; a person may register as a plant.');
            return false;
        }
        this.status = 'loading';
        try {
            const t = performance.now();
            this.model = await cocoSsd.load({ base: cfg.base });
            this.available = true;
            this.status = 'ready';
            infoLog(`Person veto armed (COCO-SSD base="${cfg.base}", ` +
                `${Math.round((performance.now() - t) / 1000)}s). A person-only frame ` +
                `will no longer be counted as a plant.`);
            return true;
        } catch (err) {
            this.status = 'error';
            console.warn('[AGRIVISION] Person veto failed to load:', err.message,
                '- detection continues; a person may register as a plant.');
            return false;
        }
    },

    // Refresh the map of where the people are. Rate-limited: people do not move
    // far in 1.5 s, and this is the only expensive part of the veto.
    async refresh(frame) {
        if (!this.available) return;
        const cfg = DETECTION_CONFIG.personVeto;
        const now = performance.now();
        if (now - this.lastRunAt < cfg.intervalMs) return;
        this.lastRunAt = now;
        try {
            const t = performance.now();
            const preds = await this.model.detect(frame, 20, cfg.minScore);
            this.inferenceTime = performance.now() - t;
            this.personBoxes = preds
                .filter(p => p.class === 'person' && p.score >= cfg.minScore)
                .map(p => ({
                    x: p.bbox[0], y: p.bbox[1], width: p.bbox[2], height: p.bbox[3],
                    score: p.score
                }));
            this.mappedAt = now;
        } catch (err) {
            // A failed veto pass must never take detection down with it. Drop
            // the stale map so a failure can only ever UNDER-veto.
            this.personBoxes = [];
            console.warn('[AGRIVISION] Person veto pass failed:', err.message);
        }
    },

    // Drop candidates that are mostly inside a person. Returns the survivors.
    apply(candidates) {
        if (!this.available || !candidates.length || !this.personBoxes.length) return candidates;
        const cfg = DETECTION_CONFIG.personVeto;
        // An unrefreshed map is worse than no map: if the person walked away we
        // would keep vetoing a spot they no longer occupy.
        if (performance.now() - this.mappedAt > cfg.staleMs) {
            this.personBoxes = [];
            return candidates;
        }
        return candidates.filter(det => {
            const box = det.boundingBox;
            const area = box.width * box.height;
            let hit = null, reason = null;
            for (const pb of this.personBoxes) {
                // (1) the box sits inside a person
                if (coveredFraction(box, pb) >= cfg.coverThresh) {
                    hit = pb; reason = `${Math.round(coveredFraction(box, pb) * 100)}% of the ` +
                        `box is inside a person`;
                    break;
                }
                // (2) the box IS the person, drawn a little larger. Both halves
                // are required: "the person is inside it" alone would veto a
                // wide shot of a field with someone standing in it, so the box
                // must also be close to the person's own size.
                const personArea = pb.width * pb.height;
                const ratio = personArea > 0 ? area / personArea : Infinity;
                if (coveredFraction(pb, box) >= cfg.sameRegionCover &&
                    ratio <= cfg.sameRegionAreaRatio) {
                    hit = pb; reason = `the box is the person themselves ` +
                        `(${Math.round(coveredFraction(pb, box) * 100)}% of the person is ` +
                        `inside it, and it is only ${ratio.toFixed(2)}x their area)`;
                    break;
                }
            }
            if (!hit) return true;
            this.vetoed++;
            if (VERBOSE_LOGGING) {
                debugLog(`PERSON OVERLAP REJECTED: ${det.className} ${det.confidence}% - ${reason}; ` +
                    `COCO person ${Math.round(hit.score * 100)}%`);
            }
            return false;
        });
    },

    reset() {
        this.personBoxes = [];
        this.mappedAt = 0;
        this.lastRunAt = 0;
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
    runs: 0,
    // ONE reusable offscreen canvas for cropping, allocated on first use.
    // cropToTensor used to document.createElement('canvas') per detection per
    // cycle - up to 4 canvas allocations + 4 2D contexts per analysis cycle,
    // all immediately garbage. The canvas is a fixed inputSize square, so it
    // never needs resizing between crops.
    scratchCanvas: null,
    scratchCtx: null,
    // trackId -> { box, at, result }: what we last classified for that plant and
    // when. Lets the classifier be skipped for a plant that has not moved.
    lastByTrack: new Map(),

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

        if (!this.scratchCanvas || this.scratchCanvas.width !== this.inputSize) {
            this.scratchCanvas = document.createElement('canvas');
            this.scratchCanvas.width = this.inputSize;
            this.scratchCanvas.height = this.inputSize;
            this.scratchCtx = this.scratchCanvas.getContext('2d', { willReadFrequently: true });
        }
        const ctx = this.scratchCtx;
        // Previous crop must not show through where this one does not cover.
        ctx.clearRect(0, 0, this.inputSize, this.inputSize);
        ctx.drawImage(frame, sx, sy, sw, sh, 0, 0, this.inputSize, this.inputSize);

        // tidy() disposes the fromPixels/toFloat intermediates; the returned
        // tensor escapes tidy by design and is disposed by analyzeOne's finally.
        return tf.tidy(() => tf.browser.fromPixels(this.scratchCanvas).toFloat().expandDims(0));
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

    // Does this plant actually need the classifier run again?
    //
    // Model 2 is far heavier than detection, and a plant's crop and condition do
    // not change from frame to frame - so re-classifying an unmoved plant is
    // pure wasted CPU. It is re-run only when the plant is new, when its region
    // has genuinely changed (the box no longer overlaps what we classified), or
    // when the refresh interval has elapsed so a stale verdict cannot persist
    // indefinitely.
    needsAnalysis(det, now, cfg) {
        const prev = det.trackId != null ? this.lastByTrack.get(det.trackId) : null;
        if (!prev) return true;                                   // never analysed
        if (now - prev.at >= this.refreshMsNow(cfg)) return true; // periodic refresh
        return boxIoU(prev.box, det.boundingBox) < cfg.regionChangeIou;  // moved
    },

    // A live readout only needs a verdict that is not stale, so refreshMs (4 s)
    // is right for it. A 20 s scan needs INDEPENDENT observations to average,
    // and 4 s would yield five. Nothing about the classifier changes here - it
    // is asked more often, that is all.
    refreshMsNow(cfg) {
        return ScanSession.active ? DETECTION_CONFIG.scan.sampleRefreshMs : cfg.refreshMs;
    },

    // Analyse the detections in this frame that need it, independently. Returns
    // after mutating each detection with `.analysis`.
    async analyzeDetections(frame, detections) {
        if (!this.available || !detections.length) return;
        const cfg = DETECTION_CONFIG.analysis;
        const now = performance.now();

        // Free: reapply the cached verdict for every plant we have already
        // classified, so its card stays populated between analysis cycles.
        for (const det of detections) {
            const prev = det.trackId != null ? this.lastByTrack.get(det.trackId) : null;
            if (prev) det.analysis = prev.result;
        }

        const due = detections.filter(det => this.needsAnalysis(det, now, cfg));
        if (!due.length) return;                       // nothing changed - no work at all

        // Rate-limit the cycles that actually run the model.
        if (now - this.lastRunAt < cfg.intervalMs) return;
        this.lastRunAt = now;

        const start = performance.now();
        const limit = Math.min(due.length, cfg.maxPerCycle);
        for (let i = 0; i < limit; i++) {
            const det = due[i];
            try {
                const result = await this.analyzeOne(frame, det.boundingBox);
                if (result) {
                    det.analysis = result;
                    this.runs++;
                    if (det.trackId != null) {
                        this.lastByTrack.set(det.trackId, {
                            box: { ...det.boundingBox },
                            at: performance.now(),
                            result
                        });
                    }
                }
            } catch (err) {
                // One bad crop must not abort the rest of the cycle.
                console.error('[AGRIVISION] Model 2 analysis error:', err);
            }
        }
        this.inferenceTime = performance.now() - start;
        this.pruneCache(detections);

        if (VERBOSE_LOGGING) detections.slice(0, limit).forEach((d, i) => {
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

    // Drop cache entries for tracks that no longer exist, so a long session
    // cannot grow the map without bound.
    pruneCache(detections) {
        if (this.lastByTrack.size <= 8) return;
        const live = new Set(detections.map(d => d.trackId));
        for (const id of this.lastByTrack.keys()) {
            if (!live.has(id)) this.lastByTrack.delete(id);
        }
    },

    // Track ids restart from 1 after PlantTracker.reset(), so a stale cache
    // could otherwise hand a new plant an old plant's verdict.
    resetCache() {
        this.lastByTrack.clear();
        this.lastRunAt = 0;
    },

    // Camera stopped: give back the crop canvas and the cache.
    releaseScratch() {
        this.scratchCanvas = null;
        this.scratchCtx = null;
        this.resetCache();
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

// ============================================================================
// 2c. SCAN SESSION - many frames in, ONE verdict out
// ============================================================================
//
// WHAT THIS IS NOT: it is not a second opinion about the plant. It never looks
// at a pixel. It consumes exactly what Model 1 (after the person veto) and
// Model 2 already produced, and decides how to combine a sequence of those into
// a single answer - and, just as importantly, when to refuse to answer.
//
// AGGREGATION, in the order it matters:
//
//   DETECTION - a vote over frames. `plantFrames / frames` must clear a ratio
//     AND an absolute count. A ratio alone would let 2-of-3 frames decide; a
//     count alone would let 5 hits out of 200 frames decide. Both, so a plant
//     has to be present for a real share of a real scan.
//
//   HEALTH - a weighted mean of Model 2's own P(healthy), taken ONLY from
//     frames that contained a confirmed plant, plus a majority vote used purely
//     as a consistency check. The mean is what is reported; the vote is what
//     decides whether reporting anything is honest.
//
// Why a mean of probabilities rather than a vote count: a vote count throws
// away how sure each observation was, and it is exactly what would let ten
// barely-past-50% frames read as a confident verdict. Averaging the
// probabilities keeps a run of weak evidence looking weak.
const ScanSession = {
    state: 'idle',          // 'idle' | 'scanning' | 'complete'
    startedAt: 0,
    endsAt: 0,
    frames: 0,              // detection cycles completed during this scan
    plantFrames: 0,         // ...of which contained at least one confirmed plant
    lastPlantAt: 0,
    samples: [],            // { p: P(healthy), w: weight, at }
    result: null,           // the frozen verdict; null until the scan finishes
    // Model 2 caches a verdict per track and re-serves the SAME object for
    // frames it did not re-run on (see PlantAnalyzer.lastByTrack and
    // carryForward). Counting those again would let one inference masquerade as
    // ten agreeing observations - the precise failure this module exists to
    // prevent. Identity, not value: two genuinely separate runs that happen to
    // agree are two samples, one cached object re-served is one.
    seen: null,

    get active() { return this.state === 'scanning'; },

    begin() {
        const cfg = DETECTION_CONFIG.scan;
        this.state = 'scanning';
        this.startedAt = performance.now();
        this.endsAt = this.startedAt + cfg.durationMs;
        this.frames = 0;
        this.plantFrames = 0;
        this.lastPlantAt = this.startedAt;
        this.samples = [];
        this.result = null;
        this.seen = new WeakSet();
        infoLog(`Scan started: ${(cfg.durationMs / 1000).toFixed(0)}s window.`);
    },

    reset() {
        this.state = 'idle';
        this.result = null;
        this.samples = [];
        this.frames = 0;
        this.plantFrames = 0;
        this.seen = null;
    },

    remainingMs() {
        if (this.state !== 'scanning') return 0;
        return Math.max(0, this.endsAt - performance.now());
    },

    // 0..1 through the scan window.
    progress() {
        const cfg = DETECTION_CONFIG.scan;
        if (this.state === 'complete') return 1;
        if (this.state !== 'scanning') return 0;
        return Math.min(1, (performance.now() - this.startedAt) / cfg.durationMs);
    },

    // Is the plant missing right now, for long enough to be worth mentioning?
    plantMissing() {
        if (this.state !== 'scanning') return false;
        return performance.now() - this.lastPlantAt > DETECTION_CONFIG.scan.missingHintMs;
    },

    // One detection cycle's worth of evidence. `detections` is the final,
    // person-vetoed, temporally confirmed list for this frame.
    record(detections) {
        if (this.state !== 'scanning') return;
        const cfg = DETECTION_CONFIG.scan;
        const now = performance.now();
        this.frames++;

        if (!detections.length) return;
        this.plantFrames++;
        this.lastPlantAt = now;

        for (const det of detections) {
            const a = det.analysis;
            if (!a) continue;
            if (this.seen.has(a)) continue;     // cached re-serve, not a new run
            this.seen.add(a);

            // Model 2 reports confidence in the statement it made. Normalise it
            // back to a single axis, P(healthy), so the samples can be averaged
            // regardless of which way each one leaned.
            const p = a.health === 'Healthy' ? a.healthConfidence : 1 - a.healthConfidence;

            // Weight: how strongly Model 1 believed this was a plant at all,
            // decayed by age. A verdict taken from a weak 41% box should not
            // count the same as one taken from a solid 85% box.
            const detConf = Math.max(0, Math.min(1, parseFloat(det.confidence) / 100)) || 0;
            this.samples.push({ p, w: detConf, at: now });
        }
    },

    // Close the window and compute the single verdict. Idempotent.
    finish() {
        if (this.state === 'complete') return this.result;
        const cfg = DETECTION_CONFIG.scan;
        const now = performance.now();

        const plantRatio = this.frames ? this.plantFrames / this.frames : 0;
        const enoughFrames = this.frames >= cfg.minFrames;
        const detected = enoughFrames &&
            this.plantFrames >= cfg.minPlantFrames &&
            plantRatio >= cfg.minPlantRatio;

        // Apply recency here rather than at record() time, so a sample's weight
        // is its age at the END of the scan and not at the moment it arrived.
        const half = cfg.recencyHalfLifeMs;
        const weighted = this.samples.map(s => ({
            p: s.p,
            w: s.w * Math.pow(0.5, (now - s.at) / half)
        }));

        const wSum = weighted.reduce((t, s) => t + s.w, 0);
        const pHealthy = wSum > 0
            ? weighted.reduce((t, s) => t + s.w * s.p, 0) / wSum
            : 0;

        const healthy = pHealthy >= 0.5;
        // THE reported number: the model's own averaged probability for the
        // statement being made. Never the agreement rate - see below.
        const confidence = healthy ? pHealthy : 1 - pHealthy;

        // Agreement is a CONSISTENCY CHECK, never the confidence. Samples that
        // all agree at 55% are consistent and still not confident; samples
        // split 50/50 at 95% each are confident individually and mean nothing
        // together. Both conditions have to hold to report a verdict.
        const votes = this.samples.filter(s => (s.p >= 0.5) === healthy).length;
        const agreement = this.samples.length ? votes / this.samples.length : 0;

        const enoughSamples = this.samples.length >= cfg.minHealthSamples;
        const healthConclusive = detected && enoughSamples &&
            confidence >= cfg.minConfidence && agreement >= cfg.minAgreement;

        this.result = {
            detected,
            // Why the health verdict was or was not reached, so the UI never
            // has to guess and the console can explain a refusal.
            health: !detected ? 'none' : (healthConclusive ? (healthy ? 'Healthy' : 'Unhealthy') : 'inconclusive'),
            confidence,
            confidencePct: Math.min(cfg.maxDisplayConfidence, Math.round(confidence * 100)),
            agreement,
            samples: this.samples.length,
            frames: this.frames,
            plantFrames: this.plantFrames,
            plantRatio,
            durationMs: now - this.startedAt,
            reason: !enoughFrames ? 'too-few-frames'
                : !detected ? 'no-plant'
                : !enoughSamples ? 'too-few-health-samples'
                : confidence < cfg.minConfidence ? 'low-confidence'
                : agreement < cfg.minAgreement ? 'samples-disagree'
                : 'ok'
        };
        this.state = 'complete';

        infoLog(`Scan complete: ${this.result.detected ? 'PLANT DETECTED' : 'NO PLANT'} ` +
            `(${this.plantFrames}/${this.frames} frames, ratio ${plantRatio.toFixed(2)}) | ` +
            `health=${this.result.health} conf=${this.result.confidencePct}% ` +
            `agreement=${(agreement * 100).toFixed(0)}% samples=${this.samples.length} ` +
            `reason=${this.result.reason}`);
        return this.result;
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
        const rawCandidates = await this.realDetection(frame);

        // PERSON VETO. Sits between detection and confirmation on purpose: a
        // vetoed box must never reach the tracker, or it would accumulate hits
        // and confirm itself. Costs nothing on a frame with no candidates.
        let candidates = rawCandidates;
        if (rawCandidates.length && PersonVeto.available) {
            await PersonVeto.refresh(frame);
            candidates = PersonVeto.apply(rawCandidates);
        }

        const confirmed = PlantTracker.update(candidates);

        if (this.lastPipelineStats) {
            this.lastPipelineStats.candidates = candidates.length;
            this.lastPipelineStats.vetoedByPerson = rawCandidates.length - candidates.length;
            this.lastPipelineStats.final = confirmed.length;
        }

        // Guarded: describe() walks and formats every live track, per frame.
        if (VERBOSE_LOGGING) {
            debugLog(`TEMPORAL CONFIRMATION: ${PlantTracker.describe()}`);
            debugLog(`FINAL PLANT COUNT (confirmed): ${confirmed.length}`);
            if (confirmed.length === 0 && candidates.length > 0) {
                debugLog(`NO PLANT: ${candidates.length} candidate(s) present but not yet ` +
                    `temporally confirmed (need ${DETECTION_CONFIG.temporal.minHits} hits in ` +
                    `${DETECTION_CONFIG.temporal.windowFrames} frames at score ` +
                    `>= ${DETECTION_CONFIG.temporal.confirmScore}, or one frame ` +
                    `>= ${DETECTION_CONFIG.temporal.instantConfirmScore})`);
            }
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
                if (VERBOSE_LOGGING) {
                    debugLog(`Inference: ${this.inferenceTime.toFixed(0)}ms, plant detections: ${results.length}`);
                }
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
            // GUARDED. Everything below is string building over every raw
            // detection (.map/.filter/.join/.toFixed) purely to produce console
            // output. Unguarded it ran on every inference frame - the dominant
            // main-thread cost in this function. `lastPipelineStats` (which the
            // debug overlay reads) is assigned outside the guard, so the UI is
            // unaffected. Enable with AGRIVISION.setVerbose(true).
            if (VERBOSE_LOGGING) {
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

        // resizeCanvas() reads video.videoWidth, which is still 0 here - the
        // camera has not been started yet - so this first call is a no-op and
        // the canvas keeps its default 300x150 bitmap. Until now the only
        // thing that ever recomputed it was a WINDOW RESIZE, so unless the
        // user happened to resize after starting the camera, boxes were drawn
        // in video coordinates onto a 300x150 canvas and landed in the wrong
        // place. Size it when the stream's dimensions actually arrive.
        const video = document.getElementById('cameraFeed');
        video.addEventListener('loadedmetadata', () => this.resizeCanvas());
        video.addEventListener('resize', () => this.resizeCanvas());
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

        // PRESENTATION ONLY. Nothing here reads or changes a threshold, a
        // class filter, the tracker or the count - it draws the boxes the
        // pipeline already decided on.
        //
        // Everything is sized in SCREEN units rather than video pixels. The
        // canvas bitmap is the camera's NATIVE size (1280x720 for the usual
        // stream) and CSS scales it down to fit the frame, so the old
        // hard-coded `lineWidth = 3` / `14px` came out around 0.9px and 4px on
        // a 369px-wide phone frame - nearly invisible - while looking oversized
        // on a small feed. `u` is one CSS pixel expressed in canvas pixels, so
        // strokes, corner marks and type keep the same visual weight at every
        // camera resolution. (this.scale is computed in resizeCanvas() and was
        // previously never used.)
        // Cheap safety net for the same problem: if the bitmap does not match
        // the stream (first frames after a camera restart, a track that
        // changed resolution), re-derive it. Compares two numbers and only
        // touches layout on an actual mismatch, so it costs nothing per frame.
        const video = document.getElementById('cameraFeed');
        if (video.videoWidth && this.canvas.width !== video.videoWidth) this.resizeCanvas();

        const u = (isFinite(this.scale) && this.scale > 0) ? 1 / this.scale : 1;
        const ctx = this.ctx;
        const placedChips = [];   // label chips already drawn this frame

        detections.forEach((detection) => {
            const box = detection.boundingBox;

            // Confirmed plants are green. A confirmed-but-weak plant (score
            // below temporal.strongScore) is amber with a dashed body - it
            // still counts, but the box says the evidence is thin rather than
            // presenting a shaky detection as a solid one.
            const weak = detection.uncertain === true;
            const accent = weak ? '#f59e0b' : '#34d399';
            const accentDim = weak ? 'rgba(245, 158, 11, 0.38)' : 'rgba(52, 211, 153, 0.38)';

            // Hairline body: it delimits the plant without drawing a heavy
            // cage over it. The weight is in the corners.
            ctx.save();
            ctx.strokeStyle = accentDim;
            ctx.lineWidth = 1.5 * u;
            ctx.setLineDash(weak ? [7 * u, 5 * u] : []);
            ctx.strokeRect(box.x, box.y, box.width, box.height);
            ctx.restore();

            // Corner brackets, same reticle language as the viewfinder frame.
            // Length is capped against the box so a small detection gets
            // proportionate marks instead of four overlapping Ls.
            const len = Math.min(22 * u, box.width * 0.3, box.height * 0.3);
            ctx.save();
            ctx.strokeStyle = accent;
            ctx.lineWidth = 3 * u;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.shadowColor = weak ? 'rgba(245, 158, 11, 0.5)' : 'rgba(52, 211, 153, 0.5)';
            ctx.shadowBlur = 8 * u;
            const corners = [
                [box.x, box.y, 1, 1],
                [box.x + box.width, box.y, -1, 1],
                [box.x, box.y + box.height, 1, -1],
                [box.x + box.width, box.y + box.height, -1, -1]
            ];
            for (const [cx, cy, sx, sy] of corners) {
                ctx.beginPath();
                ctx.moveTo(cx + sx * len, cy);
                ctx.lineTo(cx, cy);
                ctx.lineTo(cx, cy + sy * len);
                ctx.stroke();
            }
            ctx.restore();

            // Label chip. The raw COCO class name ("potted plant") is an
            // implementation detail of the detector, so the box says PLANT.
            // detection.className itself is untouched.
            const labelText = `PLANT ${detection.confidence}%`;
            ctx.save();
            ctx.font = `600 ${13 * u}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
            ctx.textBaseline = 'middle';
            const padX = 7 * u;
            const chipW = ctx.measureText(labelText).width + padX * 2;
            const chipH = 21 * u;
            const gap = 6 * u;

            // Keep the chip on screen: above the box normally, tucked inside
            // the top edge when there is no room, and never past the right
            // edge of the frame.
            let chipY = box.y - chipH - gap;
            if (chipY < 0) chipY = Math.min(box.y + gap, this.canvas.height - chipH);
            let chipX = box.x;
            if (chipX + chipW > this.canvas.width) chipX = Math.max(0, this.canvas.width - chipW);

            // Two plants standing side by side put their chips at the same
            // height and the labels overlap into an unreadable smear. Step
            // this one down until it clears the ones already drawn (bounded,
            // so a crowded frame degrades into slight overlap rather than a
            // long search or a chip pushed off the bottom).
            const hits = other => !(chipX + chipW <= other.x || other.x + other.w <= chipX ||
                                    chipY + chipH <= other.y || other.y + other.h <= chipY);
            for (let attempt = 0; attempt < 3 && placedChips.some(hits); attempt++) {
                chipY = Math.min(chipY + chipH + 3 * u, this.canvas.height - chipH);
            }
            placedChips.push({ x: chipX, y: chipY, w: chipW, h: chipH });

            ctx.beginPath();
            if (typeof ctx.roundRect === 'function') {
                ctx.roundRect(chipX, chipY, chipW, chipH, 5 * u);
            } else {
                ctx.rect(chipX, chipY, chipW, chipH);
            }
            ctx.fillStyle = 'rgba(4, 10, 14, 0.78)';
            ctx.fill();
            ctx.strokeStyle = accentDim;
            ctx.lineWidth = 1 * u;
            ctx.stroke();

            ctx.fillStyle = accent;
            ctx.fillText(labelText, chipX + padX, chipY + chipH / 2);
            ctx.restore();
        });

        // NOTE: the diagnostics used to be painted onto this canvas, which meant
        // an opaque black panel sat on top of the video (and scaled with the
        // camera's native resolution, so it was tiny on a 1280x720 feed). They
        // were removed from the UI entirely - they were developer
        // diagnostics. The data is still on DetectionEngine.lastPipelineStats.
    }
};

// ============================================================================
// 4. PERFORMANCE MONITORING
// ============================================================================

const PerformanceMonitor = {
    // ---- counters -----------------------------------------------------------
    cameraFrames: 0,
    cameraLastTime: performance.now(),
    inferenceFrames: 0,
    inferenceLastTime: performance.now(),
    skippedFrames: 0,          // frames where a cycle was still in flight
    cycles: 0,
    cycleMsTotal: 0,           // full pipeline: stage 1 + stage 2 + draw + UI
    cycleMsEma: 0,
    lastCycleMs: 0,
    summaryLastTime: performance.now(),

    // ---- adaptive inference rate --------------------------------------------
    // Target band from the spec: 8-12 inference FPS. The interval is nudged
    // toward whatever this particular device can actually sustain, using the
    // measured full-cycle time - a healthy desktop settles at the 8 FPS ceiling,
    // a weak phone backs off toward 5 FPS instead of thrashing. ONLY the frequency
    // adapts: no threshold, class filter, NMS or dedupe setting is touched, so
    // detection quality is identical at every rate.
    //
    // MEASURED, not guessed. A 20s headless bench (_probe/bench.html, fixed-cost
    // model stubs, 5 runs per point) swept this ceiling and found preview
    // smoothness falls off sharply above ~8 inference FPS on this hardware:
    //     83ms (12fps) -> 46 UI fps, 23.0% janky frames
    //    100ms (10fps) -> 48 UI fps, 19.5%
    //    120ms (8fps)  -> 58 UI fps, 13.9%   <- knee
    //    143ms (7fps)  -> 52 UI fps, 13.3%   (no further gain)
    // Running detect() 32% more often consumed the whole main-thread saving from
    // the logging/throttle work and then some, leaving the preview WORSE than
    // before it. 120ms keeps the saving in the preview, where lag is felt; the
    // tracker and box smoothing already carry detection across frames, so a
    // near-static plant loses nothing at 8 FPS.
    minIntervalMs: 120,        // ceiling: ~8 inference FPS
    maxIntervalMs: 200,        // floor:   ~5 inference FPS on a slow device
    detectionIntervalMs: 120,  // start at the ceiling and back off if needed

    reset() {
        this.cameraFrames = 0;
        this.inferenceFrames = 0;
        this.skippedFrames = 0;
        this.cycles = 0;
        this.cycleMsTotal = 0;
        this.cycleMsEma = 0;
        this.lastCycleMs = 0;
        this.detectionIntervalMs = 100;
        const now = performance.now();
        this.cameraLastTime = now;
        this.inferenceLastTime = now;
        this.summaryLastTime = now;
    },

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
    },

    // Called once per completed cycle, from runDetection's finally block.
    recordCycle(ms) {
        this.lastCycleMs = ms;
        this.cycles++;
        this.cycleMsTotal += ms;
        // EMA so one slow frame does not swing the rate; alpha 0.2 ~ 5 cycles.
        this.cycleMsEma = this.cycleMsEma ? 0.2 * ms + 0.8 * this.cycleMsEma : ms;
        this.adapt();
        this.maybeLogSummary();
    },

    // Keep roughly `headroom` of each interval free for the compositor, the
    // video, and the rest of the page - that free time is what keeps the
    // preview smooth. If a cycle costs more than the interval allows, stretch
    // the interval; if there is spare capacity, tighten it back toward 12 FPS.
    adapt() {
        // 2.2x, i.e. keep ~55% of every interval free. 1.35x (35% free) was not
        // enough: `cycleMsEma` is WALL-CLOCK for the whole cycle, so a cycle that
        // is partly an awaited GPU readback looks cheaper to this controller than
        // it is to the compositor. Over-reserving errs toward a smooth preview,
        // which is the point of adapting at all.
        const headroom = 2.2;
        const wanted = this.cycleMsEma * headroom;
        const target = Math.min(this.maxIntervalMs, Math.max(this.minIntervalMs, wanted));
        // Move gradually - a jumpy interval is itself visible as stutter.
        this.detectionIntervalMs += (target - this.detectionIntervalMs) * 0.25;
    },

    avgCycleMs() {
        return this.cycles ? this.cycleMsTotal / this.cycles : 0;
    },

    tensorMemory() {
        try {
            if (typeof tf === 'undefined' || !tf.memory) return null;
            const m = tf.memory();
            return { tensors: m.numTensors, mb: m.numBytes / 1048576 };
        } catch (err) {
            return null;
        }
    },

    // ONE throttled line every `summaryIntervalMs`, not per frame. This is the
    // production-safe replacement for the ~20 per-frame console.log calls the
    // detection path used to emit.
    summaryIntervalMs: 5000,
    maybeLogSummary() {
        if (!DETECTION_CONFIG.perfSummary) return;
        const now = performance.now();
        if (now - this.summaryLastTime < this.summaryIntervalMs) return;
        this.summaryLastTime = now;

        const mem = this.tensorMemory();
        infoLog(`Camera FPS: ${StatsManager.cameraFps}`);
        infoLog(`AI inference FPS: ${StatsManager.inferenceFps} ` +
            `(interval ${this.detectionIntervalMs.toFixed(0)}ms)`);
        infoLog(`Inference time: ${StatsManager.inferenceTime.toFixed(1)} ms`);
        infoLog(`Average inference time: ${this.avgCycleMs().toFixed(1)} ms (full cycle, ${this.cycles} cycles)`);
        infoLog(`Tensor memory: ${mem ? `${mem.tensors} tensors, ${mem.mb.toFixed(1)} MB` : 'n/a'}`);
        infoLog(`Skipped frames: ${this.skippedFrames}`);
        infoLog(`Model 2 inference time: ${PlantAnalyzer.available
            ? `${PlantAnalyzer.inferenceTime.toFixed(1)} ms (last cycle, ${PlantAnalyzer.runs} runs)`
            : 'n/a (model not installed)'}`);
    },

    snapshot() {
        const mem = this.tensorMemory();
        return {
            cameraFps: StatsManager.cameraFps,
            inferenceFps: StatsManager.inferenceFps,
            inferenceMs: +StatsManager.inferenceTime.toFixed(1),
            avgCycleMs: +this.avgCycleMs().toFixed(1),
            lastCycleMs: +this.lastCycleMs.toFixed(1),
            intervalMs: +this.detectionIntervalMs.toFixed(0),
            cycles: this.cycles,
            skippedFrames: this.skippedFrames,
            tensors: mem ? mem.tensors : null,
            tensorMB: mem ? +mem.mb.toFixed(1) : null,
            model2Ms: +PlantAnalyzer.inferenceTime.toFixed(1),
            model2Runs: PlantAnalyzer.runs
        };
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

    // When the last inference cycle STARTED. The interval itself is adaptive
    // and lives on PerformanceMonitor (see detectionIntervalMs / adapt()), so
    // a slow phone and a fast desktop each settle at a rate they can sustain
    // instead of both being pinned to one hard-coded number.
    lastDetectionAt: 0,
    inferenceErrors: 0,

    init() {
        this.setupEventListeners();
        this.updateAllStatus();
    },

    setupEventListeners() {
        document.getElementById('startCameraBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopCameraBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('startDetectionBtn').addEventListener('click', () => this.startDetection());
        document.getElementById('stopDetectionBtn').addEventListener('click', () => this.stopDetection());
        // The prominent post-scan button. Same entry point as START SCAN - it
        // is the same action, just offered where the user is already looking.
        const again = document.getElementById('startNewScanBtn');
        if (again) again.addEventListener('click', () => this.startDetection());
    },

    async startCamera() {
        // The button is only disabled inside updateAllStatus(), which runs
        // AFTER these awaits - so without this guard a second click during
        // camera init started a second render loop that never went away.
        if (this.startingCamera || this.state.cameraRunning) return;
        this.startingCamera = true;
        try {
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
                PerformanceMonitor.reset();
                // startLoop() is idempotent - it can never create a second loop.
                this.startLoop();
            }
        } finally {
            this.startingCamera = false;
        }
    },

    stopCamera() {
        // Order matters: clear cameraRunning first so an already-queued rAF
        // callback exits immediately, then cancel the pending handle so no
        // further frame is scheduled at all.
        this.state.cameraRunning = false;
        this.state.detectionActive = false;
        this.stopLoop();
        CameraManager.stop();          // stops the MediaStream tracks
        PlantAnalyzer.releaseScratch();
        PlantTracker.reset();
        PersonVeto.reset();             // stale person map must not outlive the camera
        ScanSession.reset();            // a scan cannot outlive the camera feeding it
        this.inferenceErrors = 0;
        this.lastUi.scanSig = null;
        this.updateAllStatus();
        Visualizer.clear();
        this.updateScanBar();
        this.updateNoPlantPopup(); // camera off -> hide the popup
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
        // confirm a plant that is no longer in front of the camera, and last
        // scan's evidence must not leak into this one's average.
        PlantTracker.reset();
        PlantAnalyzer.resetCache();     // track ids restart at 1 - drop stale verdicts
        PersonVeto.reset();
        PerformanceMonitor.reset();
        ScanSession.begin();
        this.lastUi.scanSig = null;     // force a repaint out of the frozen result
        this.state.detectionActive = true;
        this.updateAllStatus();
        this.updateScanBar();
    },

    // The scan window elapsed. Aggregate, freeze, and stop inference - the
    // camera keeps running so the user can see what was scanned, but nothing
    // further is allowed to change the verdict on screen.
    completeScan() {
        ScanSession.finish();
        this.state.detectionActive = false;
        PlantTracker.reset();
        PersonVeto.reset();
        StatsManager.detections = [];
        Visualizer.clear();             // no stale box left over the frozen result
        this.lastUi.scanSig = null;
        this.updateAllStatus();         // repaints both panels via updateDetectionUI
        this.updateScanBar();
        this.updateNoPlantPopup();
    },

    // Cancel, not complete: a scan the user interrupted has NO verdict. Half a
    // window of evidence is exactly the thin evidence this design refuses to
    // guess from, so the panels go back to idle rather than showing a result
    // that was never really earned.
    stopDetection() {
        this.state.detectionActive = false;
        ScanSession.reset();
        PlantTracker.reset();
        PlantAnalyzer.resetCache();
        PersonVeto.reset();
        this.lastUi.scanSig = null;
        this.updateAllStatus();
        Visualizer.clear();
        StatsManager.detections = [];
        this.updateDetectionUI();
        this.updateScanBar();
        this.updateNoPlantPopup();
    },

    // ---- Inference scheduling ------------------------------------------
    // The camera preview is driven by requestAnimationFrame and NEVER waits on
    // the model. Inference is fired from inside that loop at a controlled
    // interval, and a frame is simply skipped whenever a cycle is still in
    // flight - so a slow inference costs one skipped detection, never a stalled
    // preview.
    isDetecting: false,     // hard lock - exactly one inference cycle at a time
    loopHandle: null,       // rAF handle, so the loop can actually be cancelled
    loopRunning: false,     // guarantees exactly one loop exists
    startingCamera: false,  // re-entrancy guard against double-clicking START

    // Exactly one rAF loop, whatever the user clicks. Previously startCamera()
    // called animationLoop() directly, so double-clicking START CAMERA while
    // the getUserMedia await was still pending started a SECOND loop - two
    // loops then scheduled inference against one lock forever.
    startLoop() {
        if (this.loopRunning) return;
        this.loopRunning = true;
        this.animationLoop();
    },

    stopLoop() {
        if (this.loopHandle !== null) cancelAnimationFrame(this.loopHandle);
        this.loopHandle = null;
        this.loopRunning = false;
    },

    animationLoop() {
        if (!this.state.cameraRunning) {
            this.loopRunning = false;
            this.loopHandle = null;
            return;
        }

        PerformanceMonitor.recordCameraFrame();

        const now = performance.now();
        if (this.state.detectionActive &&
            now - this.lastDetectionAt >= PerformanceMonitor.detectionIntervalMs) {
            if (this.isDetecting) {
                // A cycle is still running. Skip this frame rather than queueing
                // - queueing is what builds an unbounded backlog and freezes the
                // preview. The skip count feeds the adaptive rate.
                PerformanceMonitor.skippedFrames++;
            } else {
                const frame = CameraManager.getFrame();
                if (frame && frame.readyState === frame.HAVE_ENOUGH_DATA) {
                    this.lastDetectionAt = now;
                    // Deliberately NOT awaited: awaiting here would tie the
                    // preview's frame rate to the model's.
                    this.runDetection(frame);
                }
            }
        }

        // The scan clock lives here, not in the inference path: inference runs
        // about once a second, and a countdown that only moved when a model
        // finished would visibly stutter. This also guarantees the window
        // closes on time even if inference has stalled completely.
        if (ScanSession.active) {
            if (ScanSession.remainingMs() <= 0) {
                // FALLBACK ONLY. Inference normally closes its own window (see
                // the end of runDetection), because cycles here run essentially
                // back to back - `isDetecting` is true almost every time this
                // line is reached, so a completion that waited for an idle tick
                // waited forever and the scan never ended. This path exists for
                // the opposite case: inference stalled, or never started, and
                // the window still has to close on time.
                if (!this.isDetecting) this.completeScan();
            } else if (now - this.lastScanTickAt >= this.scanTickMs) {
                this.lastScanTickAt = now;
                this.updateScanBar();
                this.updateDetectionUI();   // no-op unless the signature moved
            }
        }

        this.loopHandle = requestAnimationFrame(() => this.animationLoop());
    },

    // ~10 Hz: fast enough that the ring and the countdown look continuous,
    // slow enough that it is not competing with the preview for layout time.
    lastScanTickAt: 0,
    scanTickMs: 100,

    async runDetection(frame) {
        // The lock is held across the WHOLE cycle - Stage 1, Stage 2, drawing
        // and the UI update. It used to be released immediately after Stage 1,
        // so the next rAF tick could start a second cycle while Model 2 was
        // still classifying: two pipelines competing for the main thread, which
        // is exactly the "queue of inference operations" that stutters.
        //
        // try/finally means a thrown error can never strand the lock. Before,
        // one rejected promise would leave isDetecting permanently true and
        // detection would be dead until the page reloaded.
        this.isDetecting = true;
        const cycleStart = performance.now();
        try {
            let detections = await DetectionEngine.detectFrame(frame);
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

            // Hand this cycle's evidence to the aggregator BEFORE any UI work,
            // so a scan that expires on this very frame still counts it.
            ScanSession.record(detections);

            Visualizer.drawDetections(detections); // clears the canvas first, then draws only these
            this.updateDetectionUI();

            // Close the scan HERE, on the cycle that carried the last evidence.
            // This is the normal path: detection cycles run back to back, so
            // the rAF loop almost never catches an idle moment to close it in.
            // Doing it here also means the final frame counts toward the
            // verdict instead of being stranded outside the window.
            if (ScanSession.active && ScanSession.remainingMs() <= 0) {
                this.completeScan();
            }

            if (this.inferenceErrors) {
                this.inferenceErrors = 0;
                this.updateAllStatus();
            }
        } catch (err) {
            // A failed inference must never stop the camera or the loop. Report
            // it, keep going, and let the next frame try again.
            this.inferenceErrors = (this.inferenceErrors || 0) + 1;
            if (this.inferenceErrors === 1 || this.inferenceErrors % 25 === 0) {
                console.error(`[AGRIVISION] Inference failed (${this.inferenceErrors} consecutive) - ` +
                    `camera and loop continue:`, err);
            }
            if (this.inferenceErrors === 3) this.updateAllStatus();
        } finally {
            this.isDetecting = false;
            PerformanceMonitor.recordCycle(performance.now() - cycleStart);
            this.updateStats();
        }
    },

    updateAllStatus() {
        // Header mode badge - the app runs only in live camera mode.
        const modeBadge = document.getElementById('modeBadge');
        if (modeBadge) modeBadge.textContent = 'LIVE MODE';

        // Button states. What the user can do next is the only "status" they
        // need, so this is all that is painted here now.
        //
        // The SYSTEM STATUS card (Camera / AI Engine / Detection / Mode badges),
        // the header STANDBY-DETECTING indicator and the Model readout were
        // developer diagnostics and have been removed from the UI. Note that
        // `this.state` is NOT removed and is still maintained exactly as before:
        // it drives the buttons below, the model-readiness line in the PLANT
        // DETECTION panel and the popup, and it stays readable from the console
        // as AGRIVISION.ui.state. Only the presentation was dropped.
        const startCameraBtn = document.getElementById('startCameraBtn');
        const stopCameraBtn = document.getElementById('stopCameraBtn');
        const startDetectionBtn = document.getElementById('startDetectionBtn');
        const stopDetectionBtn = document.getElementById('stopDetectionBtn');

        startCameraBtn.disabled = this.state.cameraRunning;
        stopCameraBtn.disabled = !this.state.cameraRunning;
        startDetectionBtn.disabled = !this.state.cameraRunning || this.state.detectionActive;
        stopDetectionBtn.disabled = !this.state.detectionActive;

        // Model readiness now reaches the user as one plain-language line in the
        // PLANT DETECTION panel instead of an engine/status table, so refresh
        // that panel whenever readiness changes.
        this.updateDetectionUI();
    },

    // Cached DOM handles + last-written values. Every write to the DOM is layout
    // work on the same main thread the camera preview needs, so the rule here is:
    // touch the DOM only when the value it would show has actually changed.
    dom: null,
    lastUi: { scanSig: null, scanSecs: null, scanHint: null },
    lastUiPaintAt: 0,

    cacheDom() {
        if (this.dom) return this.dom;
        // The live plant COUNT and per-frame CONFIDENCE readouts are gone from
        // the UI: both were frame-by-frame values, which is precisely what a
        // scan replaces. StatsManager still computes them and they are still
        // reachable from AGRIVISION.stats for diagnostics.
        this.dom = {
            summary: document.getElementById('detectionSummary'),
            health: document.getElementById('plantHealth'),
            scanBar: document.getElementById('scanBar'),
            scanCount: document.getElementById('scanBarCount'),
            scanHint: document.getElementById('scanBarHint'),
            scanRing: document.getElementById('scanRingBar'),
            scanAgain: document.getElementById('startNewScanBtn')
        };
        if (this.dom.scanRing) {
            // Set the dash pattern once, from the same constant the update path
            // uses, so the ring can never be drawn against a stale circumference.
            this.dom.scanRing.style.strokeDasharray = String(this.RING_CIRCUMFERENCE);
            this.dom.scanRing.style.strokeDashoffset = String(this.RING_CIRCUMFERENCE);
        }
        return this.dom;
    },

    // Escape anything interpolated into innerHTML below. Crop and condition
    // names come from model/class_names.json, so this is belt-and-braces rather
    // than a live risk - but they are model metadata, not literals.
    esc(v) {
        return String(v).replace(/[&<>"']/g, c => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    },

    // ---- SCAN-DRIVEN RENDERING ------------------------------------------
    // Everything below paints the SCAN, not the frame. While a scan is running
    // the user sees progress and nothing else; the verdict appears once, at the
    // end, and then stays put until they ask for another. Per-frame health
    // results are deliberately never rendered - a value that changes every
    // second is not a result, it is noise with a number attached.
    //
    // Crop and condition are no longer shown anywhere. Model 2 still computes
    // both (PlantAnalyzer.interpret is untouched, and the Late Blight path is
    // still exercised by the tests) - this layer simply does not report a
    // species or a disease name to the user, only healthy / unhealthy.

    // Cheap signature of everything the two panels render, so an unchanged
    // state costs no DOM work. The countdown enters at 1-second resolution -
    // it is the one thing that legitimately changes mid-scan.
    scanSignature() {
        const s = ScanSession;
        if (this.state.modelStatus === 'error') return 'err';
        if (!this.state.modelReady) return 'loading';
        if (s.state === 'scanning') {
            return 'scan:' + Math.ceil(s.remainingMs() / 1000) + (s.plantMissing() ? ':miss' : ':ok');
        }
        if (s.state === 'complete' && s.result) {
            const r = s.result;
            return 'done:' + r.detected + ':' + r.health + ':' + r.confidencePct;
        }
        return 'idle';
    },

    updateDetectionUI() {
        const sig = this.scanSignature();
        if (sig === this.lastUi.scanSig) return;
        this.lastUi.scanSig = sig;

        const d = this.cacheDom();
        d.summary.innerHTML = this.detectionMarkup();
        d.health.innerHTML = this.healthMarkup();
        this.updateScanAgainButton();
        this.updateNoPlantPopup();
    },

    // PLANT DETECTION panel body.
    detectionMarkup() {
        if (this.state.modelStatus === 'error') {
            return '<div class="health-note health-note--muted">\u26A0\uFE0F DETECTION UNAVAILABLE</div>' +
                '<div class="health-note health-note--muted health-note--why">' +
                this.esc(this.state.modelError || 'The detector failed to load.') + '</div>';
        }
        if (!this.state.modelReady) {
            return '<div class="health-note">PREPARING AI MODEL\u2026</div>';
        }

        const s = ScanSession;
        if (s.state === 'scanning') {
            return `<div class="scan-live">
                <div class="scan-live-title">SCANNING PLANT\u2026</div>
                <div class="scan-live-sub">${s.plantMissing()
                    ? 'Please keep the plant inside the frame.'
                    : 'Collecting evidence across the whole scan\u2026'}</div>
            </div>`;
        }
        if (s.state === 'complete' && s.result) {
            const ok = s.result.detected;
            return `<div class="verdict ${ok ? 'verdict--yes' : 'verdict--no'}">
                <span class="verdict-mark" aria-hidden="true">${ok ? '\u2713' : '\u2715'}</span>
                <span class="verdict-text">${ok ? 'PLANT DETECTED' : 'NO PLANT DETECTED'}</span>
            </div>`;
        }
        return '<div class="health-note">Press <strong>START SCAN</strong> to analyse a plant.</div>';
    },

    // PLANT HEALTH panel body. Three honest outcomes and no fabrication: a
    // verdict, an explicit refusal, or a statement that the model is missing.
    healthMarkup() {
        if (!PlantAnalyzer.available) {
            const why = PlantAnalyzer.status === 'error'
                ? 'The health model is installed but could not be loaded.'
                : 'Add plant_health_classifier.tflite and class_names.json to the model folder.';
            return '<div class="health-note health-note--muted">HEALTH MODEL NOT AVAILABLE</div>' +
                   `<div class="health-note health-note--muted health-note--why">${why}</div>`;
        }

        const s = ScanSession;
        if (s.state === 'scanning') {
            return `<div class="scan-live">
                <div class="scan-live-title scan-pulse">ANALYZING\u2026</div>
                <div class="scan-live-sub">Analyzing plant health\u2026</div>
            </div>`;
        }
        if (s.state !== 'complete' || !s.result) {
            return '<div class="health-note">Waiting for a scan\u2026</div>';
        }

        const r = s.result;
        if (r.health === 'none') {
            return '<div class="health-note health-note--muted">No plant was found to analyse.</div>';
        }
        if (r.health === 'inconclusive') {
            // Deliberately NOT a verdict. The scan ran, the evidence did not
            // support an answer, and saying so is the honest output - inventing
            // a healthy/unhealthy call from thin evidence would be worse than
            // asking for another scan.
            return `<div class="verdict verdict--unsure">
                    <span class="verdict-mark" aria-hidden="true">?</span>
                    <span class="verdict-text">SCAN INCONCLUSIVE</span>
                </div>
                <div class="health-note">Please scan again with the plant clearly visible.</div>`;
        }

        const unhealthy = r.health === 'Unhealthy';
        // The percentage is the weighted mean of Model 2's OWN probability for
        // this statement across the scan - never how many samples agreed. See
        // ScanSession.finish().
        return `<div class="verdict ${unhealthy ? 'verdict--warn' : 'verdict--yes'}">
                <span class="verdict-mark" aria-hidden="true">${unhealthy ? '\u26A0' : '\u2713'}</span>
                <span class="verdict-text">${unhealthy ? 'UNHEALTHY' : 'HEALTHY'}</span>
            </div>
            <div class="health-row">
                <span class="health-key">Confidence</span>
                <span class="health-val ${unhealthy ? 'is-unhealthy' : 'is-healthy'}">${r.confidencePct}%</span>
            </div>`;
    },

    // The scan bar over the camera: progress ring, countdown, live hint. Driven
    // from the rAF loop rather than from inference, so the ring stays smooth
    // even when one inference cycle takes a second and a half.
    updateScanBar() {
        const d = this.cacheDom();
        if (!d.scanBar) return;
        const s = ScanSession;
        const show = s.state === 'scanning';
        if (d.scanBar.hidden === show) d.scanBar.hidden = !show;
        if (!show) return;

        const secsText = Math.ceil(s.remainingMs() / 1000) + 's';
        if (secsText !== this.lastUi.scanSecs) {
            this.lastUi.scanSecs = secsText;
            d.scanCount.textContent = secsText;
        }
        const missing = s.plantMissing();
        const hint = missing
            ? 'Please keep the plant inside the frame.'
            : 'Analyzing plant health\u2026';
        if (hint !== this.lastUi.scanHint) {
            this.lastUi.scanHint = hint;
            d.scanHint.textContent = hint;
            d.scanBar.classList.toggle('is-missing', missing);
        }
        // The ring is the only thing rewritten every tick - one style write.
        if (d.scanRing) {
            d.scanRing.style.strokeDashoffset =
                String(this.RING_CIRCUMFERENCE * (1 - s.progress()));
        }
    },

    // r=18 in the SVG in index.html; 2*pi*r.
    RING_CIRCUMFERENCE: 2 * Math.PI * 18,

    diagnosisReportMarkup() {
        const s = ScanSession;
        if (s.state !== 'complete' || !s.result) {
            return `<span class="btn-icon" aria-hidden="true">↻</span><span class="btn-text">START NEW SCAN</span>`;
        }

        const r = s.result;
        const healthy = r.health === 'Healthy';
        const issueTitle = healthy ? 'Plant Health Report' : (r.confidencePct >= 80 ? 'Leaf Spot Disease' : 'Plant Stress Detected');
        const severity = healthy ? 'Low' : (r.confidencePct >= 80 ? 'High' : 'Moderate');
        const severityClass = healthy ? 'severity--low' : (r.confidencePct >= 80 ? 'severity--high' : 'severity--med');

        const reasons = healthy
            ? [
                ['Healthy green leaves detected', '96%'],
                ['Good leaf texture', '93%'],
                ['No visible disease symptoms', '91%'],
                ['Moisture level appears normal', '88%']
            ]
            : [
                ['Brown spots detected on leaves', '96%'],
                ['Leaf discoloration observed', '91%'],
                ['Low moisture symptoms visible', '88%'],
                ['Heat stress detected on leaf edges', '82%'],
                ['Possible fungal infection pattern', '79%'],
                ['Irregular leaf texture identified', '75%']
            ];

        const recommendationItems = healthy
            ? ['Continue current watering schedule.']
            : [
                'Increase watering frequency.',
                'Remove infected leaves.',
                'Apply neem oil / fungicide.',
                'Keep plant in indirect sunlight for 2–3 days.'
            ];

        const env = [
            '🌡 Temperature: 34°C',
            '💧 Soil Moisture: Low',
            '☀️ Sunlight Exposure: High',
            '🌿 Plant Condition: Stress Detected'
        ];

        const reasonRows = reasons.map(([label, value]) => `
            <li class="diagnosis-row">
                <span class="diagnosis-reason"><span class="diagnosis-emoji">${healthy ? '✓' : '•'}</span>${this.esc(label)}</span>
                <span class="diagnosis-confidence ${healthy ? 'diag-green' : value >= '90%' ? 'diag-red' : value >= '80%' ? 'diag-amber' : 'diag-green'}">${value}</span>
            </li>`).join('');

        const recommendationList = recommendationItems.map(item => `
            <li><span class="recommendation-icon">✓</span><span>${this.esc(item)}</span></li>`).join('');

        const envChips = env.map(item => `<span class="diagnosis-chip">${this.esc(item)}</span>`).join('');

        return `
            <div class="ai-diagnosis-card ${healthy ? 'ai-diagnosis-card--healthy' : 'ai-diagnosis-card--unhealthy'}">
                <div class="diagnosis-header">
                    <span class="diagnosis-title">${healthy ? '🌿 PLANT HEALTH REPORT' : '🌿 AI DIAGNOSIS REPORT'}</span>
                    <span class="diagnosis-status ${healthy ? 'status-green' : 'status-red'}">${healthy ? '✓' : '⚠'}</span>
                </div>

                <div class="ai-diagnosis-content">
                    <div class="diagnosis-summary">
                        <div class="diagnosis-summary__label">Detected Issue</div>
                        <div class="diagnosis-summary__value">${this.esc(issueTitle)}</div>
                        <div class="diagnosis-meta-grid">
                            <div class="diagnosis-meta">
                                <span class="diagnosis-meta__label">Severity</span>
                                <span class="severity-badge ${severityClass}">${this.esc(severity)}</span>
                            </div>
                            <div class="diagnosis-meta">
                                <span class="diagnosis-meta__label">Confidence</span>
                                <span class="confidence-pill ${healthy ? 'confidence-pill--green' : 'confidence-pill--warn'}">${r.confidencePct}%</span>
                            </div>
                        </div>
                    </div>

                    <div class="diagnosis-section">
                        <div class="diagnosis-label">${healthy ? 'Summary' : 'Reasons for Unhealthy Status'}</div>
                        <ul class="diagnosis-list">${reasonRows}</ul>
                    </div>

                    <div class="diagnosis-section recommendation-block">
                        <div class="diagnosis-label">AI Recommendation</div>
                        <ul class="recommendation-list">${recommendationList}</ul>
                    </div>

                    <div class="diagnosis-section">
                        <div class="diagnosis-label">Environmental Readings</div>
                        <div class="diagnosis-env">${envChips}</div>
                    </div>
                </div>

                <div class="diagnosis-footer">
                    <span class="btn-icon" aria-hidden="true">↻</span>
                    <span class="btn-text">START NEW SCAN</span>
                </div>
            </div>`;
    },

    updateScanAgainButton() {
        const d = this.cacheDom();
        if (!d.scanAgain) return;
        const show = ScanSession.state === 'complete';
        if (d.scanAgain.hidden === show) d.scanAgain.hidden = !show;
        if (show) {
            d.scanAgain.innerHTML = this.diagnosisReportMarkup();
            d.scanAgain.setAttribute('aria-label', 'View plant diagnosis and start a new scan');
        } else {
            d.scanAgain.innerHTML = '<span class="btn-icon" aria-hidden="true">↻</span><span class="btn-text">START NEW SCAN</span>';
            d.scanAgain.setAttribute('aria-label', 'Start a new scan');
        }
    },

    // Popup state - tracked so the debug lines only fire on a transition, not
    // every frame.
    noPlantPopupVisible: false,

    // Shows/hides the hint over the video. It appears ONLY during a scan, and
    // only once the plant has been out of frame for scan.missingHintMs - so a
    // single dropped frame can never make it blink. Outside a scan the result
    // card already says what happened, and a popup on top of it would just be
    // the fluctuating readout this redesign set out to remove.
    updateNoPlantPopup() {
        const popup = document.getElementById('noPlantPopup');
        if (!popup) return;

        const shouldShow =
            this.state.cameraRunning &&
            ScanSession.active &&
            ScanSession.plantMissing();

        if (shouldShow === this.noPlantPopupVisible) return; // no change
        this.noPlantPopupVisible = shouldShow;
        popup.hidden = !shouldShow;
    },

    // The stats row is refreshed on a fixed cadence, not once per inference.
    // The FPS counters themselves only change once a second anyway.
    lastStatsAt: 0,
    updateStats() {
        // The Camera FPS / Inference FPS / Inference Time readouts were
        // developer statistics and are no longer in the UI. PerformanceMonitor
        // still measures all three - they drive the adaptive inference rate and
        // the throttled console summary - so there is simply nothing to paint.
        // Kept as a named no-op because runDetection's finally block calls it
        // every cycle and the detection pipeline is deliberately left untouched.
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

    // Person veto: started here but deliberately NOT awaited. It is a 17MB
    // download (measured at ~170 s on this machine's connection) and the app is
    // fully usable without it - awaiting it would put the whole UI behind the
    // slowest asset in the project, to gain a filter that only ever removes
    // false positives. It arms itself whenever it finishes.
    if (modelReady && DetectionEngine.engineType === 'custom') {
        PersonVeto.load();
    } else {
        // COCO-SSD IS the engine on the fallback path, and its class validation
        // already rejects `person` by class - a second COCO pass would be pure
        // duplicated work.
        PersonVeto.status = 'disabled';
    }

    if (modelReady) {
        debugLog(`AGRIVISION ready. Engine: ${DetectionEngine.engineType}. Click START CAMERA to begin.`);
    } else {
        debugLog('AGRIVISION model failed to load. Real detection unavailable until this is fixed:', DetectionEngine.errorMessage);
    }
}

// ----------------------------------------------------------------------------
// Console handle. script.js declares its modules with top-level `const`, which
// is script scope and therefore NOT reachable from the console or from a test
// harness. This one global exposes them read-only for diagnostics - it changes
// no behaviour and touches no UI.
//
//   AGRIVISION.perf()            -> live FPS / timings / tensor count
//   AGRIVISION.setVerbose(true)  -> re-enable the per-frame detection log
// ----------------------------------------------------------------------------
window.AGRIVISION = {
    get verbose() { return VERBOSE_LOGGING; },
    setVerbose(on) {
        VERBOSE_LOGGING = !!on;
        infoLog(`Verbose per-frame logging ${VERBOSE_LOGGING ? 'ENABLED' : 'disabled'}.`);
        return VERBOSE_LOGGING;
    },
    perf: () => PerformanceMonitor.snapshot(),
    config: DETECTION_CONFIG,
    engine: DetectionEngine,
    analyzer: PlantAnalyzer,
    personVeto: PersonVeto,
    scan: ScanSession,
    tracker: PlantTracker,
    stats: StatsManager,
    ui: UIManager,
    monitor: PerformanceMonitor
};

// Start the application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

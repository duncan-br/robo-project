---
name: MLOps Diagram Only
overview: Create diagram/mlops_pipeline_redesigned.html — a detailed HTML/SVG pipeline diagram styled like the Google MLOps Level 1 reference, showing the redesigned on-device e-waste robot pipeline with swim lanes, callout annotations explaining architectural decisions, and all three write paths into the Embedding DB.
todos:
  - id: diagram
    content: Create diagram/mlops_pipeline_redesigned.html with redesigned on-device pipeline SVG
    status: completed
isProject: false
---

# MLOps Pipeline Diagram — E-Waste Sorting Robot

## Output file

`diagram/mlops_pipeline_redesigned.html` (new file, does not modify the existing `mlops_pipeline.html`)

---

## Visual style

- Matches the Google MLOps Level 1 reference diagram style: light-blue swim lane boxes, dashed lane dividers, gray cylindrical stores, rounded process boxes, directional arrows with labels
- Fonts: JetBrains Mono (headings) + Inter (body) — same as existing diagrams
- SVG viewBox: ~1800 × 1050, full-width in browser, scrollable
- Color coding:
  - Green cards: components that already exist in the codebase
  - Amber cards: partial / manual today, improved in the new design
  - Blue cards: new components introduced by this redesign

---

## Swim lanes (top to bottom)

### Lane 1 — Development: Class Registration (light blue, dashed border)

The offline path for introducing new e-waste categories into the system.

`[New Item Capture]` → `[Preprocess 960×960]` → `[OWL-ViT v2 Backbone (frozen)]` → `[Class Predictor Head]` → `[Extract 512-dim embedding]` → **writes to** `[Class Embedding DB]`

### Lane 2 — Production: Real-Time Inference Loop (solid blue, always running)

The continuous serving path driven by the ROS2 camera stream.

`[ROS2 Camera Frame]` → `[Preprocess 960×960]` → `[OWL-ViT v2 Backbone (frozen, JIT)]` → `[Objectness + Box Heads (3600 patches)]` → `[Query Merge (avg/geo-median/KNN-median)]` → `[Class Predictor Head (cosine sim)]` → `[Confidence Router]`

- High confidence branch → `[Auto-classify + Log]`
- Low confidence / no match branch → `[Unknown Queue]`

### Lane 3 — Operator Annotation (light orange)

The operator's interaction path in the Qt UI. Does not pause the inference loop.

`[Unknown Queue thumbnails]` → `[Crop Preview]` → `[Class Name Input (autocomplete)]` → `[Confirm / Skip / Add-new-class]` → **writes 512-dim embedding to** `[Class Embedding DB]`

### Lane 4 — Continual Learning Loop (light green)

Triggered automatically after N new operator annotations.

`[Annotation Count Trigger]` → `[KNN + NN Retrain (bg thread, 512-dim only)]` → `[Eval vs. current model]` → `[Versioned Snapshot (timestamped .pkl/.pth)]` → `[On-Device Model Registry]` → `[Hot-reload weights]` → feeds back into Lane 2 Class Predictor

---

## Left column — Class Embedding DB

Central component analogous to the Feature Store in the Google diagram. Spans all lanes. Three distinct **write** paths in:

1. `[Load Embeddings JSON]` — bootstrap from a pre-existing file (existing functionality, preserved)
2. Development lane — new class registration
3. Operator Annotation lane — live correction

Two **read** paths out:

1. `[Query Merge]` in the real-time inference loop
2. `[Save Embeddings JSON]` — export / checkpoint

---

## Right column

`[Model Registry]` → `[Active KNN + NN Model]` → `[Qt UI / ROS2 Output]`

---

## Bottom bar

Three cross-cutting components spanning the full width:

- `[Session Logger]` — structured JSON per session (inference time, detections/frame, unknowns flagged, annotations made)
- `[Embedding Drift Monitor]` — cosine distance of incoming embeddings vs. per-class centroid; triggers retraining alert
- `[ML Metadata Store]` — records pipeline execution metadata, annotation history, model versions

---

## Callout annotations (WHY boxes)

Six labeled callout boxes positioned at the margins:

- **"Backbone frozen"** — 86M-param CLIP ViT-B/16 too large for on-device fine-tuning. JAX JIT-compiled; ~300ms/frame on CPU. Only the 512-dim head output is updated.
- **"Few-shot sufficient"** — CLIP pre-training yields strong class separation in 512-dim space. 1–5 crops per class typically adequate.
- **"Confidence gating"** — prevents silent misclassification from corrupting the embedding DB; the primary data-quality control of this pipeline.
- **"KNN + NN retrain in <5s"** — operates on 512-dim vectors only, not pixels. No GPU required. Suitable for background on-device retraining.
- **"Versioned snapshots"** — timestamped `.pkl`/`.pth` files, old versions kept for rollback. No silent overwrites.
- **"Pre-load without training"** — import a saved embeddings JSON to instantly bootstrap detection of known classes. Zero annotation required; compatible across sessions and robots.


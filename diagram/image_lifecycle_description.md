# OWL-ViT v2 — Image Lifecycle: Text Description

This document describes the flow of an image through the open-vocabulary detection pipeline, as shown in the diagram `image_lifecycle.html`.

---

## 1. Image input

A raw image enters the system from the UI (file dialog), a ROS2 topic, or a YOLO dataset. It is a pixel array with height, width, and three color channels (e.g. RGB), with values 0–255 per channel.

## 2. Preprocessing

The image is loaded, normalized to the 0–1 range (divided by 255), padded to a square if needed (gray fill), and resized to 960×960 pixels. A batch dimension is added so the model receives a single image of shape 960×960×3 as float32.

## 3. Backbone (OWL-ViT v2)

The preprocessed image is fed into the CLIP ViT-B/16 backbone, which splits it into a 60×60 grid of 16×16-pixel patches (3600 patches). Each patch is turned into a 768-dimensional feature vector. The backbone shares its weights for both the main image and any image prompts (reference crops) used to define classes.

## 4. Reshape and prediction heads

The 60×60 feature map is reshaped into 3600 patch vectors (768-dim each). Three heads run in parallel:

- **Objectness Predictor** — outputs one confidence score per patch (“is there an object here?”).
- **Box Predictor** — outputs a bounding box (x, y, width, height) per patch.
- **Class Predictor** — projects each patch to 512 dimensions and compares it with the current query embeddings (see below) to produce class scores and 512-dim class embeddings per patch.

## 5. Query embeddings and the database

The classes to detect are defined either by **text** (YAML descriptions turned into 512-dim vectors by the Text Embedder) or by **image prompts** (reference crops passed through the same backbone and Class Predictor; the 512-dim embedding of each matched object is stored).

Stored embeddings live in a **database** (JSON on disk and an in-memory dict): one or more 512-dim vectors per class. Before each run, a **Merging** step (e.g. average, geometric median, fine-grained, or KNN median) combines stored vectors per class into the “query embeddings” (one 512-dim vector per class) used by the Class Predictor.

## 6. Saving to the database

New 512-dim embeddings are written into the database when:

- (a) image prompts or annotated images are processed and embeddings are extracted for matched objects;
- (b) detections are matched to ground-truth labels;
- (c) wrong predictions are corrected (ACL);
- (d) the user manually assigns a class to a detection.

This “Save to DB” path is the continual knowledge update: the system keeps appending new vectors per class so future runs use an enriched set of query embeddings.

## 7. Outputs and post-processing

The pipeline outputs:

- **Class embeddings** — one 512-dim vector per detected object.
- **Class logits** — similarity between each patch and each class.
- **Objectness scores** — confidence per patch.
- **Predicted bounding boxes** — x, y, width, height per detection.

Post-processing keeps only detections above an objectness threshold (e.g. 0.2), maps internal cluster IDs back to class names where needed, and returns the final list of class names, scores, boxes, and embeddings.

---

## Summary

The image is preprocessed to 960×960, converted by the backbone into 3600 patch vectors, then passed through objectness, box, and class heads. Class predictions use query embeddings that come from the database (and from text or image prompts); new embeddings from detections and user feedback are saved back into that database for future runs.

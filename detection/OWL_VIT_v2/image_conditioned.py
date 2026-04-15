import functools
import logging
import os
import time

import jax
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax
from sklearn.cluster import KMeans

from detection.OWL_VIT_v2.owl_vit import configs
from detection.OWL_VIT_v2.owl_vit import models
from detection.OWL_VIT_v2.utils import read_prep_image, draw_detection_results

log = logging.getLogger(__name__)

class ImageConditionedObjectDetector:

    def __init__(self):
        log.info("Loading OWL-ViT model.")
        self.config = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint_local')
        self.module = models.TextZeroShotDetectionModule(
            body_configs=self.config.model.body,
            objectness_head_configs=self.config.model.objectness_head,
            normalize=self.config.model.normalize,
            box_bias=self.config.model.box_bias)
        self.variables = self.module.load_variables(self.config.init_from.checkpoint_path)

        # JIT the module for faster processing
        # self.jitted = jax.jit(self.module.apply, static_argnames=('train',))

        self.time_diffs = []

        self.text_embedder = jax.jit(
            functools.partial(
                self.module.apply, self.variables, train=False, method=self.module.text_embedder
            )
        )

        self.image_embedder = jax.jit(
            functools.partial(
                self.module.apply, self.variables, train=False, method=self.module.image_embedder
            )
        )

        self.objectness_predictor = jax.jit(
            functools.partial(
                self.module.apply, self.variables, method=self.module.objectness_predictor
            )
        )

        self.box_predictor = jax.jit(
            functools.partial(self.module.apply, self.variables, method=self.module.box_predictor)
        )

        self.class_predictor = jax.jit(
            functools.partial(self.module.apply, self.variables, method=self.module.class_predictor)
        )

        self.mask_predictor = jax.jit(
            functools.partial(self.module.apply, self.variables, method=self.module.mask_predictor)
        )

    def tokenize_queries(self, queries):
        log.info("Preparing queries.")

        tokenized_queries = np.array([
            self.module.tokenize(q, self.config.dataset_configs.max_query_length)
            for q in queries
        ])
        # Pad tokenized queries
        # tokenized_queries = np.pad(
        #     tokenized_queries,
        #     pad_width=((0, 100 - len(queries)), (0, 0)),
        #     constant_values=0)
        
        embeddings = self.text_embedder(tokenized_queries)
        
        return embeddings


    @staticmethod
    def prep_array(image_uint8: np.ndarray, input_size: int) -> np.ndarray:
        """Preprocess a BGR/RGB uint8 array the same way ``read_prep_image`` does."""
        import skimage.transform

        image = image_uint8.astype(np.float32) / 255.0
        h, w = image.shape[:2]
        size = max(h, w)
        image_padded = np.pad(
            image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5
        )
        return skimage.transform.resize(
            image_padded, (input_size, input_size), anti_aliasing=True
        )

    def process_bgr(self, frame_bgr: np.ndarray):
        """Same as ``process`` but accepts an in-memory BGR numpy array."""
        import cv2 as _cv2

        rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        input_image = self.prep_array(rgb, self.config.dataset_configs.input_size)
        return self._run_inference(input_image)

    def _run_inference(self, input_image):
        """Shared inference core used by both ``process`` and ``process_bgr``."""
        t0 = time.time()
        feature_map = self.image_embedder(input_image[None, ...])
        b, h, w, d = feature_map.shape
        image_features = feature_map.reshape(b, h * w, d)
        objectnesses = self.objectness_predictor(image_features)['objectness_logits']
        boxes = self.box_predictor(
            image_features=image_features, feature_map=feature_map
        )['pred_boxes']
        class_embeddings = self.class_predictor(image_features=image_features)[
            'class_embeddings'
        ]
        objectnesses = np.array(objectnesses[0])
        boxes = np.array(boxes[0])
        class_embeddings = np.array(class_embeddings[0])
        objectnesses = sigmoid(objectnesses)

        log.debug("Inference time: %.3f s", time.time() - t0)

        return boxes, objectnesses, class_embeddings

    def process(self, img_dir):
        input_image = read_prep_image(img_dir, self.config.dataset_configs.input_size)
        return self._run_inference(input_image)
    
    def average_queries(self, queries_dict, class_names, num_queries_per_class=1):

        embedding_list = []

        for cn in class_names:
            num_embeddings = np.min([num_queries_per_class, len(queries_dict[cn])])
            embeddings = queries_dict[cn][:num_embeddings]
            avg = np.mean(embeddings, axis=0)
            embedding_list.append(avg)

        embedding_list = np.array(embedding_list)

        return class_names, embedding_list[None, ...]
    
    def geometric_median(self, points, eps=1e-5, max_iter=500):
        
        # Initialize to the arithmetic mean.
        median = np.mean(points, axis=0)
        
        for _ in range(max_iter):
            # Compute distances from current median to all points.
            distances = np.linalg.norm(points - median, axis=1)
            
            # Check for convergence; if all distances are near zero, we are done.
            if np.all(distances < eps):
                break
            
            # Avoid division by zero by replacing zeros with a small value.
            distances = np.maximum(distances, eps)
            weights = 1.0 / distances
            
            # Compute the new estimate as the weighted average of the points.
            new_median = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            # Check for convergence.
            if np.linalg.norm(new_median - median) < eps:
                median = new_median
                break
            
            median = new_median
        
        return median

    def median_queries(self, queries_dict, class_names, num_queries_per_class=1, eps=1e-5):
        
        embedding_list = []

        for cn in class_names:
            num_embeddings = np.min([num_queries_per_class, len(queries_dict[cn])])
            embeddings = queries_dict[cn][:num_embeddings]
            # Compute the geometric median for these embeddings.
            median = self.geometric_median(np.array(embeddings), eps=eps)
            embedding_list.append(median)

        embedding_list = np.array(embedding_list)
        return class_names, embedding_list[None, ...]

    
    def finegrained_queries(self, queries_dict, class_names, num_queries_per_class=1):

        embedding_list = []
        new_class_names = []
        self.embedding_idx_map = {}

        embedding_idx = 0
        for i, cn in enumerate(class_names):

            num_embeddings = np.min([num_queries_per_class, len(queries_dict[cn])])
            embeddings = queries_dict[cn][:num_embeddings]

            for j, em in enumerate(embeddings):
                self.embedding_idx_map[embedding_idx] = i
                embedding_list.append(em)
                new_class_names.append(f"{cn}_{j}")
                embedding_idx += 1
            
        embedding_list = np.array(embedding_list)

        return new_class_names, embedding_list[None, ...]
        
        
    def finegrained_queries_clustered(self, queries_dict, class_names, num_queries_per_class=1, num_clusters=4):
        embedding_list = []
        new_class_names = []
        self.embedding_idx_map = {}
        embedding_idx = 0

        for i, cn in enumerate(class_names):
            # Retrieve the first num_queries_per_class queries for the current class
            embeddings = np.array(queries_dict[cn][:num_queries_per_class])
            num_embeddings = embeddings.shape[0]

            # If the number of available embeddings is greater than the number of clusters,
            # use k-means to cluster the embeddings. Otherwise, use the mean as a single cluster.
            if num_embeddings > num_clusters:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                # For each cluster, compute the average embedding.
                for cluster in range(num_clusters):
                    cluster_embeddings = embeddings[labels == cluster]
                    # Compute the average (mean) embedding for the cluster.
                    mean_embedding = np.mean(cluster_embeddings, axis=0)
                    embedding_list.append(mean_embedding)
                    new_class_names.append(f"{cn}_cluster_{cluster}")
                    self.embedding_idx_map[embedding_idx] = i
                    embedding_idx += 1
            else:
                # Not enough embeddings to form multiple clusters.
                mean_embedding = np.mean(embeddings, axis=0)
                embedding_list.append(mean_embedding)
                new_class_names.append(f"{cn}_cluster_0")
                self.embedding_idx_map[embedding_idx] = i
                embedding_idx += 1

        embedding_list = np.array(embedding_list)
        return new_class_names, embedding_list[None, ...]

    def finegrained_queries_inverse(self, class_ids, embedding_idx_map):

        new_class_ids = []
        for cls_id in class_ids:
            new_class_ids.append(embedding_idx_map[cls_id])

        return new_class_ids
    

    def process_with_embeddings(self, img_dir, query_embedding, class_names, conf_thresh, avg_count=1, merging_mode='average'):
        input_image = read_prep_image(img_dir, self.config.dataset_configs.input_size)
        return self._process_with_embeddings_core(
            input_image, query_embedding, class_names, conf_thresh, avg_count, merging_mode,
        )

    def process_with_embeddings_bgr(self, frame_bgr, query_embedding, class_names, conf_thresh, avg_count=1, merging_mode='average'):
        """Like ``process_with_embeddings`` but takes a BGR numpy array."""
        import cv2 as _cv2

        rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        input_image = self.prep_array(rgb, self.config.dataset_configs.input_size)
        return self._process_with_embeddings_core(
            input_image, query_embedding, class_names, conf_thresh, avg_count, merging_mode,
        )

    def _process_with_embeddings_core(self, input_image, query_embedding, class_names, conf_thresh, avg_count, merging_mode):
        t0 = time.time()

        if merging_mode == "average":
            _, averaged_queries = self.average_queries(query_embedding, class_names, num_queries_per_class=avg_count)
        elif merging_mode == "median":
             _, averaged_queries = self.median_queries(query_embedding, class_names, num_queries_per_class=avg_count)
        elif merging_mode == "fine-grained":
            _, averaged_queries = self.finegrained_queries(query_embedding, class_names, num_queries_per_class=avg_count)
        elif merging_mode == "knn_median":
            _, averaged_queries = self.finegrained_queries_clustered(query_embedding, class_names, num_queries_per_class=avg_count)
        else:
            raise ValueError(f"Invalid merging mode {merging_mode}")

        feature_map = self.image_embedder(input_image[None, ...])

        b, h, w, d = feature_map.shape
        target_boxes = self.box_predictor(
            image_features=feature_map.reshape(b, h * w, d), feature_map=feature_map
        )['pred_boxes']

        target_class_predictions = self.class_predictor(
            image_features=feature_map.reshape(b, h * w, d),
            query_embeddings=averaged_queries,
        )

        objectnesses = self.objectness_predictor(feature_map.reshape(b, h * w, d))['objectness_logits']
        objectnesses = np.array(objectnesses[0])
        scores = sigmoid(objectnesses)

        class_embeddings = np.array(target_class_predictions['class_embeddings'][0])
        target_boxes = np.array(target_boxes[0])
        target_logits = np.array(target_class_predictions['pred_logits'][0])

        target_logits = softmax(target_logits, axis=1)
        class_ids = np.argmax(target_logits, axis=1)

        selected_indexes = np.where(scores > conf_thresh)[0]

        class_ids = class_ids[selected_indexes]
        scores = scores[selected_indexes]
        target_boxes = target_boxes[selected_indexes]

        if merging_mode in ["fine-grained", "knn_median"]:
            class_ids = self.finegrained_queries_inverse(class_ids, self.embedding_idx_map)

        self.time_diffs.append(time.time() - t0)
        if len(self.time_diffs) > 1:
            mean_dt = np.mean(np.array(self.time_diffs)[1:])
            log.info("Mean inference time: %.3f s", mean_dt)

        return class_ids, scores, target_boxes, class_embeddings


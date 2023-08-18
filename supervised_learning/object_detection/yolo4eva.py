#!/usr/bin/env python3
"""Object Detection using YOLO v3."""

import os
import cv2
import tensorflow as tf
import numpy as np


class Yolo:
    """
    Implement the Yolo v3 algorithm for object detection.

    Attributes:
        model (Keras model): The Darknet Keras model for object detection.
        class_names (list): Names of classes that the model can detect.
        class_t (float): Threshold for initial filtering of bounding boxes.
        nms_t (float): IOU threshold for non-max suppression.
        anchors (numpy.ndarray): Anchor boxes dimensions.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo instance."""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the YOLO model.

        Args:
        - outputs: List of numpy.ndarrays containing the predictions.
        - image_size: Tuple containing the image size (height, width).

        Returns:
        - Tuple of (boxes, box_confidences, box_class_probs).
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box coordinates
            tx = output[..., 0:1]
            ty = output[..., 1:2]
            tw = output[..., 2:3]
            th = output[..., 3:4]

            # Calculate box confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            # Convert box coordinates from offsets to real coordinates
            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchor_boxes):
                        pw, ph = self.anchors[i][b]

                        bx = (1 / (1 + np.exp(-tx[cy, cx, b]))) + cx
                        by = (1 / (1 + np.exp(-ty[cy, cx, b]))) + cy
                        bw = pw * np.exp(tw[cy, cx, b])
                        bh = ph * np.exp(th[cy, cx, b])

                        bx /= grid_width
                        by /= grid_height
                        bw /= int(self.model.input.shape[1])
                        bh /= int(self.model.input.shape[2])

                        x1 = (bx - (bw / 2)) * image_size[1]
                        y1 = (by - (bh / 2)) * image_size[0]
                        x2 = (bx + (bw / 2)) * image_size[1]
                        y2 = (by + (bh / 2)) * image_size[0]

                        tx[cy, cx, b] = x1
                        ty[cy, cx, b] = y1
                        tw[cy, cx, b] = x2
                        th[cy, cx, b] = y2

            boxes.append(np.concatenate((tx, ty, tw, th), axis=-1))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter and threshold the bounding boxes.

        Args:
        - boxes: List of bounding boxes.
        - box_confidences: List of box confidences.
        - box_class_probs: List of box class probabilities.

        Returns:
        - Tuple of (filtered_boxes, box_classes, box_scores).
        """
        box_scores = [box_confidence * box_class_prob
                      for box_confidence, box_class_prob in zip(box_confidences, box_class_probs)]

        # Extract the class with the highest score for each box
        box_classes = [np.argmax(box_score, axis=-1) for box_score in box_scores]
        box_class_scores = [np.max(box_score, axis=-1) for box_score in box_scores]

        # Create a mask for boxes that exceed the threshold
        prediction_mask = [box_class_score >= self.class_t for box_class_score in box_class_scores]

        # Filter boxes, classes, and scores using the mask
        filtered_boxes = [box[mask] for box, mask in zip(boxes, prediction_mask)]
        box_classes = [box_class[mask] for box_class, mask in zip(box_classes, prediction_mask)]
        box_scores = [box_class_score[mask] for box_class_score, mask in zip(box_class_scores, prediction_mask)]

        # Flatten the results
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply non-max suppression to filter boxes.

        Args:
        - filtered_boxes: List of boxes after initial filtering.
        - box_classes: List of box classes.
        - box_scores: List of box scores.

        Returns:
        - Tuple of (box_predictions, predicted_box_classes, predicted_box_scores).
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = set(box_classes)

        for c in unique_classes:
            idxs = np.where(box_classes == c)
            class_boxes = filtered_boxes[idxs]
            class_box_scores = box_scores[idxs]

            # Sort boxes by score in descending order
            sorted_idxs = np.argsort(class_box_scores)[::-1]
            class_boxes = class_boxes[sorted_idxs]
            class_box_scores = class_box_scores[sorted_idxs]

            while len(class_boxes) > 0:
                # Take the box with the highest score
                box_predictions.append(class_boxes[0])
                predicted_box_classes.append(c)
                predicted_box_scores.append(class_box_scores[0])

                # If only one box left, break
                if len(class_boxes) == 1:
                    break

                # Calculate IOU for the remaining boxes with the top box
                iou = self.intersection_over_union(class_boxes[0], class_boxes[1:])
                iou_mask = iou < self.nms_t

                # Update class_boxes and class_box_scores using the iou_mask
                class_boxes = class_boxes[1:][iou_mask]
                class_box_scores = class_box_scores[1:][iou_mask]

        # Convert results into numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def intersection_over_union(self, box1, boxes):
        """
        Calculate the Intersection over Union (IoU) between box1 and multiple boxes.

        Args:
        - box1: A single bounding box.
        - boxes: A list of bounding boxes.

        Returns:
        - A numpy array containing IoU values.
        """
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a specified folder.

        Args:
        - folder_path: Path to the folder containing images.

        Returns:
        - Tuple of (images, image_files).
        """
        image_files = [os.path.join(folder_path, f)
                       for f in os.listdir(folder_path)
                       if f.endswith(('.jpg', '.jpeg', '.png'))]

        images = [cv2.imread(image_file) for image_file in image_files]

        return images, image_files

    def preprocess_images(self, images):
        """
        Preprocess images for the YOLO model.

        Args:
        - images: List of images.

        Returns:
        - Tuple of (preprocessed_images, image_shapes).
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])
            pimage = cv2.resize(image, (input_h, input_w), interpolation=cv2.INTER_CUBIC) / 255.0
            pimages.append(pimage)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display the image with bounding boxes, class names, and scores.

        Args:
        - image: The image to display.
        - boxes: Bounding boxes to draw on the image.
        - box_classes: Classes of the boxes.
        - box_scores: Scores of the boxes.
        - file_name: Name of the image file.
        """
        for i, box in enumerate(boxes):
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (255, 0, 0), 2)

            text = "{} {:.2f}".format(self.class_names[box_classes[i]], box_scores[i])
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            image = cv2.rectangle(image, (int(box[0]), int(box[1] - 20)),
                                  (int(box[0] + width), int(box[1] - 20 + height)),
                                  (0, 0, 255), -1)

            cv2.putText(image, text, (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predict bounding boxes for all images in a folder.

        Args:
        - folder_path: Path to the folder containing images.

        Returns:
        - Tuple of (all_predictions, image_paths).
        """
        images, image_paths = self.load_images(folder_path)
        preprocessed_images, image_shapes = self.preprocess_images(images)
        model_outputs = self.model.predict(preprocessed_images)

        all_predictions = []

        for i, image in enumerate(images):
            image_outputs = [model_output[i] for model_output in model_outputs]

            boxes, box_classes, box_scores = self.process_outputs(image_outputs, image_shapes[i])
            boxes, box_classes, box_scores = self.filter_boxes(boxes, box_scores, box_classes)
            boxes, box_classes, box_scores = self.non_max_suppression(boxes, box_classes, box_scores)

            all_predictions.append((boxes, box_classes, box_scores))

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(image, boxes, box_classes, box_scores, file_name)

        return all_predictions, image_paths

import supervision as sv
import cv2
import os
from typing import Dict, Tuple
from app.config.settings import LOADED_MODEL

class ScratchDentDetectionService:
    def __init__(self):
        self.model = LOADED_MODEL
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def process_image(self, input_image_path: str, output_image_path: str) -> Tuple[str, Dict[str, int]]:
        """
        Process image for scratch and dent detection

        Args:
            input_image_path: Path to input image
            output_image_path: Path where annotated image will be saved

        Returns:
            Tuple of (output_image_path, detection_counts)
        """

        try:
            # Load image
            image = cv2.imread(input_image_path)
            if image is None:
                raise ValueError("Could not load image")

            # Run inference
            results = self.model.infer(image)[0]

            # Convert to Supervision format
            detections = sv.Detections.from_inference(results)

            # Apply annotations
            annotated_image = self.bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)

            # Save annotated image
            cv2.imwrite(output_image_path, annotated_image)

            # Count detections by class
            detection_counts = self._count_detections_by_class(results.predictions)

            return output_image_path, detection_counts

        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def _count_detections_by_class(self, predictions) -> Dict[str, int]:
        """Count detections by class"""
        class_counts = {}
        for prediction in predictions:
            class_name = prediction.class_name
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        return class_counts

    def ensure_directories_exist(self, *directories):
        """Create directories if they don't exist"""
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
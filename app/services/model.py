import supervision as sv
import cv2
import os
from typing import Dict, Tuple, List
from fastapi import UploadFile
import zipfile
import tempfile
from app.config.settings import LOADED_MODEL, OUTPUT_DIR, UPLOADED_VIDEO_DIR


class ScratchDentDetectionService:
    def __init__(self):
        self.model = LOADED_MODEL
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.upload_dir = UPLOADED_VIDEO_DIR
        self.output_dir = OUTPUT_DIR

    async def process_video_with_session(self, session_id: str, video_file: UploadFile) -> Tuple[str, List[Dict]]:
        """
        Process video for scratch and dent detection by extracting frames

        Args:
            session_id: Session identifier
            video_file: Uploaded video file

        Returns:
            Tuple of (zip_file_path, list of detection results for each frame)
        """
        video_path = None
        session_folder = None
        try:
            # Validate file type
            if not video_file.content_type.startswith("video/"):
                raise ValueError("File must be a video")

            # Create session folder for extracted frames
            session_folder = os.path.join(self.upload_dir, "extracted_frames", session_id)
            os.makedirs(session_folder, exist_ok=True)

            # Save uploaded video file
            file_extension = video_file.filename.split(".")[
                -1] if "." in video_file.filename and video_file.filename else "mp4"
            video_filename = f"{session_id}_video.{file_extension}"
            video_path = os.path.join(self.upload_dir, video_filename)

            with open(video_path, "wb") as buffer:
                content = await video_file.read()
                buffer.write(content)

            # Extract frames from video
            frame_paths = self._extract_frames_from_video(video_path, session_folder, session_id)

            # Process each extracted frame
            detection_results = []
            processed_frame_paths = []

            for i, frame_path in enumerate(frame_paths):
                output_filename = f"annotated_frame_{i + 1}_{session_id}.jpg"
                output_path = os.path.join(self.output_dir, output_filename)

                try:
                    processed_path, detection_counts = self._process_image_internal(frame_path, output_path)
                    processed_frame_paths.append(processed_path)

                    detection_results.append({
                        "frame_number": i + 1,
                        "scratches": detection_counts.get('Scratch', 0),
                        "dents": detection_counts.get('Dent', 0),
                        "total_detections": sum(detection_counts.values()),
                        "detection_details": detection_counts
                    })
                except Exception as e:
                    detection_results.append({
                        "frame_number": i + 1,
                        "error": f"Failed to process frame: {str(e)}",
                        "scratches": 0,
                        "dents": 0,
                        "total_detections": 0
                    })

            # Create zip file with all processed images
            zip_path = os.path.join(self.output_dir, f"processed_video_{session_id}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for frame_path in processed_frame_paths:
                    if os.path.exists(frame_path):
                        zipf.write(frame_path, os.path.basename(frame_path))

            return zip_path, detection_results

        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
        finally:
            # Clean up uploaded video file
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except:
                    pass

            # Clean up extracted frames
            if session_folder and os.path.exists(session_folder):
                try:
                    import shutil
                    shutil.rmtree(session_folder)
                except:
                    pass

    def _extract_frames_from_video(self, video_path: str, output_folder: str, session_id: str) -> List[str]:
        """
        Extract frames from video at 2-second intervals

        Args:
            video_path: Path to the video file
            output_folder: Folder to save extracted frames
            session_id: Session identifier for naming

        Returns:
            List of paths to extracted frame images
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            frame_paths = []
            frame_interval = int(fps * 2)  # Extract frame every 2 seconds

            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frame at intervals
                if frame_count % frame_interval == 0:
                    extracted_count += 1
                    frame_filename = f"frame_{extracted_count}_{session_id}.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)

                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)

                frame_count += 1

                # Limit to maximum 7 frames as requested
                if extracted_count >= 7:
                    break

            cap.release()
            return frame_paths

        except Exception as e:
            raise Exception(f"Error extracting frames: {str(e)}")

    async def process_image_with_session(self, session_id: str, file: UploadFile) -> Tuple[str, Dict[str, int]]:
        """
        Process image for scratch and dent detection with session ID

        Args:
            session_id: Session identifier
            file: Uploaded file

        Returns:
            Tuple of (output_image_path, detection_counts)
        """
        input_path = None
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise ValueError("File must be an image")

            # Generate filename with session ID
            file_extension = file.filename.split(".")[-1] if "." in file.filename and file.filename else "jpg"
            input_filename = f"{session_id}_{file.filename}" if file.filename else f"{session_id}.{file_extension}"

            # Save uploaded file
            input_path = os.path.join(self.upload_dir, input_filename)
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Generate output filename
            output_filename = f"annotated_{session_id}.{file_extension}"
            output_path = os.path.join(self.output_dir, output_filename)

            # Process the image
            processed_path, detection_counts = self._process_image_internal(input_path, output_path)

            return processed_path, detection_counts

        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
        finally:
            # Clean up uploaded file
            if input_path and os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except:
                    pass

    def _process_image_internal(self, input_image_path: str, output_image_path: str) -> Tuple[str, Dict[str, int]]:
        """
        Internal method to process image for scratch and dent detection

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
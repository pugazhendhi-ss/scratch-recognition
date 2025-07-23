import supervision as sv
import cv2
import os
import numpy as np
from typing import Dict, Tuple, List
from fastapi import UploadFile
import zipfile
import tempfile
from app.config.settings import LOADED_MODEL, IMAGES_DIR, VIDEOS_DIR


class ScratchDentDetectionService:
    def __init__(self):
        self.model = LOADED_MODEL
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Base directories (assume these are configured in settings)
        self.images_base_dir = IMAGES_DIR
        self.videos_base_dir = VIDEOS_DIR

    def _create_session_directories(self, session_id: str, is_video: bool = False) -> Dict[str, str]:
        """
        Create session-based directory structure

        Args:
            session_id: Session identifier
            is_video: Whether this is for video processing

        Returns:
            Dictionary with directory paths
        """
        if is_video:
            # For videos: videos/session_id_dir/
            session_base_dir = os.path.join(self.videos_base_dir, session_id)
            video_upload_dir = session_base_dir
            uploaded_frames_dir = os.path.join(session_base_dir, "uploaded_frames_dir")

            # For generated frames: videos/session_id_dir/generated_frames_dir/
            generated_frames_dir = os.path.join(session_base_dir, "generated_frames_dir")

            directories = {
                "session_base_dir": session_base_dir,
                "video_upload_dir": video_upload_dir,
                "uploaded_frames_dir": uploaded_frames_dir,
                "generated_frames_dir": generated_frames_dir
            }

            # Create all directories
            for dir_path in directories.values():
                os.makedirs(dir_path, exist_ok=True)

            return directories
        else:
            # For images: images/session_id_dir/
            session_base_dir = os.path.join(self.images_base_dir, session_id)
            uploaded_images_dir = os.path.join(session_base_dir, "uploaded_images_dir")
            generated_images_dir = os.path.join(session_base_dir, "generated_images_dir")

            directories = {
                "session_base_dir": session_base_dir,
                "uploaded_images_dir": uploaded_images_dir,
                "generated_images_dir": generated_images_dir
            }

            # Create all directories
            for dir_path in directories.values():
                os.makedirs(dir_path, exist_ok=True)

            return directories

    async def process_video_with_session_and_create_video(self, session_id: str, video_file: UploadFile) -> Tuple[
        str, List[Dict]]:
        """
        Process video and create side-by-side comparison video

        Args:
            session_id: Session identifier
            video_file: Uploaded video file

        Returns:
            Tuple of (comparison_video_path, list of detection results for each frame)
        """
        try:
            # First process the video normally to get all frames and detections
            zip_file_path, detection_results = await self.process_video_with_session(session_id, video_file)

            # Get directories
            directories = self._create_session_directories(session_id, is_video=True)

            # Create side-by-side comparison video
            comparison_video_path = await self._create_side_by_side_video(
                session_id,
                directories["uploaded_frames_dir"],
                directories["generated_frames_dir"],
                directories["session_base_dir"]
            )

            return comparison_video_path, detection_results

        except Exception as e:
            raise Exception(f"Error processing video and creating comparison: {str(e)}")

    async def _create_side_by_side_video(self, session_id: str, original_frames_dir: str,
                                         annotated_frames_dir: str, output_dir: str) -> str:
        """
        Create a side-by-side comparison video from original and annotated frames

        Args:
            session_id: Session identifier
            original_frames_dir: Directory containing original extracted frames
            annotated_frames_dir: Directory containing annotated frames
            output_dir: Directory to save the output video

        Returns:
            Path to the created comparison video
        """
        try:
            # Get all original and annotated frame files
            original_frames = sorted([f for f in os.listdir(original_frames_dir) if f.endswith('.jpg')])
            annotated_frames = sorted(
                [f for f in os.listdir(annotated_frames_dir) if f.endswith('.jpg') and not f.endswith('.zip')])

            if not original_frames or not annotated_frames:
                raise ValueError("No frames found for video creation")

            # Setup video writer
            comparison_video_path = os.path.join(output_dir, f"comparison_video_{session_id}.mp4")

            # Read first frame to get dimensions
            first_original = cv2.imread(os.path.join(original_frames_dir, original_frames[0]))
            first_annotated = cv2.imread(os.path.join(annotated_frames_dir, annotated_frames[0]))

            if first_original is None or first_annotated is None:
                raise ValueError("Could not read frame images")

            # Resize frames to same height if needed
            height = min(first_original.shape[0], first_annotated.shape[0])
            width_original = int(first_original.shape[1] * height / first_original.shape[0])
            width_annotated = int(first_annotated.shape[1] * height / first_annotated.shape[0])

            # Total width for side-by-side
            total_width = width_original + width_annotated + 10  # 10 pixels gap

            # Video writer setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 2.0  # 2 FPS since we extracted frames every 2 seconds
            video_writer = cv2.VideoWriter(comparison_video_path, fourcc, fps, (total_width, height))

            # Process each frame pair
            min_frames = min(len(original_frames), len(annotated_frames))

            for i in range(min_frames):
                # Read original and annotated frames
                original_path = os.path.join(original_frames_dir, original_frames[i])
                annotated_path = os.path.join(annotated_frames_dir, annotated_frames[i])

                original_frame = cv2.imread(original_path)
                annotated_frame = cv2.imread(annotated_path)

                if original_frame is None or annotated_frame is None:
                    continue

                # Resize frames
                original_resized = cv2.resize(original_frame, (width_original, height))
                annotated_resized = cv2.resize(annotated_frame, (width_annotated, height))

                # Create side-by-side frame
                side_by_side = np.zeros((height, total_width, 3), dtype=np.uint8)
                side_by_side[:, :width_original] = original_resized
                side_by_side[:, width_original + 10:] = annotated_resized

                # Add labels
                cv2.putText(side_by_side, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(side_by_side, "Detected", (width_original + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)

                # Write frame to video
                video_writer.write(side_by_side)

                # Hold each frame for 2 seconds (4 additional frames at 2 FPS = 2 seconds)
                for _ in range(3):
                    video_writer.write(side_by_side)

            video_writer.release()

            if not os.path.exists(comparison_video_path):
                raise Exception("Failed to create comparison video")

            print(f"Successfully created comparison video: {comparison_video_path}")
            return comparison_video_path

        except Exception as e:
            raise Exception(f"Error creating side-by-side video: {str(e)}")

    async def process_video_with_session(self, session_id: str, video_file: UploadFile) -> Tuple[str, List[Dict]]:
        """
        Process video for scratch and dent detection by extracting frames

        Args:
            session_id: Session identifier
            video_file: Uploaded video file

        Returns:
            Tuple of (zip_file_path, list of detection results for each frame)
        """
        try:
            # Validate file type
            if not video_file.content_type.startswith("video/"):
                raise ValueError("File must be a video")

            # Create session directories
            directories = self._create_session_directories(session_id, is_video=True)

            # Save uploaded video file
            file_extension = video_file.filename.split(".")[
                -1] if "." in video_file.filename and video_file.filename else "mp4"
            video_filename = f"{session_id}_video.{file_extension}"
            video_path = os.path.join(directories["video_upload_dir"], video_filename)

            with open(video_path, "wb") as buffer:
                content = await video_file.read()
                buffer.write(content)

            # Extract frames from video
            frame_paths = self._extract_frames_from_video(
                video_path,
                directories["uploaded_frames_dir"],
                session_id
            )

            # Process each extracted frame
            detection_results = []
            processed_frame_paths = []

            for i, frame_path in enumerate(frame_paths):
                output_filename = f"annotated_frame_{i + 1}_{session_id}.jpg"
                output_path = os.path.join(directories["generated_frames_dir"], output_filename)

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

            # Create zip file with all processed images in the generated_frames_dir
            zip_path = os.path.join(directories["generated_frames_dir"], f"processed_video_{session_id}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for frame_path in processed_frame_paths:
                    if os.path.exists(frame_path):
                        zipf.write(frame_path, os.path.basename(frame_path))

            return zip_path, detection_results

        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")

    def _extract_frames_from_video(self, video_path: str, output_folder: str, session_id: str) -> List[str]:
        """
        Extract frames from video at 2-second intervals

        Args:
            video_path: Path to the video file
            output_folder: Folder to save extracted frames (uploaded_frames_dir)
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
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise ValueError("File must be an image")

            # Create session directories
            directories = self._create_session_directories(session_id, is_video=False)

            # Generate filename with session ID
            file_extension = file.filename.split(".")[-1] if "." in file.filename and file.filename else "jpg"
            input_filename = f"{session_id}_{file.filename}" if file.filename else f"{session_id}.{file_extension}"

            # Save uploaded file in uploaded_images_dir
            input_path = os.path.join(directories["uploaded_images_dir"], input_filename)
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Generate output filename in generated_images_dir
            output_filename = f"annotated_{session_id}.{file_extension}"
            output_path = os.path.join(directories["generated_images_dir"], output_filename)

            # Process the image
            processed_path, detection_counts = self._process_image_internal(input_path, output_path)

            return processed_path, detection_counts

        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

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
            # Ensure output directory exists
            output_dir = os.path.dirname(output_image_path)
            os.makedirs(output_dir, exist_ok=True)

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
            success = cv2.imwrite(output_image_path, annotated_image)
            if not success:
                raise Exception(f"Failed to save image to {output_image_path}")

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

    def ensure_base_directories_exist(self):
        """Create base directories if they don't exist"""
        os.makedirs(self.images_base_dir, exist_ok=True)
        os.makedirs(self.videos_base_dir, exist_ok=True)
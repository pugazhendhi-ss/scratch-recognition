import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from app.services.model import ScratchDentDetectionService

model_router = APIRouter()
service = ScratchDentDetectionService()


@model_router.post("/detect-scratch-dent")
async def detect_scratch_dent(file: UploadFile = File(...)):
    """
    Upload an image and get back annotated image with scratch/dent detection results
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Process image using service
        processed_path, detection_counts = await service.process_image_with_session(session_id, file)

        # Prepare response data
        response_data = {
            "scratches": detection_counts.get('Scratch', 0),
            "dents": detection_counts.get('Dent', 0)
        }

        # Get output filename for response
        output_filename = f"annotated_{session_id}.jpg"

        # Return file response with headers containing detection counts
        return FileResponse(
            path=processed_path,
            media_type="image/jpeg",
            filename=output_filename,
            headers={
                "X-Detection-Data": str(response_data),
                "X-Scratches": str(response_data["scratches"]),
                "X-Dents": str(response_data["dents"]),
                "X-Session-ID": session_id
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@model_router.post("/detect-scratch-dent-video")
async def detect_scratch_dent_video(
        file: UploadFile = File(...),
        output_format: str = Query("zip", regex="^(zip|video)$",
                                   description="Output format: 'zip' for images or 'video' for side-by-side comparison video")
):
    """
    Upload a video and get back either:
    - zip file with annotated images (output_format=zip)
    - side-by-side comparison video (output_format=video)
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())

        if output_format == "video":
            # Process video and create side-by-side comparison video
            video_path, detection_results = await service.process_video_with_session_and_create_video(session_id, file)

            # Calculate summary statistics
            total_scratches = sum(result.get("scratches", 0) for result in detection_results)
            total_dents = sum(result.get("dents", 0) for result in detection_results)
            total_frames_processed = len(detection_results)

            # Prepare response headers with summary data
            response_headers = {
                "X-Session-ID": session_id,
                "X-Total-Frames": str(total_frames_processed),
                "X-Total-Scratches": str(total_scratches),
                "X-Total-Dents": str(total_dents),
                "X-Output-Format": "video",
                "X-Detection-Summary": str({
                    "total_frames": total_frames_processed,
                    "total_scratches": total_scratches,
                    "total_dents": total_dents,
                    "frame_results": detection_results
                })
            }

            # Return side-by-side comparison video
            return FileResponse(
                path=video_path,
                media_type="video/mp4",
                filename=f"comparison_video_{session_id}.mp4",
                headers=response_headers
            )

        else:  # output_format == "zip"
            # Process video using existing service (returns zip)
            zip_file_path, detection_results = await service.process_video_with_session(session_id, file)

            # Calculate summary statistics
            total_scratches = sum(result.get("scratches", 0) for result in detection_results)
            total_dents = sum(result.get("dents", 0) for result in detection_results)
            total_frames_processed = len(detection_results)

            # Prepare response headers with summary data
            response_headers = {
                "X-Session-ID": session_id,
                "X-Total-Frames": str(total_frames_processed),
                "X-Total-Scratches": str(total_scratches),
                "X-Total-Dents": str(total_dents),
                "X-Output-Format": "zip",
                "X-Detection-Summary": str({
                    "total_frames": total_frames_processed,
                    "total_scratches": total_scratches,
                    "total_dents": total_dents,
                    "frame_results": detection_results
                })
            }

            # Return zip file containing all processed images
            return FileResponse(
                path=zip_file_path,
                media_type="application/zip",
                filename=f"processed_video_{session_id}.zip",
                headers=response_headers
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")



@model_router.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "healthy",
        "message": "Scratch & Dent Detection API is running",
        "timestamp": "2025-01-01T00:00:00Z"
    }


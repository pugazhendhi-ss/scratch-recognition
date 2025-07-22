import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.services.model import ScratchDentDetectionService

from app.config.settings import OUTPUT_DIR, UPLOADED_VIDEO_DIR
UPLOAD_DIR=UPLOADED_VIDEO_DIR


model_router = APIRouter()
service = ScratchDentDetectionService()


# Ensure directories exist
service.ensure_directories_exist(UPLOAD_DIR, OUTPUT_DIR)


@model_router.post("/detect-scratch-dent")
async def detect_scratch_dent(file: UploadFile = File(...)):
    """
    Upload an image and get back annotated image with scratch/dent detection results
    """
    try:

        # Generate unique filename
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        # Save uploaded file
        input_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process image
        output_filename = f"annotated_{unique_filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        processed_path, detection_counts = service.process_image(input_path, output_path)

        # Prepare response data
        response_data = {
            "scratches": detection_counts.get('Scratch', 0),
            "dents": detection_counts.get('Dent', 0)
        }

        # Return file response with headers containing detection counts
        return FileResponse(
            path=processed_path,
            media_type="image/jpeg",
            filename=output_filename,
            headers={
                "X-Detection-Data": str(response_data),
                "X-Scratches": str(response_data["scratches"]),
                "X-Dents": str(response_data["dents"])
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass
# Metal Defect Detection System

A computer vision system for detecting scratches and dents on metal surfaces using advanced ML models.

## Features

- Image analysis for scratch and dent detection
- Video processing with frame extraction
- Side-by-side comparison video generation
- Batch processing capabilities
- RESTful API endpoints
- Web-based user interface

## Project Structure

```
scratch-recognition/
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── nudge.py
│   └── services/
│       ├── __init__.py
│       └── model.py
├── utils/
│   └── __init__.py
├── .env
├── .gitignore
├── main.py
├── streamlit_ui.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd scratch-recognition
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file and add your ROBOFLOW_API_KEY
```

## Environment Configuration

Create a `.env` file in the root directory:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

## Usage

### Start the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Start the Web Interface

```bash
streamlit run streamlit_ui.py --server.port 8501
```

The web interface will be available at `http://localhost:8501`

## API Endpoints

### Health Check
- **GET** `/health` - Check API status

### Image Detection
- **POST** `/detect-scratch-dent` - Upload image for analysis
  - Input: Image file (JPG, PNG)
  - Output: Annotated image with detection results

### Video Detection
- **POST** `/detect-scratch-dent-video` - Upload video for analysis
  - Parameters:
    - `output_format`: `zip` (default) or `video`
  - Input: Video file (MP4, AVI, MOV)
  - Output: ZIP file with annotated frames OR side-by-side comparison video

## Response Format

All detection endpoints return:
- Detection counts in response headers
- Session ID for tracking
- Processed files

Headers:
- `X-Scratches`: Number of scratches detected
- `X-Dents`: Number of dents detected
- `X-Session-ID`: Unique session identifier

## Directory Structure

The system creates organized directories for each processing session:

```
images/
└── {session_id}/
    ├── uploaded_images_dir/
    └── generated_images_dir/

videos/
└── {session_id}/
    ├── uploaded_frames_dir/
    ├── generated_frames_dir/
    └── comparison_video_{session_id}.mp4
```

## Development

### Running in Development Mode

```bash
# API Server
uvicorn main:app --reload

# Web Interface
streamlit run streamlit_ui.py --server.port 8501
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`

## Requirements

- Python 3.8+
- OpenCV
- FastAPI
- Streamlit
- Roboflow Inference SDK
- Valid Roboflow API key

## License

MIT License
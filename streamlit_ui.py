import streamlit as st
import requests
from PIL import Image
import io
import tempfile
import os

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Scratch & Dent Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)


def check_api_health():
    """Check if the backend API is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def process_image(uploaded_file):
    """Send image to backend for processing"""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(f"{BACKEND_URL}/detect-scratch-dent", files=files)

    if response.status_code == 200:
        # Get detection counts from headers
        scratches = int(response.headers.get('X-Scratches', 0))
        dents = int(response.headers.get('X-Dents', 0))
        session_id = response.headers.get('X-Session-ID', 'unknown')

        # Convert response content to PIL Image
        processed_image = Image.open(io.BytesIO(response.content))

        return processed_image, scratches, dents, session_id
    else:
        st.error(f"Error processing image: {response.text}")
        return None, 0, 0, None


def process_video(uploaded_file, output_format):
    """Send video to backend for processing"""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    params = {"output_format": output_format}
    response = requests.post(f"{BACKEND_URL}/detect-scratch-dent-video", files=files, params=params)

    if response.status_code == 200:
        # Get detection counts from headers
        total_scratches = int(response.headers.get('X-Total-Scratches', 0))
        total_dents = int(response.headers.get('X-Total-Dents', 0))
        total_frames = int(response.headers.get('X-Total-Frames', 0))
        session_id = response.headers.get('X-Session-ID', 'unknown')
        format_type = response.headers.get('X-Output-Format', 'unknown')

        return response.content, total_scratches, total_dents, total_frames, session_id, format_type
    else:
        st.error(f"Error processing video: {response.text}")
        return None, 0, 0, 0, None, None


def main():
    # Sidebar for API health check
    with st.sidebar:
        st.header("API Status")
        if check_api_health():
            st.success("API is running")
        else:
            st.error("API is not accessible")

        st.markdown("---")
        st.markdown("**Backend URL:**")
        st.code(BACKEND_URL)

    # Main content
    st.title("SS Suite - Defect detection")
    st.markdown("Upload images or videos to detect scratches and dents in the metal surface")

    # Tab selection
    tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

    # Image Detection Tab
    with tab1:
        st.header("Metal Image Analysis")

        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )

        if uploaded_image is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Original Image")
                original_image = Image.open(uploaded_image)
                st.image(original_image, width=400)

            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    processed_image, scratches, dents, session_id = process_image(uploaded_image)

                if processed_image:
                    with col2:
                        st.subheader("Detection Results")
                        st.image(processed_image, width=400)

                    # Detection statistics
                    st.markdown("---")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)

                    with col_stats1:
                        st.metric("Scratches Detected", scratches)

                    with col_stats2:
                        st.metric("Dents Detected", dents)

                    with col_stats3:
                        st.metric("Total Defects", scratches + dents)

                    st.info(f"Session ID: {session_id}")

    # Video Detection Tab
    with tab2:
        st.header("Metal Video Analysis")

        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4'],
            key="video_uploader"
        )

        if uploaded_video is not None:
            st.subheader("Original Video")
            st.video(uploaded_video, start_time=0)

            # Output format selection
            col_format1, col_format2 = st.columns([1, 1])
            with col_format1:
                output_format = st.radio(
                    "Select output format:",
                    ["video", "zip"],
                    index=0
                )

            with col_format2:
                format_description = {
                    "video": "Get side-by-side comparison video",
                    "zip": "Get ZIP file with annotated images"
                }
                st.info(format_description[output_format])

            if st.button("Analyze Video", type="primary", use_container_width=True):
                with st.spinner("Processing video..."):
                    result_content, total_scratches, total_dents, total_frames, session_id, format_type = process_video(
                        uploaded_video, output_format)

                if result_content:
                    # Detection statistics
                    st.markdown("---")
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

                    with col_stats1:
                        st.metric("Total Scratches", total_scratches)

                    with col_stats2:
                        st.metric("Total Dents", total_dents)

                    with col_stats3:
                        st.metric("Total Defects", total_scratches + total_dents)

                    with col_stats4:
                        st.metric("Frames Processed", total_frames)

                    if format_type == "video":
                        st.subheader("Processing Complete")
                        st.info(
                            "Comparison video has been generated successfully. Use the download button below to get your results.")

                        # Download button for video only - no display
                        st.download_button(
                            label="📥 Download Comparison Video",
                            data=result_content,
                            file_name=f"comparison_video_{session_id}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

                    elif format_type == "zip":
                        st.subheader("Processing Complete")
                        st.info("ZIP file with annotated frames has been generated successfully.")

                        # Download button for ZIP
                        st.download_button(
                            label="📥 Download ZIP File",
                            data=result_content,
                            file_name=f"processed_video_{session_id}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                    st.info(f"Session ID: {session_id}")


if __name__ == "__main__":
    main()

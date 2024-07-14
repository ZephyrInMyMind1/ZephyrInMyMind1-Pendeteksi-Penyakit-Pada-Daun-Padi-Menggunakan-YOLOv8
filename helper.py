from yt_dlp import YoutubeDL
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np

def load_model(model_path):
    model = YOLO(model_path)
    return model

def resize_video_frame(frame, height):
    aspect_ratio = frame.shape[1] / frame.shape[0]
    width = int(height * aspect_ratio)
    return cv2.resize(frame, (width, height))

def _display_detected_frames(conf, model, st_frame, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    resized_frame = resize_video_frame(res_plotted, 300)
    st_frame.image(resized_frame, caption="Deteksi Objek", use_column_width=False)

def play_youtube_video(conf, model):
    url = st.text_input("Masukkan URL Video YouTube")
    if url:
        try:
            ydl_opts = {
                'format': 'best',
                'outtmpl': tempfile.mktemp(suffix='.mp4'),
                'noplaylist': True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info_dict)

            video = cv2.VideoCapture(video_path)
            st_frame = st.empty()

            if st.button('Deteksi Video YouTube', key='detect_youtube'):
                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break
                    _display_detected_frames(conf, model, st_frame, frame)

            video.release()
        except Exception as e:
            st.error(f"Error loading video: {e}")

def play_webcam(conf, model):
    video = cv2.VideoCapture(0)
    st_frame = st.empty()

    if st.button('Mulai Deteksi Webcam', key='detect_webcam'):
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            _display_detected_frames(conf, model, st_frame, frame)

        video.release()

def play_stored_video(conf, model, video_path):
    try:
        video = cv2.VideoCapture(video_path)
        st_frame = st.empty()

        if st.button('Deteksi Video Tersimpan', key='detect_stored'):
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                _display_detected_frames(conf, model, st_frame, frame)

        video.release()
    except Exception as e:
        st.error(f"Error loading video: {e}")

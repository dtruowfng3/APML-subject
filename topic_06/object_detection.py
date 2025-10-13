from ultralytics import YOLO
import yt_dlp
import cv2
import os

def download_youtube_video(url, output_path='input_video.mp4'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Video download: {output_path}")
    return output_path

def detect_objects_in_video(video_path, output_path='output_yolo.mp4'):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        annotated = results.plot()

        if out is None:
            h, w, _ = annotated.shape
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

        out.write(annotated)

    cap.release()
    out.release()
    print(f"MP4 video saved: {output_path}")

youtube_url = "https://www.youtube.com/watch?v=lbYKYpqVEmw"
video_path = download_youtube_video(youtube_url)
detect_objects_in_video(video_path)

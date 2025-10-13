# !pip install -q yt-dlp transformers torchvision opencv-python-headless

import os
import cv2
import torch
import numpy as np
from yt_dlp import YoutubeDL
from torchvision import transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# def download_video_segment(url, output_path="input_video.mp4", start="00:00:00", end="00:00:10"):
#     ydl_opts = {
#         'format': 'mp4',
#         'outtmpl': output_path,
#         'quiet': True,
#         'download_sections': [f"*{start}-{end}"],
#     }
#     with YoutubeDL(ydl_opts) as ydl:
#         ydl.download([url])
#     print(f"Video segment downloaded: {output_path}")
#     return output_path


def download_video_segment(url, output_path="input_video_segment.mp4", start_time=0, end_time=10):
    duration = end_time - start_time
    
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': 'temp_video.%(ext)s',
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    temp_files = [f for f in os.listdir('.') if f.startswith('temp_video.')]
    if not temp_files:
        raise Exception("Không tìm thấy file video tạm")
    
    temp_path = temp_files[0]
    
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if start_frame <= frame_count < end_frame:
            out.write(frame)
        
        frame_count += 1
        if frame_count >= end_frame:
            break
    
    cap.release()
    out.release()
    os.remove(temp_path)
    
    print(f"Downloaded video segment: {start_time}s-{end_time}s -> {output_path}")
    return output_path


def segment_video_save_masks(input_path, output_dir="seg_masks"):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        use_safetensors=True
    ).to(device)

    cap = cv2.VideoCapture(input_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1)[0].cpu().numpy()

        color_map = np.random.randint(0, 255, (150, 3), dtype=np.uint8)
        seg_img = color_map[preds]
        seg_img = cv2.resize(seg_img, (frame.shape[1], frame.shape[0]))

        mask_path = os.path.join(output_dir, f"mask_{frame_idx:04d}.png")
        cv2.imwrite(mask_path, seg_img)
        frame_idx += 1

    cap.release()
    print(f"Segmentation masks saved in directory: {output_dir}")


def segment_video(input_path, output_path="segformer_output_b2.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        use_safetensors=True
    ).to(device)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1)[0].cpu().numpy()

        color_map = np.random.randint(0, 255, (150, 3), dtype=np.uint8)
        seg_img = color_map[preds]
        seg_img = cv2.resize(seg_img, (frame.shape[1], frame.shape[0]))
        blended = cv2.addWeighted(frame, 0.5, seg_img, 0.5, 0)
        out.write(blended)

    cap.release()
    out.release()
    print(f"Segmented video saved: {output_path}")
    
youtube_url = "https://www.youtube.com/watch?v=lbYKYpqVEmw"
# video_path = download_video_segment(youtube_url, start="00:00:00", end="00:00:10")
video_path = download_video_segment(youtube_url, start_time=5, end_time=15)
segment_video_save_masks(video_path, output_dir="seg_masks3")
segment_video(video_path, output_path="segformer_output_b2.mp4")
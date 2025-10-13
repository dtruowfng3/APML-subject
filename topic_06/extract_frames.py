# extract_frames.py
import cv2
import os
import sys

def extract_frames(video_path="input_video_segment", out_dir="frames", every_n=1):
    if not os.path.exists(video_path):
        print("File video not found", video_path)
        return
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Can't open video file:", video_path)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total}, FPS: {fps:.2f}")

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            fname = os.path.join(out_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
            if saved % 100 == 0:
                print(f"Saved {saved} frames...")

        idx += 1

    cap.release()
    print(f"Saved {saved} frames in {out_dir}")

if __name__ == "__main__":
    extract_frames("input_video_segment.mp4", "frames", 1)

"""
accident_severity_infer.py

Run accident severity inference (minor / moderate / major) on a local
video clip, using a checkpoint produced by the accident_severity_classifier
notebook.

Setup (one time):
    pip install torch torchvision opencv-python numpy

Usage:
    python accident_severity_infer.py path/to/video.mp4
    python accident_severity_infer.py path/to/video.mp4 --checkpoint best_accident_model.pth

Note: if a video fails to load, you may need ffmpeg installed on your
system (OpenCV relies on it for many codecs):
    macOS:   brew install ffmpeg
    Windows: https://ffmpeg.org/download.html (add to PATH)
    Linux:   sudo apt install ffmpeg
"""

import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models.video import    r3d_18


# ============================================================
# Motion-Aware Frame Extraction
#
# This MUST match the extraction logic used during training.
# Any mismatch here (different windowing, different frame count,
# different resize) creates a silent distribution shift between
# training and inference, and accuracy will look worse than your
# test-set numbers even though the model itself is unchanged.
# ============================================================

def compute_motion_scores(cap, stride=2):
    scores = []
    prev_gray = None
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scores.append((idx, float(diff.mean())))

            prev_gray = gray

        idx += 1

    return scores


def find_peak_motion_window(video_path, window_seconds, stride=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = int(window_seconds * fps)

    if total_frames <= window_frames:
        cap.release()
        return 0, max(total_frames - 1, 0)

    scores = compute_motion_scores(cap, stride=stride)
    cap.release()

    if len(scores) == 0:
        return 0, total_frames - 1

    idxs = [s[0] for s in scores]
    vals = [s[1] for s in scores]

    best_start = 0
    best_sum = -1

    for i in range(len(scores)):
        w_start = idxs[i]
        w_end = w_start + window_frames
        w_sum = sum(v for ix, v in zip(idxs, vals) if w_start <= ix < w_end)

        if w_sum > best_sum:
            best_sum = w_sum
            best_start = w_start

    return best_start, min(best_start + window_frames, total_frames - 1)


def extract_frames(video_path, num_frames, image_size, window_seconds, stride=2):
    start, end = find_peak_motion_window(video_path, window_seconds, stride=stride)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    frame_indices = np.linspace(start, end, num_frames, dtype=int)

    frames = []
    current_frame = 0
    target_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        while (
            target_index < len(frame_indices)
            and current_frame == frame_indices[target_index]
        ):
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = cv2.resize(processed, (image_size, image_size))
            processed = processed.astype(np.float32) / 255.0
            frames.append(processed)
            target_index += 1

        current_frame += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No readable frames: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.array(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))

    return torch.tensor(frames, dtype=torch.float32)


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # weights=None: we're loading our own fine-tuned weights right
    # after, so there's no need to also download ImageNet/Kinetics
    # pretrained weights first.
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


# ============================================================
# Inference
# ============================================================

def predict_video(video_path, model, checkpoint, device):
    class_names = checkpoint["class_names"]
    num_frames = checkpoint["num_frames"]
    image_size = checkpoint["image_size"]
    window_seconds = checkpoint["motion_window_seconds"]
    stride = checkpoint.get("motion_sample_stride", 2)

    frames = extract_frames(
        video_path,
        num_frames=num_frames,
        image_size=image_size,
        window_seconds=window_seconds,
        stride=stride,
    )  # (T, C, H, W)

    videos = frames.unsqueeze(0)                  # (1, T, C, H, W)
    videos = videos.permute(0, 2, 1, 3, 4).to(device)  # (1, C, T, H, W)

    with torch.no_grad():
        outputs = model(videos)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_id = int(probs.argmax())

    return {
        "predicted_class": class_names[pred_id],
        "confidence": float(probs[pred_id]),
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
    }


def main():
    parser = argparse.ArgumentParser(description="Accident severity inference")
    parser.add_argument("video", help="Path to a video file (15-30s crash clip)")
    parser.add_argument(
        "--checkpoint",
        default="accident_severity_model.pth",
        help="Path to the model checkpoint (default: accident_severity_model.pth)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, checkpoint = load_model(args.checkpoint, device)
    result = predict_video(args.video, model, checkpoint, device)

    print()
    print(f"Predicted severity : {result['predicted_class'].upper()}")
    print(f"Confidence         : {result['confidence'] * 100:.1f}%")
    print()
    print("Full breakdown:")
    for cls, p in result["probabilities"].items():
        print(f"  {cls:10s}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
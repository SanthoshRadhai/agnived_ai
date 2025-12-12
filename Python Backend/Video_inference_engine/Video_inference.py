from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Tuple, Dict, Any

import cv2
import torch
import yt_dlp
import numpy as np
from PIL import Image
import json
from datetime import timedelta

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification


# -----------------------------------------------------------------------------#
# Configuration                                                                #
# -----------------------------------------------------------------------------#
BUFFER_MINUTES = 10          # 10-minute rolling buffer (live frames kept in RAM)
FRAME_SKIP = 20              # detect every 10th frame (like test_video.py)
CLASSIFICATION_FRAMES_PER_TRACK = 3
MODEL_IMAGE_SIZE = 224
TRACK_TIMEOUT = 60           # frames before track expires
OUTPUT_DIR = Path("video_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------#
# Core detector class (mirrors Test_CameraTraps/test_video.py)                 #
# -----------------------------------------------------------------------------#
class VideoWildlifeDetector:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.detector = None
        self.classifiers: Dict[str, Any] = {}
        self.track_classifications: Dict[int, Dict[str, Any]] = {}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 0
        self.frame_buffer: Dict[int, list] = {}
        self.current_frame_detections: Dict[int, Dict[str, Any]] = {}

    # --- model loading -------------------------------------------------------
    def load_models(self):
        print("=" * 60)
        print("LOADING MODELS")
        print("=" * 60)

        # MegaDetector
        print("\n📥 Loading MegaDetector v6...")
        self.detector = pw_detection.MegaDetectorV6(
            device=self.device,
            pretrained=True,
            version="MDV6-yolov10-e",
        )
        print("✓ MegaDetector loaded")

        # Classifiers (same set as test_video, Deepfaune optional)
        classifier_configs = {
            "Serengeti": lambda: pw_classification.AI4GSnapshotSerengeti(
                device=self.device, pretrained=True
            ),
            # Uncomment if you want Deepfaune and have downloaded its weights once
            # "Deepfaune": lambda: pw_classification.DFNE(device=self.device),
            "Amazon": lambda: pw_classification.AI4GAmazonRainforest(
                device=self.device, pretrained=True
            ),
        }

        for name, loader in classifier_configs.items():
            print(f"\n📥 Loading {name}...")
            try:
                self.classifiers[name] = loader()
                print(f"✓ {name} loaded")
            except Exception as e:
                print(f"⚠️  Failed to load {name}: {e}")

        print(f"\n✓ All models loaded on {self.device.upper()}")

    # --- helpers -------------------------------------------------------------
    def iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(
            0, inter_y_max - inter_y_min
        )
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    # --- detection / classification -----------------------------------------
    def detect_frame(self, frame_rgb, frame_idx):
        """Detect animals in a single frame (same as test_video.detect_frame)."""
        temp_path = f"/tmp/frame_{frame_idx}.jpg"
        Image.fromarray(frame_rgb).save(temp_path)

        result = self.detector.single_image_detection(temp_path)
        detections_obj = result["detections"]

        if len(detections_obj) == 0 or len(detections_obj.xyxy) == 0:
            return []

        detections_obj = detections_obj[0]
        animals = []
        for det_idx in range(len(detections_obj.xyxy)):
            category_id = int(detections_obj.class_id[det_idx])
            if category_id == 0:  # Animal
                animals.append(
                    {
                        "bbox": detections_obj.xyxy[det_idx].tolist(),
                        "confidence": float(detections_obj.confidence[det_idx]),
                    }
                )
        return animals

    def match_detection_to_track(self, detection, frame_idx):
        """Match detection to existing track or create new (same as test_video)."""
        best_match = None
        best_iou = 0.3

        for track_id, track_data in self.active_tracks.items():
            if frame_idx - track_data["last_seen"] > TRACK_TIMEOUT:
                continue

            overlap = self.iou(detection["bbox"], track_data["last_bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_match = track_id

        if best_match is not None:
            self.active_tracks[best_match]["last_bbox"] = detection["bbox"]
            self.active_tracks[best_match]["last_seen"] = frame_idx
            self.active_tracks[best_match]["detections"].append(detection)
            return best_match

        track_id = self.next_track_id
        self.next_track_id += 1

        self.active_tracks[track_id] = {
            "last_bbox": detection["bbox"],
            "last_seen": frame_idx,
            "first_seen": frame_idx,
            "detections": [detection],
            "frames": [],
        }
        self.frame_buffer[track_id] = []
        return track_id

    def resize_image(self, img: Image.Image, target_size: int) -> Image.Image:
        width, height = img.size
        scale = min(target_size / width, target_size / height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))

        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        padded.paste(img_resized, (paste_x, paste_y))

        return padded

    def classify_detection_realtime(self, frame_rgb, bbox):
        """Classify a single detection (same logic as test_video.classify_detection_realtime)."""
        x1, y1, x2, y2 = bbox
        h, w = frame_rgb.shape[:2]
        pad = 0.15

        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - width * pad)
        y1 = max(0, y1 - height * pad)
        x2 = min(w, x2 + width * pad)
        y2 = min(h, y2 + height * pad)

        cropped = frame_rgb[int(y1) : int(y2), int(x1) : int(x2)]
        cropped_pil = Image.fromarray(cropped)

        cropped_resized = self.resize_image(cropped_pil, MODEL_IMAGE_SIZE)
        temp_crop = "/tmp/classify_crop.jpg"
        cropped_resized.save(temp_crop)

        model_predictions = {}
        for model_name, classifier in self.classifiers.items():
            try:
                result = classifier.single_image_classification(temp_crop)
                if isinstance(result, dict) and "prediction" in result:
                    model_predictions[model_name] = {
                        "species": result["prediction"],
                        "confidence": result.get("confidence", 0.0),
                    }
            except Exception as e:
                print(f"⚠️  {model_name} failed: {e}")

        if model_predictions:
            best_pred = max(model_predictions.values(), key=lambda x: x["confidence"])
            best_species = best_pred["species"]
            best_confidence = best_pred["confidence"]
            return best_species, best_confidence, model_predictions

        return None, 0.0, {}

    # --- drawing & per-frame pipeline ---------------------------------------
    def draw_realtime_annotations(self, frame_rgb, frame_idx, total_frames=None):
        """Draw annotations with current classifications (same as test_video.draw_realtime_annotations)."""
        annotated = frame_rgb.copy()

        for track_id, detection_info in self.current_frame_detections.items():
            bbox = detection_info["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            if track_id in self.track_classifications:
                classification = self.track_classifications[track_id]
                species = classification["species"]
                confidence = classification["confidence"]
                model_predictions = classification["model_predictions"]

                if confidence >= 0.75:
                    color = (0, 255, 0)
                elif confidence >= 0.50:
                    color = (255, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                label = f"ID{track_id}: {species} ({confidence:.1%})"
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 10, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                panel_y = y2 + 20
                for model_name, pred in model_predictions.items():
                    model_label = (
                        f"{model_name}: {pred['species']} ({pred['confidence']:.1%})"
                    )
                    cv2.putText(
                        annotated,
                        model_label,
                        (x1, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )
                    panel_y += 18
            else:
                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2), (128, 128, 128), 2
                )
                label = f"ID{track_id}: Classifying..."
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 128, 128),
                    2,
                )

        if total_frames is not None:
            cv2.putText(
                annotated,
                f"Frame: {frame_idx}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        return annotated

    def process_stream_frame(self, frame_rgb, frame_idx, total_frames=None):
        """
        Live stream pipeline:
        - Detect every FRAME_SKIP frames using MegaDetector.
        - Focus on ONE highest-confidence animal per detection cycle.
        - After a few frames per track, classify with ensemble.
        - For in-between frames, carry over last bbox.
        """
        self.current_frame_detections = {}

        if frame_idx % FRAME_SKIP == 0:
            detections = self.detect_frame(frame_rgb, frame_idx)

            # Focus on only one animal (highest-confidence detection)
            if detections:
                best_det = max(detections, key=lambda d: d["confidence"])
                detections = [best_det]

            for detection in detections:
                track_id = self.match_detection_to_track(detection, frame_idx)

                self.current_frame_detections[track_id] = {
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                }

                self.frame_buffer[track_id].append(
                    {
                        "frame": frame_rgb.copy(),
                        "bbox": detection["bbox"],
                        "confidence": detection["confidence"],
                    }
                )

                if (
                    track_id not in self.track_classifications
                    and len(self.frame_buffer[track_id])
                    >= CLASSIFICATION_FRAMES_PER_TRACK
                ):
                    best_frames = sorted(
                        self.frame_buffer[track_id],
                        key=lambda x: x["confidence"],
                        reverse=True,
                    )[:CLASSIFICATION_FRAMES_PER_TRACK]

                    best_frame = best_frames[0]
                    species, confidence, model_preds = self.classify_detection_realtime(
                        best_frame["frame"], best_frame["bbox"]
                    )

                    if species:
                        self.track_classifications[track_id] = {
                            "species": species,
                            "confidence": confidence,
                            "model_predictions": model_preds,
                        }
        else:
            # Carry forward active tracks using last known bbox
            for track_id, track_data in self.active_tracks.items():
                if frame_idx - track_data["last_seen"] <= FRAME_SKIP:
                    self.current_frame_detections[track_id] = {
                        "bbox": track_data["last_bbox"],
                        "confidence": track_data["detections"][-1]["confidence"]
                        if track_data["detections"]
                        else 0.0,
                    }

        annotated = self.draw_realtime_annotations(
            frame_rgb, frame_idx, total_frames=total_frames
        )
        return annotated

    def save_results(self, video_name: str, fps: float):
        """Save detection/classification summary to JSON (same idea as test_video)."""
        output_file = OUTPUT_DIR / f"{video_name}_results.json"

        results = {}
        for track_id, track_data in self.active_tracks.items():
            if track_id in self.track_classifications:
                classification = self.track_classifications[track_id]

                first_frame = track_data["first_seen"]
                last_frame = track_data["last_seen"]

                results[f"track_{track_id}"] = {
                    "species": classification["species"],
                    "confidence": float(classification["confidence"]),
                    "first_seen": str(timedelta(seconds=int(first_frame / fps))),
                    "last_seen": str(timedelta(seconds=int(last_frame / fps))),
                    "frame_count": len(track_data["detections"]),
                    "model_predictions": {
                        k: {
                            "species": v["species"],
                            "confidence": float(v["confidence"]),
                        }
                        for k, v in classification["model_predictions"].items()
                    },
                }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        # Simple summary
        species_count = defaultdict(int)
        for data in results.values():
            species_count[data["species"]] += 1

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"\nTotal unique animals (tracks): {len(results)}")
        print("\nSpecies breakdown:")
        for species, count in species_count.items():
            print(f"  {species}: {count}")


# -----------------------------------------------------------------------------#
# YouTube live streaming + 10-min buffer                                       #
# -----------------------------------------------------------------------------#
def get_youtube_stream_url(youtube_url: str) -> str:
    """
    Use yt_dlp to get direct media URL.
    We already constrain to <=720p and <=30fps, so no extra resize needed.
    """
    ydl_opts = {
        "format": "best[height<=720][fps<=30]/best[height<=1080]/best",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


def run_youtube_live_inference(youtube_url: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("AGNIVED LIVE YOUTUBE WILDLIFE INFERENCE")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"YouTube URL: {youtube_url}\n")

    print("Resolving YouTube stream URL...")
    stream_url = get_youtube_stream_url(youtube_url)
    print("Stream URL resolved.\n")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Could not open YouTube stream with OpenCV.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    print(f"Estimated FPS: {fps:.2f}")

    max_buffer_frames = int(BUFFER_MINUTES * 60 * fps)
    print(f"Buffering up to {BUFFER_MINUTES} minutes (~{max_buffer_frames} frames)\n")

    buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=max_buffer_frames)

    detector = VideoWildlifeDetector(device=device)
    detector.load_models()

    frame_idx = 0
    window_name = "AgniVed Live Wildlife Inference (YouTube)"

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Stream ended or temporarily unavailable.")
            break

        frame_idx += 1

        buffer.append((frame_idx, frame_bgr.copy()))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated_rgb = detector.process_stream_frame(
            frame_rgb, frame_idx, total_frames=None
        )
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, annotated_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            print("Stopping inference by user request.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTotal frames seen: {frame_idx}")
    print(f"Frames kept in {BUFFER_MINUTES}-minute buffer: {len(buffer)}")
    detector.save_results("youtube_live", fps)


if __name__ == "__main__":
    # Set your desired live URL here
    YOUTUBE_LIVE_URL = "https://www.youtube.com/watch?v=WaszO4l4E2c"
    run_youtube_live_inference(YOUTUBE_LIVE_URL)
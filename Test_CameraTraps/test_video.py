import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PIL import Image
import json
from datetime import timedelta
import time

# Configuration
TEST_VIDEO_PATH = "/home/ray/Downloads/Example Footage _ Browning Defender Pro Scout Max Solar Cellular Trail Camera.mp4"
OUTPUT_DIR = Path("/home/ray/agnived_models/video_results")
FRAME_SKIP = 10  # Detect every 10th frame for smoother real-time
CLASSIFICATION_FRAMES_PER_TRACK = 3
MODEL_IMAGE_SIZE = 224
TRACK_TIMEOUT = 60  # Frames before track expires

class VideoWildlifeDetector:
    def __init__(self, device="cuda"):
        self.device = device
        self.detector = None
        self.classifiers = {}
        self.track_classifications = {}
        self.active_tracks = {}  # For real-time tracking
        self.next_track_id = 0
        self.frame_buffer = {}  # Store frames for each track
        self.current_frame_detections = {}  # NEW: Store current frame's detections
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_models(self):
        """Load detection and classification models"""
        print("=" * 60)
        print("LOADING MODELS")
        print("=" * 60)
        
        # Load MegaDetector
        print("\n📥 Loading MegaDetector v6...")
        self.detector = pw_detection.MegaDetectorV6(
            device=self.device,
            pretrained=True,
            version="MDV6-yolov10-e"
        )
        print("✓ MegaDetector loaded")
        
        # Load classifiers
        classifier_configs = {
            "Serengeti": lambda: pw_classification.AI4GSnapshotSerengeti(
                device=self.device, pretrained=True
            ),
            "Deepfaune": lambda: pw_classification.DFNE(device=self.device),
            "Amazon": lambda: pw_classification.AI4GAmazonRainforest(
                device=self.device, pretrained=True
            )
        }
        
        for name, loader in classifier_configs.items():
            print(f"\n📥 Loading {name}...")
            try:
                self.classifiers[name] = loader()
                print(f"✓ {name} loaded")
            except Exception as e:
                print(f"⚠️  Failed to load {name}: {e}")
        
        print(f"\n✓ All models loaded on {self.device.upper()}")
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def detect_frame(self, frame, frame_idx):
        """Detect animals in a single frame"""
        # Save frame temporarily
        temp_path = f"/tmp/frame_{frame_idx}.jpg"
        frame_pil = Image.fromarray(frame)
        frame_pil.save(temp_path)
        
        # Run detection
        result = self.detector.single_image_detection(temp_path)
        detections_obj = result['detections']
        
        # Check if detections is empty
        if len(detections_obj) == 0 or len(detections_obj.xyxy) == 0:
            return []
        
        detections_obj = detections_obj[0]
        
        # Extract animal detections
        animals = []
        for det_idx in range(len(detections_obj.xyxy)):
            category_id = int(detections_obj.class_id[det_idx])
            
            if category_id == 0:  # Animal
                animals.append({
                    'bbox': detections_obj.xyxy[det_idx].tolist(),
                    'confidence': float(detections_obj.confidence[det_idx])
                })
        
        return animals
    
    def match_detection_to_track(self, detection, frame_idx):
        """Match detection to existing track or create new"""
        best_match = None
        best_iou = 0.3
        
        # Try to match with existing tracks
        for track_id, track_data in self.active_tracks.items():
            if frame_idx - track_data['last_seen'] > TRACK_TIMEOUT:
                continue
            
            overlap = self.iou(detection['bbox'], track_data['last_bbox'])
            if overlap > best_iou:
                best_iou = overlap
                best_match = track_id
        
        if best_match is not None:
            # Update existing track
            self.active_tracks[best_match]['last_bbox'] = detection['bbox']
            self.active_tracks[best_match]['last_seen'] = frame_idx
            self.active_tracks[best_match]['detections'].append(detection)
            return best_match
        else:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.active_tracks[track_id] = {
                'last_bbox': detection['bbox'],
                'last_seen': frame_idx,
                'first_seen': frame_idx,
                'detections': [detection],
                'frames': []
            }
            self.frame_buffer[track_id] = []
            
            return track_id
    
    def classify_detection_realtime(self, frame, bbox):
        """Classify a single detection immediately"""
        # Expand bbox
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 0.15
        
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, x1 - width * pad)
        y1 = max(0, y1 - height * pad)
        x2 = min(w, x2 + width * pad)
        y2 = min(h, y2 + height * pad)
        
        # Crop
        cropped = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_pil = Image.fromarray(cropped)
        
        # Resize
        cropped_resized = self.resize_image(cropped_pil, MODEL_IMAGE_SIZE)
        
        # Save
        temp_crop = f"/tmp/classify_crop.jpg"
        cropped_resized.save(temp_crop)
        
        # Run all classifiers
        model_predictions = {}
        for model_name, classifier in self.classifiers.items():
            try:
                result = classifier.single_image_classification(temp_crop)
                if isinstance(result, dict) and 'prediction' in result:
                    model_predictions[model_name] = {
                        'species': result['prediction'],
                        'confidence': result.get('confidence', 0.0)
                    }
            except Exception as e:
                print(f"⚠️  {model_name} failed: {e}")
        
        # Ensemble vote - get highest confidence prediction
        if model_predictions:
            # Find the prediction with highest confidence across all models
            best_pred = max(model_predictions.values(), key=lambda x: x['confidence'])
            best_species = best_pred['species']
            best_confidence = best_pred['confidence']
            
            return best_species, best_confidence, model_predictions
        
        return None, 0.0, {}
    
    def resize_image(self, img, target_size):
        """Resize maintaining aspect ratio with padding"""
        width, height = img.size
        scale = min(target_size / width, target_size / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        padded.paste(img_resized, (paste_x, paste_y))
        
        return padded
    
    def draw_realtime_annotations(self, frame, frame_idx):
        """Draw annotations with current classifications"""
        annotated = frame.copy()
        
        # Draw only detections from current frame stored in self.current_frame_detections
        for track_id, detection_info in self.current_frame_detections.items():
            bbox = detection_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Check if we have classification for this track
            if track_id in self.track_classifications:
                classification = self.track_classifications[track_id]
                species = classification['species']
                confidence = classification['confidence']
                model_predictions = classification['model_predictions']
                
                # Color by confidence
                if confidence >= 0.75:
                    color = (0, 255, 0)  # Green
                elif confidence >= 0.50:
                    color = (255, 255, 0)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                # Main label with highest confidence prediction
                label = f"ID{track_id}: {species} ({confidence:.1%})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show all model predictions below the box
                panel_y = y2 + 20
                for model_name, pred in model_predictions.items():
                    model_label = f"{model_name}: {pred['species']} ({pred['confidence']:.1%})"
                    cv2.putText(annotated, model_label, (x1, panel_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    panel_y += 18
            else:
                # Still detecting/classifying
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
                label = f"ID{track_id}: Classifying..."
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        
        return annotated
    
    def process_video_realtime(self, video_path):
        """Process video in real-time with live display"""
        video_path = Path(video_path)
        
        print("=" * 60)
        print(f"PROCESSING VIDEO: {video_path.name}")
        print("=" * 60)
        print("Press 'q' to quit, 'p' to pause/resume")
        print("=" * 60)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {timedelta(seconds=int(total_frames / fps))}")
        
        frame_idx = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Clear previous frame's detections
                self.current_frame_detections = {}
                
                # Detect every Nth frame
                if frame_idx % FRAME_SKIP == 0:
                    print(f"\n🔍 Frame {frame_idx}/{total_frames} - Detecting...")
                    
                    detections = self.detect_frame(frame_rgb, frame_idx)
                    
                    if detections:
                        print(f"   Found {len(detections)} animal(s)")
                        
                        for detection in detections:
                            # Match to track
                            track_id = self.match_detection_to_track(detection, frame_idx)
                            
                            # Store current frame detection for this track
                            self.current_frame_detections[track_id] = {
                                'bbox': detection['bbox'],
                                'confidence': detection['confidence']
                            }
                            
                            # Store frame for this track
                            self.frame_buffer[track_id].append({
                                'frame': frame_rgb.copy(),
                                'bbox': detection['bbox'],
                                'confidence': detection['confidence']
                            })
                            
                            # Classify after collecting a few frames
                            if track_id not in self.track_classifications and len(self.frame_buffer[track_id]) >= CLASSIFICATION_FRAMES_PER_TRACK:
                                print(f"   🧠 Classifying Track {track_id}...")
                                
                                # Use best frames
                                best_frames = sorted(self.frame_buffer[track_id], 
                                                   key=lambda x: x['confidence'], 
                                                   reverse=True)[:CLASSIFICATION_FRAMES_PER_TRACK]
                                
                                # Classify first frame to get quick result
                                best_frame = best_frames[0]
                                species, confidence, model_preds = self.classify_detection_realtime(
                                    best_frame['frame'], 
                                    best_frame['bbox']
                                )
                                
                                if species:
                                    self.track_classifications[track_id] = {
                                        'species': species,
                                        'confidence': confidence,
                                        'model_predictions': model_preds
                                    }
                                    print(f"   ✓ Track {track_id}: {species} ({confidence:.1%})")
                else:
                    # On non-detection frames, carry forward active tracks using last known bbox
                    for track_id, track_data in self.active_tracks.items():
                        if frame_idx - track_data['last_seen'] <= FRAME_SKIP:
                            # Still active - show last known position
                            self.current_frame_detections[track_id] = {
                                'bbox': track_data['last_bbox'],
                                'confidence': track_data['detections'][-1]['confidence'] if track_data['detections'] else 0.0
                            }
                
                # Draw annotations using current frame's detections
                annotated = self.draw_realtime_annotations(frame_rgb, frame_idx)
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                
                # Add frame counter
                cv2.putText(annotated_bgr, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('AgniVed Wildlife Detection - Live', annotated_bgr)
                
                frame_idx += 1
            
            # Adjust wait time for smoother playback
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸ Paused" if paused else "▶ Resumed")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save results
        self.save_results(video_path.stem, fps)
    
    def save_results(self, video_name, fps):
        """Save detection results to JSON"""
        output_file = OUTPUT_DIR / f"{video_name}_results.json"
        
        results = {}
        for track_id, track_data in self.active_tracks.items():
            if track_id in self.track_classifications:
                classification = self.track_classifications[track_id]
                
                first_frame = track_data['first_seen']
                last_frame = track_data['last_seen']
                
                results[f"track_{track_id}"] = {
                    'species': classification['species'],
                    'confidence': float(classification['confidence']),
                    'first_seen': str(timedelta(seconds=int(first_frame / fps))),
                    'last_seen': str(timedelta(seconds=int(last_frame / fps))),
                    'frame_count': len(track_data['detections']),
                    'model_predictions': {k: {'species': v['species'], 'confidence': float(v['confidence'])} 
                                         for k, v in classification['model_predictions'].items()}
                }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        
        species_count = defaultdict(int)
        for data in results.values():
            species_count[data['species']] += 1
        
        print(f"\nTotal unique animals: {len(results)}")
        print("\nSpecies breakdown:")
        for species, count in species_count.items():
            print(f"  {species}: {count}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}\n")
    
    detector = VideoWildlifeDetector(device=device)
    detector.load_models()
    
    # Process video in real-time
    detector.process_video_realtime(TEST_VIDEO_PATH)

if __name__ == "__main__":
    main()
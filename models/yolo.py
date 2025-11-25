import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

# Fix for PyTorch 2.6+ weights_only security
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except:
    pass

class YOLOModel:
    def __init__(self, model_size='n'):
        """
        Initialize YOLO model
        Args:
            model_size: 'n', 's', 'm', 'l', or 'x' for different model sizes
        """
        model_name = f'yolov8{model_size}.pt'
        print(f"Loading YOLOv8{model_size.upper()}...")
        # Use weights_only=False for trusted ultralytics models
        self.model = YOLO(model_name)
        self.model_name = f"YOLOv8{model_size.upper()}"
        
    def predict_image(self, image_path, conf_threshold=0.25):
        """
        Predict objects in a single image
        Returns: results object with detections
        """
        results = self.model(image_path, conf=conf_threshold)
        return results[0]
    
    def predict_video(self, video_path, output_path, conf_threshold=0.25):
        """
        Process entire video and save annotated output
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Time the inference
            start_time = time.time()
            results = self.model(frame, conf=conf_threshold, verbose=False)
            inference_time = time.time() - start_time
            processing_times.append(inference_time)
            
            # Draw annotations
            annotated_frame = results[0].plot()
            
            # Add performance info
            fps_text = f"FPS: {1/inference_time:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # Calculate statistics
        avg_fps = 1 / np.mean(processing_times)
        print(f"✅ YOLOv8 processing complete!")
        print(f"Average FPS: {avg_fps:.2f}")
        
        return {
            'avg_fps': avg_fps,
            'processing_times': processing_times,
            'total_frames': frame_count
        }
    
    def evaluate_on_images(self, image_paths, conf_threshold=0.25):
        """
        Evaluate model on multiple images
        Returns detection statistics
        """
        all_results = []
        processing_times = []
        
        for img_path in image_paths:
            start_time = time.time()
            results = self.predict_image(img_path, conf_threshold)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            all_results.append(results)
        
        avg_fps = 1 / np.mean(processing_times)
        
        return {
            'results': all_results,
            'avg_fps': avg_fps,
            'processing_times': processing_times
        }

# Example usage
if __name__ == "__main__":
    # Test the model
    yolo = YOLOModel('n')  # Use nano model for speed
    
    # Test on single image
    # results = yolo.predict_image('path/to/image.jpg')
    # print(f"Found {len(results.boxes)} objects")
import cv2
import time
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class MaskRCNNModel:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize Mask R-CNN model using TorchVision
        """
        print("Loading Mask R-CNN (TorchVision)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained model with new API
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        self.model_name = "Mask R-CNN"
        
    def predict_image(self, image_path):
        """
        Predict objects and masks in a single image
        """
        # Handle Path objects, strings, and numpy arrays
        if isinstance(image_path, (str, Path)):
            image = Image.open(str(image_path)).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            # Convert from OpenCV BGR to RGB
            image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        else:
            image = image_path
        
        # Transform image
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        return predictions, np.array(image)
    
    def draw_predictions(self, image, predictions):
        """
        Draw bounding boxes, labels, and masks on image
        """
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()
        
        # Filter by confidence
        mask_filter = scores > self.confidence_threshold
        boxes = boxes[mask_filter]
        labels = labels[mask_filter]
        scores = scores[mask_filter]
        masks = masks[mask_filter]
        
        # Draw on image
        annotated = image.copy()
        
        # Generate random colors for each instance
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=np.uint8)
        
        for idx, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[idx].tolist()
            
            # Draw mask
            mask = mask[0] > 0.5  # Threshold mask
            mask_overlay = annotated.copy()
            mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array(color) * 0.5
            annotated = mask_overlay.astype(np.uint8)
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            cv2.putText(annotated, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def predict_video(self, video_path, output_path):
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
            predictions, _ = self.predict_image(frame)
            annotated_frame = self.draw_predictions(frame, predictions)
            inference_time = time.time() - start_time
            processing_times.append(inference_time)
            
            # Add performance info
            fps_text = f"FPS: {1/inference_time:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # Calculate statistics
        avg_fps = 1 / np.mean(processing_times)
        print(f"✅ Mask R-CNN processing complete!")
        print(f"Average FPS: {avg_fps:.2f}")
        
        return {
            'avg_fps': avg_fps,
            'processing_times': processing_times,
            'total_frames': frame_count
        }
    
    def evaluate_on_images(self, image_paths):
        """
        Evaluate model on multiple images
        """
        all_predictions = []
        processing_times = []
        
        for img_path in image_paths:
            start_time = time.time()
            predictions, image = self.predict_image(img_path)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            all_predictions.append(predictions)
        
        avg_fps = 1 / np.mean(processing_times)
        
        return {
            'outputs': all_predictions,
            'avg_fps': avg_fps,
            'processing_times': processing_times
        }
    
# Example usage
if __name__ == "__main__":
    # Test the model
    mask_rcnn = MaskRCNNModel()
    print("✅ Model loaded successfully!")
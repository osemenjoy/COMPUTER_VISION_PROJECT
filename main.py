"""
Simplified Object Recognition System
Outputs: Video with position, shape, and name of detected objects
"""
import os
from pathlib import Path
from models.yolo import YOLOModel
from models.mask_rcnn import MaskRCNNModel
from models.faster_rcnn import FasterRCNNModel
from utils.visualize import generate_summary_report

def setup_directories():
    """Create output directories"""
    dirs = ['output', 'output/videos', 'output/comparison']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Output directories created\n")

def create_test_video():
    """Create a short test video from COCO images"""
    import cv2
    import random
    
    print("Creating test video from sample images...")
    
    images_dir = Path('data/val2017')
    if not images_dir.exists():
        print("❌ COCO dataset not found at data/val2017")
        return None
    
    # Get 50 random images
    all_images = list(images_dir.glob('*.jpg'))
    sample_images = random.sample(all_images, min(50, len(all_images)))
    
    # Read first image for dimensions
    first_img = cv2.imread(str(sample_images[0]))
    height, width = 480, 640  # Standard size
    
    # Create video
    output_path = 'output/test_input.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
    
    for img_path in sample_images:
        img = cv2.imread(str(img_path))
        img_resized = cv2.resize(img, (width, height))
        video.write(img_resized)
    
    video.release()
    print(f"✅ Test video created: {output_path}")
    print(f"   - Duration: {len(sample_images)/10:.1f} seconds")
    print(f"   - Frames: {len(sample_images)}\n")
    
    return output_path

def process_video_with_model(model, model_name, input_video, output_video):
    """
    Process video with a model and save output
    
    Output video shows:
    - Position: Bounding boxes around objects
    - Name: Object labels (person, car, dog, etc.)
    - Shape: Segmentation masks (for Mask R-CNN)
    """
    print(f"{'='*70}")
    print(f"Processing with {model_name}")
    print(f"{'='*70}")
    
    result = model.predict_video(input_video, output_video)
    
    print(f"\n✅ {model_name} Output:")
    print(f"   - Video saved: {output_video}")
    print(f"   - Processing speed: {result['avg_fps']:.2f} FPS")
    print(f"   - Total frames: {result['total_frames']}")
    print()
    
    return result

def compare_models_on_video(input_video):
    """
    Compare all three models on the same video
    
    This is what the coursework asks for:
    - Evaluate 3 state-of-the-art approaches
    - Show position, shape, and name of objects
    - Present best system in final video
    """
    
    print("\n" + "="*70)
    print("OBJECT RECOGNITION SYSTEM - MODEL COMPARISON")
    print("="*70)
    print(f"\nInput video: {input_video}\n")
    
    # Initialize models
    print("Loading models...\n")
    yolo = YOLOModel('n')
    mask_rcnn = MaskRCNNModel()
    faster_rcnn = FasterRCNNModel()
    
    results = {}
    
    # Process with each model
    output_files = {
        'YOLOv8': 'output/videos/1_yolov8_output.mp4',
        'Mask R-CNN': 'output/videos/2_mask_rcnn_output.mp4',
        'Faster R-CNN': 'output/videos/3_faster_rcnn_output.mp4'
    }
    
    # YOLOv8
    results['YOLOv8'] = process_video_with_model(
        yolo, 'YOLOv8', input_video, output_files['YOLOv8']
    )
    
    # Mask R-CNN
    results['Mask R-CNN'] = process_video_with_model(
        mask_rcnn, 'Mask R-CNN', input_video, output_files['Mask R-CNN']
    )
    
    # Faster R-CNN
    results['Faster R-CNN'] = process_video_with_model(
        faster_rcnn, 'Faster R-CNN', input_video, output_files['Faster R-CNN']
    )
    
    # Generate comparison report
    print("="*70)
    print("GENERATING COMPARISON ANALYSIS")
    print("="*70 + "\n")
    
    generate_summary_report(results, 'output/comparison')
    
    # Determine best model
    best_model = max(results.keys(), key=lambda x: results[x]['avg_fps'])
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nAll models show:")
    print("  ✓ Position: Bounding boxes around detected objects")
    print("  ✓ Name: Object labels (person, car, dog, etc.)")
    print("  ✓ Shape: Segmentation masks (Mask R-CNN only)\n")
    
    print("Performance Comparison:")
    for model, data in results.items():
        print(f"  - {model:15s}: {data['avg_fps']:6.2f} FPS")
    
    print(f"\n🏆 Fastest Model: {best_model}")
    print(f"   Recommended for real-time applications")
    
    print(f"\n🎯 Best Quality: Mask R-CNN")
    print(f"   Provides detailed segmentation masks (shape)")
    
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print("\nProcessed Videos (position + name + shape):")
    for model, path in output_files.items():
        print(f"  - {path}")
    
    print("\nComparison Analysis:")
    print(f"  - output/comparison/performance_comparison.png")
    print(f"  - output/comparison/comparison_table.csv")
    print(f"  - output/comparison/summary_report.md")
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR YOUR REPORT")
    print("="*70)
    print("""
1. Upload ONE of the output videos to YouTube/OneDrive
   Recommended: Mask R-CNN (best quality, shows shape)
   
2. In your report, include:
   - Link to the uploaded video
   - Performance comparison (from comparison_table.csv)
   - Qualitative analysis (which model performs best in what scenarios)
   - Screenshots showing position, shape, and name annotations
   
3. Key points to discuss:
   - YOLOv8: Fastest, good for real-time
   - Mask R-CNN: Best quality, shows object shapes
   - Faster R-CNN: Balanced speed and accuracy
    """)

def main():
    """Main entry point"""
    setup_directories()
    
    print("="*70)
    print("OBJECT RECOGNITION SYSTEM")
    print("Output: Video with Position, Shape, and Name of Objects")
    print("="*70 + "\n")
    
    print("Options:")
    print("1. Create test video from COCO images and process it")
    print("2. Process your own video file")
    
    choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == '1':
        # Create test video
        input_video = create_test_video()
        if not input_video:
            print("❌ Failed to create test video")
            return
        
        # Process with all models
        compare_models_on_video(input_video)
        
    elif choice == '2':
        # Use custom video
        video_path = input("Enter path to your video file: ").strip()

        # Accept quoted paths (e.g. "data\\video_sample.webm") by removing surrounding quotes
        if (video_path.startswith('"') and video_path.endswith('"')) or (video_path.startswith("'") and video_path.endswith("'")):
            video_path = video_path[1:-1]

        # Expand user (~) and normalize separators so both slashes and backslashes work
        video_path = os.path.normpath(os.path.expanduser(video_path))

        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return

        # Process with all models
        compare_models_on_video(video_path)
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
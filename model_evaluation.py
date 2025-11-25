"""
Complete Model Evaluation on Video Dataset
Evaluates models on video sequences from data folder

This script provides:
- Temporal consistency metrics
- Processing speed on video
- Comparison of 3 state-of-the-art models
- Results for your report
"""
import cv2
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from models.yolo import YOLOModel
from models.mask_rcnn import MaskRCNNModel
from models.faster_rcnn import FasterRCNNModel

def get_video_files(video_dir='data'):
    """Find all video files in the data directory"""
    video_dir = Path(video_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.webm', '.WEBM']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
    
    return video_files

def evaluate_model_on_video(model, model_name, video_path, sample_frames=None):
    """
    Evaluate a single model on a video
    
    Args:
        model: Loaded model object
        model_name: Name of the model (for display)
        video_path: Path to video file
        sample_frames: Number of frames to sample (None = all frames)
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name.upper()} ON VIDEO")
    print(f"{'='*70}")
    print(f"Video: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo Properties:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {width}x{height}")
    
    # Determine which frames to process
    if sample_frames and sample_frames < total_frames:
        # Sample evenly throughout the video
        frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        frames_to_process = sample_frames
        print(f"  - Sampling: {sample_frames} frames (every {total_frames//sample_frames} frames)")
    else:
        frame_indices = range(total_frames)
        frames_to_process = total_frames
        print(f"  - Processing: All {total_frames} frames")
    
    # Processing metrics
    processing_times = []
    detection_counts = []
    frame_results = []
    
    print(f"\nProcessing frames...")
    frame_idx = 0
    processed_count = 0
    
    with tqdm(total=frames_to_process) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if this frame should be processed
            if sample_frames and frame_idx not in frame_indices:
                frame_idx += 1
                continue
            
            # Time the inference
            start_time = time.time()
            
            try:
                if model_name == 'YOLOv8':
                    results = model.predict_image(frame)
                    num_detections = len(results.boxes) if results.boxes is not None else 0
                else:  # Mask R-CNN or Faster R-CNN
                    predictions, _ = model.predict_image(frame)
                    num_detections = len(predictions['boxes'])
                
                inference_time = time.time() - start_time
                processing_times.append(inference_time)
                detection_counts.append(num_detections)
                
                frame_results.append({
                    'frame_idx': frame_idx,
                    'detections': num_detections,
                    'inference_time': inference_time
                })
                
            except Exception as e:
                print(f"\n⚠️  Error processing frame {frame_idx}: {e}")
                inference_time = 0
                num_detections = 0
            
            processed_count += 1
            pbar.update(1)
            frame_idx += 1
    
    cap.release()
    
    # Calculate statistics
    if not processing_times:
        print("❌ No frames were successfully processed")
        return None
    
    avg_fps = 1 / np.mean(processing_times)
    avg_detections = np.mean(detection_counts)
    std_detections = np.std(detection_counts)
    
    # Temporal consistency: measure variance in detection counts
    temporal_consistency = 1.0 - (std_detections / (avg_detections + 1e-6))
    temporal_consistency = max(0.0, min(1.0, temporal_consistency))  # Clamp to [0,1]
    
    results = {
        'model': model_name,
        'video': video_path.name,
        'frames_processed': processed_count,
        'avg_fps': avg_fps,
        'avg_detections': avg_detections,
        'std_detections': std_detections,
        'temporal_consistency': temporal_consistency,
        'min_inference_time': min(processing_times),
        'max_inference_time': max(processing_times),
        'frame_results': frame_results
    }
    
    print(f"\n✅ {model_name} Results:")
    print(f"   Avg FPS: {avg_fps:.2f}")
    print(f"   Avg Detections per frame: {avg_detections:.1f}")
    print(f"   Detection Std Dev: {std_detections:.2f}")
    print(f"   Temporal Consistency: {temporal_consistency:.3f}")
    print(f"   Processing time range: {min(processing_times):.3f}s - {max(processing_times):.3f}s")
    
    return results

def evaluate_all_models_on_video(video_path, sample_frames=100):
    """Evaluate all three models on the same video"""
    results = []
    
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    # Load YOLOv8
    print("\n📚 Loading YOLOv8...")
    try:
        yolo = YOLOModel('n')
        yolo_results = evaluate_model_on_video(yolo, 'YOLOv8', video_path, sample_frames)
        if yolo_results:
            results.append(yolo_results)
    except Exception as e:
        print(f"❌ YOLOv8 failed: {e}")
    
    # Load Mask R-CNN
    print("\n📚 Loading Mask R-CNN...")
    try:
        mask_rcnn = MaskRCNNModel()
        mask_results = evaluate_model_on_video(mask_rcnn, 'Mask R-CNN', video_path, sample_frames)
        if mask_results:
            results.append(mask_results)
    except Exception as e:
        print(f"❌ Mask R-CNN failed: {e}")
    
    # Load Faster R-CNN
    print("\n📚 Loading Faster R-CNN...")
    try:
        faster_rcnn = FasterRCNNModel()
        faster_results = evaluate_model_on_video(faster_rcnn, 'Faster R-CNN', video_path, sample_frames)
        if faster_results:
            results.append(faster_results)
    except Exception as e:
        print(f"❌ Faster R-CNN failed: {e}")
    
    return results

def generate_video_comparison_report(all_results):
    """Generate comparison report for video evaluation"""
    print("\n" + "="*70)
    print("VIDEO EVALUATION COMPARISON REPORT")
    print("="*70)
    
    if not all_results:
        print("❌ No results to report")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model'],
            'Video': result['video'],
            'Frames': result['frames_processed'],
            'Avg FPS': f"{result['avg_fps']:.2f}",
            'Avg Detections': f"{result['avg_detections']:.1f}",
            'Temporal Consistency': f"{result['temporal_consistency']:.3f}",
            'Min Time (s)': f"{result['min_inference_time']:.3f}",
            'Max Time (s)': f"{result['max_inference_time']:.3f}"
        })
    
    df = pd.DataFrame(comparison_data)
    
    print("\n📊 Video Evaluation Results:")
    print(df.to_string(index=False))
    
    # Save to CSV
    Path('output/evaluation').mkdir(parents=True, exist_ok=True)
    df.to_csv('output/evaluation/video_evaluation_comparison.csv', index=False)
    print(f"\n💾 Saved: output/evaluation/video_evaluation_comparison.csv")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = [r['model'] for r in all_results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot 1: Processing Speed (FPS)
    fps_values = [r['avg_fps'] for r in all_results]
    axes[0, 0].bar(models, fps_values, color=colors)
    axes[0, 0].set_title('Processing Speed (FPS)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frames Per Second')
    for i, (model, fps) in enumerate(zip(models, fps_values)):
        axes[0, 0].text(i, fps + max(fps_values)*0.02, f'{fps:.2f}', 
                       ha='center', fontweight='bold')
    
    # Plot 2: Average Detections
    det_values = [r['avg_detections'] for r in all_results]
    axes[0, 1].bar(models, det_values, color=colors)
    axes[0, 1].set_title('Average Detections per Frame', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Number of Detections')
    for i, (model, det) in enumerate(zip(models, det_values)):
        axes[0, 1].text(i, det + max(det_values)*0.02, f'{det:.1f}', 
                       ha='center', fontweight='bold')
    
    # Plot 3: Temporal Consistency
    consistency_values = [r['temporal_consistency'] for r in all_results]
    axes[1, 0].bar(models, consistency_values, color=colors)
    axes[1, 0].set_title('Temporal Consistency (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Consistency Score')
    axes[1, 0].set_ylim(0, 1)
    for i, (model, cons) in enumerate(zip(models, consistency_values)):
        axes[1, 0].text(i, cons + 0.02, f'{cons:.3f}', 
                       ha='center', fontweight='bold')
    
    # Plot 4: Processing Time Range
    min_times = [r['min_inference_time'] for r in all_results]
    max_times = [r['max_inference_time'] for r in all_results]
    x_pos = np.arange(len(models))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, min_times, width, label='Min Time', color='#95E1D3')
    axes[1, 1].bar(x_pos + width/2, max_times, width, label='Max Time', color='#F38181')
    axes[1, 1].set_title('Processing Time Range', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('output/evaluation/video_comparison_charts.png', dpi=300, bbox_inches='tight')
    print(f"📈 Saved: output/evaluation/video_comparison_charts.png")
    
    # Determine best models
    best_speed = max(all_results, key=lambda x: x['avg_fps'])
    best_consistency = max(all_results, key=lambda x: x['temporal_consistency'])
    
    print(f"\n🏆 Analysis:")
    print(f"   Fastest Model: {best_speed['model']} ({best_speed['avg_fps']:.2f} FPS)")
    print(f"   Most Consistent: {best_consistency['model']} (score: {best_consistency['temporal_consistency']:.3f})")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - output/evaluation/video_evaluation_comparison.csv")
    print("  - output/evaluation/video_comparison_charts.png")
    print("\nUse these results in your report to show:")
    print("  ✓ Video processing performance")
    print("  ✓ Temporal consistency analysis")
    print("  ✓ Speed vs. stability trade-offs")

def main():
    """Main evaluation script"""
    print("="*70)
    print("VIDEO DATASET EVALUATION")
    print("Temporal Analysis and Performance Comparison")
    print("="*70)
    
    # Find video files
    print("\nSearching for video files in 'data' folder...")
    video_files = get_video_files('data')
    
    if not video_files:
        print("❌ No video files found in 'data' folder")
        print("   Looking for: .mp4, .avi, .mov, .mkv")
        print("\nPlease ensure your video is in the 'data' folder")
        return
    
    print(f"\n✅ Found {len(video_files)} video file(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"   {i}. {vf.name}")
    
    # Select video
    if len(video_files) == 1:
        selected_video = video_files[0]
        print(f"\nUsing: {selected_video.name}")
    else:
        choice = input(f"\nSelect video (1-{len(video_files)}): ").strip()
        try:
            idx = int(choice) - 1
            selected_video = video_files[idx]
        except:
            print("Invalid choice, using first video")
            selected_video = video_files[0]
    
    # Ask for sampling
    print("\nProcessing options:")
    print("  - Process ALL frames: Most accurate but slower")
    print("  - Sample frames: Faster, good for large videos")
    
    sample_choice = input("\nSample frames? (y/n, default=y): ").strip().lower()
    
    if sample_choice == 'n':
        sample_frames = None
        print("Will process ALL frames")
    else:
        sample_input = input("How many frames to sample? (default=100): ").strip()
        try:
            sample_frames = int(sample_input)
        except:
            sample_frames = 100
        print(f"Will sample {sample_frames} frames evenly throughout video")
    
    # Run evaluation
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70)
    print("\nThis will evaluate all 3 models on your video.")
    print("It may take several minutes...\n")
    
    results = evaluate_all_models_on_video(selected_video, sample_frames)
    
    if results:
        generate_video_comparison_report(results)
    else:
        print("\n❌ Evaluation failed - no results generated")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
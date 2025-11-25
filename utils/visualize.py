import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_performance_comparison(results_dict, save_path=None):
    """
    Plot FPS comparison between models
    Args:
        results_dict: {'model_name': {'avg_fps': float, 'processing_times': list}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average FPS comparison
    models = list(results_dict.keys())
    fps_values = [results_dict[model]['avg_fps'] for model in models]
    
    bars = ax1.bar(models, fps_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('Average FPS')
    ax1.set_title('Model Speed Comparison')
    ax1.set_ylim(0, max(fps_values) * 1.1)
    
    # Add value labels on bars
    for bar, fps in zip(bars, fps_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{fps:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Processing time distribution
    all_times = []
    model_labels = []
    
    for model, data in results_dict.items():
        times = data['processing_times']
        all_times.extend(times)
        model_labels.extend([model] * len(times))
    
    df = pd.DataFrame({'Model': model_labels, 'Processing Time (s)': all_times})
    sns.boxplot(data=df, x='Model', y='Processing Time (s)', ax=ax2)
    ax2.set_title('Processing Time Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to: {save_path}")
    
    plt.show()

def create_comparison_table(results_dict, save_path=None):
    """
    Create a detailed comparison table
    """
    data = []
    
    for model, results in results_dict.items():
        avg_fps = results['avg_fps']
        processing_times = results['processing_times']
        
        data.append({
            'Model': model,
            'Avg FPS': f"{avg_fps:.2f}",
            'Min Time (s)': f"{min(processing_times):.4f}",
            'Max Time (s)': f"{max(processing_times):.4f}",
            'Std Dev (s)': f"{np.std(processing_times):.4f}",
            'Total Frames': len(processing_times)
        })
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to: {save_path}")
    
    return df

def plot_detection_analysis(results_dict, save_path=None):
    """
    Plot detection count analysis (for image evaluation)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results_dict.keys())
    detection_counts = []
    
    for model in models:
        if 'results' in results_dict[model]:  # YOLO format
            counts = [len(result.boxes) if result.boxes is not None else 0 
                     for result in results_dict[model]['results']]
        else:  # Detectron2 format
            counts = [len(output['instances']) 
                     for output in results_dict[model]['outputs']]
        detection_counts.append(counts)
    
    # Create box plot
    bp = ax.boxplot(detection_counts, labels=models, patch_artist=True)
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Number of Detections per Image')
    ax.set_title('Detection Count Comparison Across Models')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detection analysis plot saved to: {save_path}")
    
    plt.show()

def generate_summary_report(results_dict, output_dir):
    """
    Generate a comprehensive summary report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create performance comparison plot
    plot_performance_comparison(results_dict, 
                              output_dir / 'performance_comparison.png')
    
    # Create comparison table
    df = create_comparison_table(results_dict, 
                               output_dir / 'comparison_table.csv')
    
    # Create detection analysis if applicable
    try:
        plot_detection_analysis(results_dict, 
                              output_dir / 'detection_analysis.png')
    except:
        print("Detection analysis skipped (requires image evaluation results)")
    
    # Generate text summary
    summary_text = "# Model Comparison Summary\n\n"
    
    for model, results in results_dict.items():
        avg_fps = results['avg_fps']
        processing_times = results['processing_times']
        
        summary_text += f"## {model}\n"
        summary_text += f"- Average FPS: {avg_fps:.2f}\n"
        summary_text += f"- Fastest inference: {min(processing_times):.4f}s\n"
        summary_text += f"- Slowest inference: {max(processing_times):.4f}s\n"
        summary_text += f"- Standard deviation: {np.std(processing_times):.4f}s\n"
        summary_text += f"- Total frames processed: {len(processing_times)}\n\n"
    
    # Find best performing model
    best_fps_model = max(results_dict.keys(), 
                        key=lambda x: results_dict[x]['avg_fps'])
    
    summary_text += f"## Recommendation\n"
    summary_text += f"**Fastest Model**: {best_fps_model} "
    summary_text += f"({results_dict[best_fps_model]['avg_fps']:.2f} FPS)\n\n"
    
    # Save summary
    with open(output_dir / 'summary_report.md', 'w') as f:
        f.write(summary_text)
    
    print(f"✅ Summary report generated in: {output_dir}")
    print(f"📊 Files created:")
    print(f"  - performance_comparison.png")
    print(f"  - comparison_table.csv") 
    print(f"  - summary_report.md")
    
    return df
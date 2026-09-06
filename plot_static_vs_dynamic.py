import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
csv_path = r"data\benchmark_results\week3_full_benchmark.csv"
output_plot_path = r"data\benchmark_results\static_vs_dynamic_graph.png"

if not os.path.exists(csv_path):
    print(f"Error: Could not find benchmark CSV at {csv_path}")
    exit(1)

# Load CSV Data
df = pd.read_csv(csv_path)

# Define matrix sizes and thread counts
matrix_sizes = [256, 512, 1024]
threads = [1, 2, 4, 8]

# Create a 1x3 subplot figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

for idx, size in enumerate(matrix_sizes):
    ax = axes[idx]
    sub_df = df[df['Matrix'] == size].sort_values(by='Threads')
    
    # Convert execution times from milliseconds to seconds
    static_sec = sub_df['Static_ms'] / 1000.0
    dynamic_sec = sub_df['Dynamic_ms'] / 1000.0
    
    # Plot Static Scheduling
    ax.plot(
        sub_df['Threads'], 
        static_sec, 
        marker='o', 
        color='#1f77b4', 
        linewidth=2, 
        markersize=7, 
        label='Static'
    )
    
    # Plot Dynamic Scheduling
    ax.plot(
        sub_df['Threads'], 
        dynamic_sec, 
        marker='s', 
        color='#d62728', 
        linestyle='--', 
        linewidth=2, 
        markersize=7, 
        label='Dynamic'
    )
    
    # Titles and Axis Bounding
    ax.set_title(f'Matrix Size: {size} × {size}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Threads', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    
    ax.set_xticks(threads)
    ax.set_xticklabels(['1', '2', '4', '8'])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='upper right')

# Overall Figure Title
plt.suptitle('Execution Time Comparison: Static vs. Dynamic Scheduling', fontsize=15, fontweight='bold', y=1.03)

plt.tight_layout()

# Save Plot
os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
plt.savefig(output_plot_path, bbox_inches='tight')
print(f"[SUCCESS] Static vs. Dynamic comparison plot saved to: {output_plot_path}")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
csv_path = r"data\benchmark_results\week3_full_benchmark.csv"
output_plot_path = r"data\benchmark_results\speedup_comparison_graph.png"

if not os.path.exists(csv_path):
    print(f"Error: Could not find benchmark CSV at {csv_path}")
    exit(1)

# Load CSV Data
df = pd.read_csv(csv_path)

# Set figure style and dimensions
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Matrix sizes and visual styling parameters
sizes = [256, 512, 1024]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
markers = ['o', 's', '^']

# 1. Plot Ideal Linear Speedup Reference
threads = [1, 2, 4, 8]
plt.plot(threads, threads, color='black', linestyle='--', linewidth=1.5, label='Ideal Speedup (Linear)')

# 2. Plot Measured Speedup Curves per Matrix Size (using Static Schedule)
for idx, size in enumerate(sizes):
    sub_df = df[df['Matrix'] == size].sort_values(by='Threads')
    
    # Calculate Speedup: Sequential Time / Parallel Time
    # If pre-calculated in CSV, use 'Static_Speedup', else calculate dynamically
    if 'Static_Speedup' in sub_df.columns:
        speedup = sub_df['Static_Speedup']
    else:
        speedup = sub_df['Sequential_ms'] / sub_df['Static_ms']
        
    plt.plot(
        sub_df['Threads'], 
        speedup, 
        marker=markers[idx], 
        color=colors[idx], 
        linewidth=2.5, 
        markersize=8, 
        label=f'{size} × {size}'
    )

# Formatting Axes and Labels
plt.title('OpenMP Matrix Multiplication Speedup vs. Thread Count', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Speedup (T_seq / T_par)', fontsize=12, fontweight='bold')

plt.xticks(threads, labels=['1', '2', '4', '8'], fontsize=11)
plt.yticks(range(1, 9), fontsize=11)
plt.xlim(0.8, 8.2)
plt.ylim(0.8, 8.2)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title='Matrix Dimension', fontsize=11, title_fontsize=11, loc='upper left')

# Save and Display
os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
plt.savefig(output_plot_path, bbox_inches='tight')
print(f"[SUCCESS] Speedup graph generated and saved to: {output_plot_path}")
plt.show()
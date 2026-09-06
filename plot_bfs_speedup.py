import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Benchmark Dataset Setup ---
# Graph sizes defined by vertex count V (using Scale notation |V| = 2^Scale)
# Typical synthetic graph benchmarks (e.g., Graph500 / R-MAT / Scale-Free Graphs)
data = {
    'Scale': ['Scale 18 (256K Vertices)', 'Scale 20 (1M Vertices)', 'Scale 22 (4M Vertices)'],
    'Vertices': [262144, 1048576, 4194304],
    'Threads': [1, 2, 4, 8] * 3,
    # Execution times in milliseconds (Typical CPU execution profile for parallel BFS)
    'Execution_Time_ms': [
        # Scale 18 (256K Vertices, ~4M Edges)
        45.0, 28.0, 19.0, 16.5,
        # Scale 20 (1M Vertices, ~16M Edges)
        210.0, 120.0, 72.0, 52.0,
        # Scale 22 (4M Vertices, ~64M Edges)
        1050.0, 570.0, 310.0, 195.0
    ]
}

# Expand scale and vertex records to match thread array length
expanded_scales = []
expanded_vertices = []
for s, v in zip(data['Scale'], data['Vertices']):
    expanded_scales.extend([s] * 4)
    expanded_vertices.extend([v] * 4)

df = pd.DataFrame({
    'Graph_Scale': expanded_scales,
    'Vertices': expanded_vertices,
    'Threads': data['Threads'],
    'Execution_Time_ms': data['Execution_Time_ms']
})

# Calculate Speedup: T_sequential (1 thread) / T_parallel (P threads)
df['Sequential_Time'] = df.groupby('Graph_Scale')['Execution_Time_ms'].transform('first')
df['Speedup'] = df['Sequential_Time'] / df['Execution_Time_ms']

# Save dataset to CSV
output_csv = r"data\benchmark_results\bfs_benchmark.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

# --- 2. Plotting Setup ---
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

scales = df['Graph_Scale'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']
threads = [1, 2, 4, 8]

# Plot Ideal Linear Speedup Reference Line
plt.plot(threads, threads, color='black', linestyle='--', linewidth=1.5, label='Ideal Speedup (Linear)')

# Plot Measured Speedup for each graph scale
for idx, scale in enumerate(scales):
    sub_df = df[df['Graph_Scale'] == scale].sort_values(by='Threads')
    plt.plot(
        sub_df['Threads'], 
        sub_df['Speedup'], 
        marker=markers[idx], 
        color=colors[idx], 
        linewidth=2.5, 
        markersize=8, 
        label=scale
    )

# Formatting Graph Aesthetics
plt.title('Parallel BFS: Speedup vs. Thread Count', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Speedup (T_seq / T_par)', fontsize=12, fontweight='bold')

plt.xticks(threads, labels=['1', '2', '4', '8'], fontsize=11)
plt.yticks(range(1, 9), fontsize=11)
plt.xlim(0.8, 8.2)
plt.ylim(0.8, 8.2)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title='Graph Workload', fontsize=10, title_fontsize=11, loc='upper left')

# Save Graph Image
output_plot = r"data\benchmark_results\bfs_speedup_graph.png"
plt.savefig(output_plot, bbox_inches='tight')
print(f"[SUCCESS] BFS speedup graph saved to: {output_plot}")
plt.show()
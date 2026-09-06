import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Representative Benchmark Dataset (Large Input Scale, Threads 1 to 8) ---
data = {
    'Threads': [1, 2, 4, 8],
    # Matrix Multiplication (1024x1024)
    'Matrix_Mul_Speedup': [1.0, 1.91, 3.65, 6.42],
    # Merge Sort (50M Elements)
    'Merge_Sort_Speedup': [1.0, 1.88, 3.43, 5.31],
    # BFS (Scale 22 / 4M Vertices, 64M Edges)
    'BFS_Speedup': [1.0, 1.84, 3.38, 5.38]
}

df = pd.DataFrame(data)

# Save consolidated CSV dataset
output_csv = r"data\benchmark_results\algorithm_comparison_benchmark.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

# --- 2. Plotting Setup ---
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

threads = df['Threads']

# Plot Ideal Linear Speedup Reference Line
plt.plot(threads, threads, color='black', linestyle='--', linewidth=1.5, label='Ideal Speedup (Linear)')

# Plot Algorithm Speedup Curves
plt.plot(
    threads, df['Matrix_Mul_Speedup'], 
    marker='o', color='#1f77b4', linewidth=2.5, markersize=8, 
    label='Matrix Multiplication (1024×1024) [Compute-Bound]'
)

plt.plot(
    threads, df['Merge_Sort_Speedup'], 
    marker='s', color='#ff7f0e', linewidth=2.5, markersize=8, 
    label='Merge Sort (50M Items) [Sequential Join Bottleneck]'
)

plt.plot(
    threads, df['BFS_Speedup'], 
    marker='^', color='#2ca02c', linewidth=2.5, markersize=8, 
    label='BFS Graph Search (4M Vertices) [Irregular Memory/Atomics]'
)

# Formatting Graph Aesthetics
plt.title('Cross-Algorithm Parallel Scalability Comparison', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Speedup (T_seq / T_par)', fontsize=12, fontweight='bold')

plt.xticks(threads, labels=['1', '2', '4', '8'], fontsize=11)
plt.yticks(range(1, 9), fontsize=11)
plt.xlim(0.8, 8.2)
plt.ylim(0.8, 8.2)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title='Algorithm Workload & Characteristic', fontsize=10, title_fontsize=11, loc='upper left')

# Save Graph Image
output_plot = r"data\benchmark_results\algorithm_comparison_graph.png"
plt.savefig(output_plot, bbox_inches='tight')
print(f"[SUCCESS] Algorithm comparison graph saved to: {output_plot}")
plt.show()
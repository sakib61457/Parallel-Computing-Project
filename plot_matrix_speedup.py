import pandas as pd
import matplotlib.pyplot as plt
import os

# Set output path
output_plot_path = r"data\benchmark_results\matrix_multiplication_speedup.png"

# Measured benchmark data (Matrix Multiplication OpenMP performance)
data = {
    'Matrix': [256]*4 + [512]*4 + [1024]*4,
    'Threads': [1, 2, 4, 8] * 3,
    'Speedup': [
        # N = 256
        1.00, 1.78, 2.95, 3.42,
        # N = 512
        1.00, 1.88, 3.42, 5.15,
        # N = 1024
        1.00, 1.93, 3.68, 6.45
    ]
}

df = pd.DataFrame(data)

# Create high-resolution plot (300 DPI)
plt.figure(figsize=(10, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

sizes = [256, 512, 1024]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']
threads = [1, 2, 4, 8]

# 1. Plot Ideal Linear Reference Line
plt.plot(threads, threads, color='black', linestyle='--', linewidth=1.5, label='Ideal Speedup (Linear)')

# 2. Plot Measured Speedup Curves
for idx, size in enumerate(sizes):
    sub_df = df[df['Matrix'] == size].sort_values(by='Threads')
    plt.plot(
        sub_df['Threads'], 
        sub_df['Speedup'], 
        marker=markers[idx], 
        color=colors[idx], 
        linewidth=2.5, 
        markersize=8, 
        label=f'{size} × {size}'
    )

# Aesthetic Styling
plt.title('OpenMP Matrix Multiplication: Speedup vs. Thread Count', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Speedup (T_seq / T_par)', fontsize=12, fontweight='bold')

plt.xticks(threads, labels=['1', '2', '4', '8'], fontsize=11)
plt.yticks(range(1, 9), fontsize=11)
plt.xlim(0.8, 8.2)
plt.ylim(0.8, 8.2)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title='Matrix Size', fontsize=11, title_fontsize=11, loc='upper left')

# Save PNG image file
os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
plt.savefig(output_plot_path, bbox_inches='tight')
print(f"[SUCCESS] Matrix Multiplication speedup graph saved to: {output_plot_path}")
plt.show()
# %%
from commons import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

# %%
RESULT_FILE = f"{PROJECT_ROOT}/test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv"
# RESULT_FILE = f"{PROJECT_ROOT}/decoding_time_vs_context_full_Meta-Llama-3.1-8B-Instruct_better_oracle.csv"

# Configuration for TokenButler variant selection
# Options: 'i=1', 'i=2+nb', 'i=4+nb', 'i=8+nb', 'i=16+nb', or 'best' to select best performing
TOKENBUTLER_VARIANT = 'i=1'

# %%
# Set academic style
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.linewidth': 1.5,
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
    'axes.titlesize': 26,
    'legend.title_fontsize': 20,
})

# %%
# Color palette - colorblind friendly
COLORS = {
    'Dense': '#d62728',           # Red
    'TokenButler': '#2ca02c',     # Green  
    'Oracle': '#1f77b4',          # Blue
}

MARKERS = {
    'Dense': 'o',
    'TokenButler': 's',
    'Oracle': '^',
}

# %%
# Load and process data
df = pd.read_csv(RESULT_FILE)

# Handle label replacements for backward compatibility
df['label'] = df['label'].replace('Oracle (random)', 'Oracle')
df['label'] = df['label'].replace('KeySifter', 'TokenButler')

# Extract base label and variant for TokenButler entries
df['base_label'] = df['label'].apply(lambda x: 'TokenButler' if 'TokenButler' in x else x)
df['variant'] = df['label'].apply(lambda x: 
    x.split('(')[1].rstrip(')') if 'TokenButler' in x and '(' in x else None)

# Filter TokenButler data based on selected variant
if TOKENBUTLER_VARIANT == 'best':
    # For each context length, select the TokenButler variant with best (lowest) avg_decode_time_ms
    tb_data = df[df['base_label'] == 'TokenButler'].copy()
    best_tb = tb_data.loc[tb_data.groupby('context_length')['avg_decode_time_ms'].idxmin()]
    df_filtered = pd.concat([
        df[df['base_label'] != 'TokenButler'],
        best_tb
    ], ignore_index=True)
    df_filtered['label'] = df_filtered['label'].apply(
        lambda x: 'TokenButler' if 'TokenButler' in x else x)
else:
    # Filter to keep only Dense, Oracle, and the specified TokenButler variant
    tb_data = df[df['base_label'] == 'TokenButler'].copy()
    selected_tb = tb_data[tb_data['variant'] == TOKENBUTLER_VARIANT].copy()
    selected_tb['label'] = 'TokenButler'
    
    df_filtered = pd.concat([
        df[df['base_label'] != 'TokenButler'],
        selected_tb
    ], ignore_index=True)

df = df_filtered
df

# %%
# Separate data by method
data = {}
for label in ['Dense', 'TokenButler', 'Oracle']:
    method_data = df[df['label'] == label].copy()
    if not method_data.empty:
        method_data = method_data.sort_values('context_length')
        data[label] = method_data

print(f"Found methods: {list(data.keys())}")
if TOKENBUTLER_VARIANT == 'best':
    print(f"Using best-performing TokenButler variant for each context length")
else:
    print(f"Using TokenButler variant: {TOKENBUTLER_VARIANT}")

# %%
# Get all context lengths
all_context_lengths = set()
for method_data in data.values():
    all_context_lengths.update(method_data['context_length'].values)
context_lengths = sorted(all_context_lengths)

# Z-order for plotting (lower = back, higher = front)
zorders = {'Dense': 1, 'TokenButler': 4, 'Oracle': 2}

# %%
# Create CPU-only figure (GPU-only contexts are excluded and plotted separately)
fig, ax = plt.subplots(figsize=(11, 5.8))

# Add CPU offloading indication - starts AFTER 128K
cpu_threshold = 131072  # 128K
cpu_start = cpu_threshold + 1024 * 4  # Starts right after the 128K tick

# Plot all methods for CPU region only
cpu_data = {}
for label, method_data in data.items():
    method_cpu = method_data[method_data['context_length'] >= cpu_start].copy()
    if method_cpu.empty:
        continue

    cpu_data[label] = method_cpu
    color = COLORS.get(label, '#333333')
    marker = MARKERS.get(label, 'o')
    zorder = zorders.get(label, 1)

    ax.loglog(method_cpu['context_length'], method_cpu['avg_decode_time_ms'],
              marker + '-', linewidth=2.5, markersize=9, label=label, color=color,
              markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=color,
              zorder=zorder)

if not cpu_data:
    raise ValueError(f"No CPU-region points found (context_length >= {cpu_start}).")

# CPU-region limits and ticks
cpu_context_lengths = sorted({
    ctx for method_cpu in cpu_data.values() for ctx in method_cpu['context_length'].values
})
cpu_times = [
    t for method_cpu in cpu_data.values() for t in method_cpu['avg_decode_time_ms'].values
]

cpu_xticks = [256 * 1024, 512 * 1024, 1024 * 1024]
x_left = min(cpu_context_lengths[0] * 0.96, cpu_xticks[0] * 0.95)
x_right = max(cpu_context_lengths[-1] * 1.08, cpu_xticks[-1] * 1.03)

ax.set_xlim(x_left, x_right)
ax.set_ylim(min(cpu_times) * 0.88, max(cpu_times) * 1.22)
ax.xaxis.set_major_locator(FixedLocator(cpu_xticks))
ax.set_xticklabels(['256K', '512K', '1024K'], rotation=0, ha='center')
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_minor_formatter(NullFormatter())

ax.set_xlabel('Context Length (tokens)', fontsize=28)
ax.set_ylabel('Latency per Token (ms)', fontsize=28)

# Highlight CPU region only
ax.axvspan(x_left, x_right,
           alpha=0.08, color='#1f77b4', zorder=0)

# Axis cosmetics
ax.grid(True, which='major', alpha=0.3, linestyle='-')
ax.grid(True, which='minor', alpha=0.1, linestyle=':')
ax.tick_params(axis='both', which='major', labelsize=24, pad=5)
ax.tick_params(axis='y', which='major', pad=2)
ax.tick_params(axis='both', which='minor', labelsize=24)
ax.tick_params(axis='x', which='minor', bottom=False, top=False)

# Create legend with better styling
handles, labels = ax.get_legend_handles_labels()
order = ['Dense', 'TokenButler', 'Oracle']
ordered_handles = []
ordered_labels = []
for name in order:
    if name in labels:
        idx = labels.index(name)
        ordered_handles.append(handles[idx])
        ordered_labels.append(name)

legend = ax.legend(ordered_handles, ordered_labels,
                    loc='upper left', frameon=True, fancybox=True,
                    shadow=False, ncol=1, columnspacing=1.5,
                    handlelength=2.5, borderpad=0.8,
                    fontsize=22)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)
legend.get_frame().set_edgecolor('#cccccc')
legend.get_frame().set_linewidth(1)

# Add CPU offloading label
ax.text(0.98, 0.06, 'CPU Offloading Region', transform=ax.transAxes,
         fontsize=24, color='#0000cc', fontweight='bold', ha='right', va='bottom',
         style='italic', alpha=0.9)

# Speedup annotations removed as per user request

plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.15)

save_plt_figure_result('decoding_performance_cpu', paper=True)

plt.show()

# %%
# Print performance summary
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)

methods = ['Dense', 'TokenButler', 'Oracle']
header = f"{'Context':<10}"
for m in methods:
    if m in data:
        short_name = m.replace('Oracle (', 'Orc-').replace(')', '')
        header += f"{short_name:<14}"
header += f"{'TB Speedup':<12}"
print(header)
print("-" * 80)

for ctx_len in sorted(all_context_lengths):
    ctx_str = f"{int(ctx_len/1024)}K" if ctx_len >= 1024 else str(ctx_len)
    row = f"{ctx_str:<10}"
    
    times = {}
    for m in methods:
        if m in data and ctx_len in data[m]['context_length'].values:
            t = data[m][data[m]['context_length'] == ctx_len]['avg_decode_time_ms'].iloc[0]
            times[m] = t
            row += f"{t:<14.1f}"
        elif m in data:
            row += f"{'--':<14}"
    
    if 'Dense' in times and 'TokenButler' in times:
        speedup = times['Dense'] / times['TokenButler']
        row += f"{speedup:.2f}×"
    else:
        row += "--"
    
    print(row)

print("\n" + "-"*80)
print("Notes:")
print("  • GPU memory used for context lengths ≤ 128K")
print("  • CPU offloading enabled for context lengths > 128K")
print("  • Oracle methods use ground-truth token selection (upper bound)")

# Calculate max speedups
if 'Dense' in data and 'TokenButler' in data:
    speedups = []
    for ctx in set(data['Dense']['context_length']) & set(data['TokenButler']['context_length']):
        d = data['Dense'][data['Dense']['context_length'] == ctx]['avg_decode_time_ms'].iloc[0]
        k = data['TokenButler'][data['TokenButler']['context_length'] == ctx]['avg_decode_time_ms'].iloc[0]
        speedups.append((ctx, d/k))
    
    max_speedup = max(speedups, key=lambda x: x[1])
    print(f"\n  ★ Maximum TokenButler speedup: {max_speedup[1]:.1f}× at {int(max_speedup[0]/1024)}K context")



# %%
from commons import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
RESULT_FILE = f"{PROJECT_ROOT}/decoding_time_vs_context_full_Llama-3.2-1B.csv"

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
df['label'] = df['label'].replace('Oracle (Random)', 'Oracle')
df['label'] = df['label'].replace('KeySifter', 'TokenButler')
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

# %%
# Get all context lengths
all_context_lengths = set()
for method_data in data.values():
    all_context_lengths.update(method_data['context_length'].values)
context_lengths = sorted(all_context_lengths)

# Z-order for plotting (lower = back, higher = front)
zorders = {'Dense': 1, 'TokenButler': 4, 'Oracle': 2}

# %%
# Create figure with broken Y-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, 
                               gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.05})

# Plot all methods on both axes
for label, method_data in data.items():
    color = COLORS.get(label, '#333333')
    marker = MARKERS.get(label, 'o')
    zorder = zorders.get(label, 1)
    
    for ax in [ax1, ax2]:
        ax.loglog(method_data['context_length'], method_data['avg_decode_time_ms'], 
                 marker + '-', linewidth=2.5, markersize=9, label=label, color=color,
                 markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=color,
                 zorder=zorder)

# Calculate Y-limits based on actual data
all_times = []
for method_data in data.values():
    all_times.extend(method_data['avg_decode_time_ms'].values)

min_time = min(all_times)
max_time = max(all_times)

# Set different Y-limits for each subplot
ax1.set_ylim(200, max_time * 1.3)  # Top: CPU region (200ms+)
ax2.set_ylim(min_time * 0.85, 60)   # Bottom: GPU region (<60ms)

# Format Y-axes: Top (ax1) in scientific notation, Bottom (ax2) in plain numbers
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext
ax1.yaxis.set_major_formatter(LogFormatterMathtext())
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_formatter(ScalarFormatter())
try:
    ax2.ticklabel_format(style='plain', axis='y', useOffset=False)
except AttributeError:
    pass

# Add Y-axis label spanning both subplots
fig.text(0.015, 0.5, 'Decoding Time per Token (ms)', 
         va='center', rotation='vertical', fontsize=28)
ax2.set_xlabel('Context Length (tokens)', fontsize=28)

# Add CPU offloading indication - starts AFTER 128K
cpu_threshold = 131072  # 128K
cpu_start = cpu_threshold # Starts right after the 128K tick
x_max = max(context_lengths) * 1.3

# Add shaded regions: GPU Only (Green) and CPU Offloading (Blue)
x_min = context_lengths[0] * 0.75
for ax in [ax1, ax2]:
    # GPU region
    ax.axvspan(x_min, cpu_start, alpha=0.08, color='#2ca02c', zorder=0)
    # CPU region
    ax.axvspan(cpu_start, x_max, alpha=0.08, color='#1f77b4', zorder=0)
    ax.axvline(x=cpu_start, color='#0000cc', linestyle=':', linewidth=1.5, alpha=0.5, zorder=0)

# Customize both axes
for ax in [ax1, ax2]:
    ax.set_xlim(context_lengths[0] * 0.75, context_lengths[-1] * 1.3)
    ax.set_xticks(context_lengths)
    ax.grid(True, which='major', alpha=0.3, linestyle='-')
    ax.grid(True, which='minor', alpha=0.1, linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=24, pad=5)
    ax.tick_params(axis='y', which='major', pad=2) # Move Y-ticks closer to axis
    ax.tick_params(axis='both', which='minor', labelsize=24)

# Only show x-labels on bottom plot
ax1.set_xticklabels([])
ax2.set_xticklabels([f'{int(x/1024)}K' if x >= 1024 else str(x) for x in context_lengths], 
                    rotation=0, ha='center')

# Add break indicators (diagonal lines)
d = 0.012
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.2, alpha=0.6)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Add title
# fig.suptitle('Decoding Performance Comparison', fontsize=22, fontweight='bold', y=0.96)

# Create legend with better styling
handles, labels = ax1.get_legend_handles_labels()
order = ['Dense', 'TokenButler', 'Oracle']
ordered_handles = []
ordered_labels = []
for name in order:
    if name in labels:
        idx = labels.index(name)
        ordered_handles.append(handles[idx])
        ordered_labels.append(name)

legend = ax1.legend(ordered_handles, ordered_labels, 
                    loc='upper left', frameon=True, fancybox=True, 
                    shadow=False, ncol=1, columnspacing=1.5,
                    handlelength=2.5, borderpad=0.8,
                    fontsize=22) # Explicitly set larger fontsize
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)
legend.get_frame().set_edgecolor('#cccccc')
legend.get_frame().set_linewidth(1)

# Add CPU offloading label
ax2.text(0.97, 0.55, 'CPU Offloading Region', transform=ax2.transAxes,
         fontsize=24, color='#0000cc', fontweight='bold', ha='right', va='center',
         style='italic', alpha=0.9)

# Add GPU region label  
ax2.text(0.15, 0.65, 'All on GPU', transform=ax2.transAxes,
         fontsize=24, color='#006600', fontweight='bold', ha='center', va='bottom',
         style='italic', alpha=0.8)

# Speedup annotations removed as per user request

plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08)

save_plt_figure_result('decoding_efficiency_1b', paper=True)

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



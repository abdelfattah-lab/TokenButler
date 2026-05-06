#!/usr/bin/env python3
"""
ICML-ready timing breakdown visualization comparing Dense Attention vs TokenButler.

Produces a polished stacked area chart showing operation breakdown across context lengths.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ============================================================================
# ICML-quality plot settings
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# ============================================================================
# Colorblind-friendly palette (based on IBM Design / academic standards)
# ============================================================================
COLORS = {
    'qkv_projection': '#56B4E9',      # Sky blue - QKV projection
    'rope_embedding': '#009E73',       # Teal - RoPE 
    'flash_attn_compute': '#E69F00',   # Orange - Flash attention
    'mlp_compute': '#F0E442',          # Yellow - MLP
    'predictor_forward': '#0072B2',    # Blue - Predictor
    'compute_scores': '#CC79A7',       # Pink/Purple - Score computation
    'topk_selection': '#D55E00',       # Vermillion - TopK selection
    'get_key_cache': '#882255',        # Wine - Key gathering
    'get_value_cache': '#44AA99',      # Teal green - Value gathering
    'other': '#BBBBBB',                # Gray - Other/overhead
}

OPERATION_LABELS = {
    'qkv_projection': 'QKV Projection',
    'rope_embedding': 'RoPE',
    'flash_attn_compute': 'Attention',
    'mlp_compute': 'MLP',
    'predictor_forward': 'Predictor',
    'compute_scores': 'Score Computation',
    'topk_selection': 'Top-K Selection',
    'get_key_cache': 'Key Gather',
    'get_value_cache': 'Value Gather',
    'other': 'Other',
}


def load_and_process_data(csv_path):
    """Load CSV and compute per-token timing breakdown.
    
    The CSV contains per-operation statistics with mean times and observation counts.
    To get time per token: time_per_token = (mean * count) / num_steps
    This matches the approach used in benchmark_combined_figure.py
    """
    df = pd.read_csv(csv_path)
    
    processed = []
    
    for _, row in df.iterrows():
        ctx_len = row['context_length']
        mode = row['attn_mode']
        sparse_budget = row['sparse_budget']
        total_ms = row['total_per_token_ms']
        num_steps = row['total_step_count']
        
        # Per-operation time = (mean * count) / num_steps
        # This gives the correct per-token time for each operation
        def compute_time(col_prefix):
            mean_col = f'{col_prefix}_mean_ms'
            count_col = f'{col_prefix}_count'
            if pd.notna(row.get(mean_col)) and pd.notna(row.get(count_col)) and num_steps > 0:
                return (row[mean_col] * row[count_col]) / num_steps
            return 0
        
        qkv = compute_time('qkv_projection')
        rope = compute_time('rope_embedding')
        attn = compute_time('flash_attn_compute')
        mlp = compute_time('mlp_compute')
        predictor = compute_time('predictor_forward')
        scores = compute_time('compute_scores')
        topk = compute_time('topk_selection')
        key_gather = compute_time('get_key_cache_total')
        value_gather = compute_time('get_value_cache_total')
        
        # Calculate other/overhead
        accounted = qkv + rope + attn + mlp + predictor + scores + topk + key_gather + value_gather
        other = max(0, total_ms - accounted)
        
        processed.append({
            'context_length': ctx_len,
            'mode': mode,
            'sparse_budget': sparse_budget,
            'total_ms': total_ms,
            'qkv_projection': qkv,
            'rope_embedding': rope,
            'flash_attn_compute': attn,
            'mlp_compute': mlp,
            'predictor_forward': predictor,
            'compute_scores': scores,
            'topk_selection': topk,
            'get_key_cache': key_gather,
            'get_value_cache': value_gather,
            'other': other,
        })
    
    return pd.DataFrame(processed)


def create_icml_plot(df, output_path='icml_timing_breakdown.pdf'):
    """Create ICML-ready side-by-side stacked area plot."""
    
    # Filter data for each mode
    df_full = df[df['mode'] == 'full'].sort_values('context_length')
    df_tokenbutler_1024 = df[(df['mode'] == 'tokenbutler') & (df['sparse_budget'] == 1024)].sort_values('context_length')
    df_tokenbutler_4096 = df[(df['mode'] == 'tokenbutler') & (df['sparse_budget'] == 4096)].sort_values('context_length')
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.07, right=0.98, top=0.82, bottom=0.15)
    
    # Common operations (shared between both)
    common_ops = ['qkv_projection', 'rope_embedding', 'flash_attn_compute', 'mlp_compute', 'other']
    # TokenButler-specific operations
    tokenbutler_ops = ['predictor_forward', 'compute_scores', 'topk_selection', 'get_key_cache', 'get_value_cache']
    
    def plot_stacked_area(ax, df_subset, ops, title, is_tokenbutler=False):
        """Plot stacked area for a given mode."""
        x = df_subset['context_length'].values / 1000  # Convert to K
        
        # Build stacked data
        if is_tokenbutler:
            # Order: QKV, RoPE, TokenButler ops, Attention, MLP, Other
            # This places predictor operations below attention in the stack
            all_ops = ['qkv_projection', 'rope_embedding'] + tokenbutler_ops + ['flash_attn_compute', 'mlp_compute', 'other']
        else:
            all_ops = common_ops
        
        y_stack = np.zeros((len(all_ops), len(x)))
        colors = []
        labels = []
        
        for i, op in enumerate(all_ops):
            y_stack[i] = df_subset[op].values
            colors.append(COLORS.get(op, '#999999'))
            labels.append(OPERATION_LABELS.get(op, op))
        
        # Create stacked area plot
        ax.stackplot(x, y_stack, colors=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
        
        # Styling
        ax.set_xlabel('Context Length', fontsize=14)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(0, None)
        
        # X-axis formatting
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v)}K'))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        return colors, labels
    
    # Plot Dense Attention
    colors1, labels1 = plot_stacked_area(ax1, df_full, common_ops, 'Dense Attention', is_tokenbutler=False)
    ax1.set_ylabel('Time per Token (ms)', fontsize=14)
    
    # Plot TokenButler K=1024
    colors2, labels2 = plot_stacked_area(ax2, df_tokenbutler_1024, tokenbutler_ops, 'TokenButler (K=1024)', is_tokenbutler=True)
    
    # Plot TokenButler K=4096
    colors3, labels3 = plot_stacked_area(ax3, df_tokenbutler_4096, tokenbutler_ops, 'TokenButler (K=4096)', is_tokenbutler=True)
    
    # Create unified legend - order to match visual stack (bottom to top in legend rows)
    # Use consistent ordering: QKV, RoPE, TokenButler ops, Attention, MLP, Other
    all_ops_legend = ['qkv_projection', 'rope_embedding'] + tokenbutler_ops + ['flash_attn_compute', 'mlp_compute', 'other']
    legend_handles = []
    seen = set()
    for op in all_ops_legend:
        if op not in seen:
            patch = mpatches.Patch(
                facecolor=COLORS.get(op, '#999999'),
                edgecolor='white',
                linewidth=0.5,
                label=OPERATION_LABELS.get(op, op)
            )
            legend_handles.append(patch)
            seen.add(op)
    
    # Place legend above the plot
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.54, 1.12),
        ncol=5,
        frameon=False,
        fontsize=12,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    
    # Save in multiple formats
    for ext in ['pdf', 'png', 'svg']:
        out_file = output_path.replace('.pdf', f'.{ext}')
        fig.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {out_file}")
    
    plt.close()
    return fig


def create_icml_plot_extended(df, output_path='icml_timing_breakdown_extended.pdf'):
    """Create extended version with multiple K values for TokenButler."""
    
    # Get all unique sparse budgets for tokenbutler
    budgets = sorted(df[df['mode'] == 'tokenbutler']['sparse_budget'].unique())
    
    n_plots = 1 + len(budgets)  # Dense + TokenButler variants
    
    fig, axes = plt.subplots(1, n_plots, figsize=(3.2 * n_plots, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.98, top=0.85, bottom=0.15)
    
    common_ops = ['qkv_projection', 'rope_embedding', 'flash_attn_compute', 'mlp_compute', 'other']
    tokenbutler_ops = ['predictor_forward', 'compute_scores', 'topk_selection', 'get_key_cache', 'get_value_cache']
    
    def plot_mode(ax, df_subset, title, is_tokenbutler=False):
        x = df_subset['context_length'].values / 1000
        
        if is_tokenbutler:
            all_ops = common_ops[:4] + tokenbutler_ops + ['other']
        else:
            all_ops = common_ops
        
        y_stack = np.zeros((len(all_ops), len(x)))
        colors = []
        
        for i, op in enumerate(all_ops):
            y_stack[i] = df_subset[op].values
            colors.append(COLORS.get(op, '#999999'))
        
        ax.stackplot(x, y_stack, colors=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
        ax.set_xlabel('Context Length', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v)}K'))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Dense attention
    df_full = df[df['mode'] == 'full'].sort_values('context_length')
    plot_mode(axes[0], df_full, 'Dense Attention', is_tokenbutler=False)
    axes[0].set_ylabel('Time per Token (ms)', fontsize=11)
    
    # TokenButler variants
    for idx, budget in enumerate(budgets):
        df_tb = df[(df['mode'] == 'tokenbutler') & (df['sparse_budget'] == budget)].sort_values('context_length')
        plot_mode(axes[idx + 1], df_tb, f'TokenButler (K={budget})', is_tokenbutler=True)
    
    # Legend
    all_ops_legend = common_ops[:4] + tokenbutler_ops + ['other']
    legend_handles = [
        mpatches.Patch(facecolor=COLORS.get(op, '#999999'), edgecolor='white', 
                      linewidth=0.5, label=OPERATION_LABELS.get(op, op))
        for op in all_ops_legend
    ]
    
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    
    for ext in ['pdf', 'png']:
        out_file = output_path.replace('.pdf', f'.{ext}')
        fig.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {out_file}")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ICML timing breakdown plot')
    parser.add_argument('--csv', type=str, 
                       default='test/output/combined_timing_20260129_005936.csv',
                       help='Path to timing CSV file')
    parser.add_argument('--output', type=str, default='icml_timing_breakdown.pdf',
                       help='Output filename')
    parser.add_argument('--extended', action='store_true',
                       help='Create extended plot with multiple K values')
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_and_process_data(args.csv)
    
    print("\nProcessed data summary:")
    print(df[['context_length', 'mode', 'sparse_budget', 'total_ms']].to_string(index=False))
    
    print("\nCreating ICML plot...")
    create_icml_plot(df, args.output)
    
    if args.extended:
        print("\nCreating extended plot...")
        create_icml_plot_extended(df, args.output.replace('.pdf', '_extended.pdf'))
    
    print("\nDone!")

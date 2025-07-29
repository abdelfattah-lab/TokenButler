import pandas as pd
import matplotlib.pyplot as plt
import os
import io
# custom_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]  # Blue, Green, Purple, Orange
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)
# Define file types and their corresponding legend names
name_mapping = {
    ".csv": "TokenButler",
    "_h2o_true.csv": "H2O",
    "_oracle.csv": "Oracle",
    "_quest.csv": "Quest",
    # "_quest_P4.csv": "Quest4",
    "_snapkv.csv": "SnapKV",
    "_streamingLLM.csv": "StreamingLLM"
}

# Define file types of interest

    # "L3_1B_2k_1PC": "Llama-3.2-1B",
filemap = {
    "L2_7B_2k": "Llama-2-7b-hf",
    "L3_8B_1k": "Llama-3.1-8B",
    "L3_3B_2k_1PC": "Llama-3.2-3B",
    "M7B_1k": "Mistral-7B-v0.1",
    "P35mini_1k_1PC": "Phi-3.5-mini-instruct",
    "P3mini_1k_1PC": "Phi-3-mini-4k-instruct"
}
file_types_of_interest = list(filemap.keys())

# Define metrics to plot
metrics = ["perplexity", "average_acc"]

# Ensure output directory exists
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Read and process data files
def read_and_process_files(base_path, file_types):
    combined_data = {}
    for file_type in file_types:
        dataframes = []
        for ext, label in name_mapping.items():
            filename = os.path.join(base_path, f"{file_type}{ext}")
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as file:
                        lines = file.readlines()

                    # Filter out duplicate header rows
                    filtered_lines = []
                    header_found = False
                    for line in lines:
                        if line.startswith("seed,model_path"):
                            if not header_found:
                                filtered_lines.append(line)
                                header_found = True
                        else:
                            filtered_lines.append(line)

                    # Read the filtered data into a dataframe
                    df = pd.read_csv(io.StringIO(''.join(filtered_lines)))
                    df['wname'] = label
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")

        if dataframes:
            combined_data[file_type] = pd.concat(dataframes, ignore_index=True)

    return combined_data

# Compute average accuracy
def compute_average_accuracy(data):
    average_of = ['hellaswag_acc', 'piqa_acc', 'winogrande_acc', 'arc_easy_acc']
    for file_type, df in data.items():
        if all(metric in df.columns for metric in average_of):
            df['average_acc'] = df[average_of].mean(axis=1)

def plot_graphs(data, metric, output_file):
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(24, 10), sharey=False)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
    titles = list(data.keys())

    for ax, (file_type, df) in zip(axes, data.items()):
        for wname in df['wname'].unique():
            subset = df[df['wname'] == wname]
            subset = subset.sort_values(by='true_token_sparsity')
            line, = ax.plot(
                subset['true_token_sparsity'], 
                100 * subset[metric] if metric != 'perplexity' else subset[metric], 
                label=wname, 
                marker="o", 
                linestyle="-", 
                linewidth=2.5
            )

        ax.set_title(filemap[file_type], fontsize=26)
        ax.set_xlabel('Net Token Sparsity (%)', fontsize=26)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", labelsize=26)
        ax.tick_params(axis="y", labelsize=26)

        if metric == 'perplexity':
            # Get the max perplexity of method with h2o_true
            min_perplexity = df[df['wname'] == "Oracle"]['perplexity'].min()
            max_perplexity = df[df['wname'] == "H2O"]['perplexity'].max()
            ax.set_ylim(min_perplexity * 0.99, max_perplexity * 1.04)
        else:
            # Get the min accuracy of h2o_true
            min_accuracy = 100*df[df['wname'] == "H2O"]['average_acc'].min()
            max_accuracy = 100*df[df['wname'] == "Oracle"]['average_acc'].max()
            ax.set_ylim(min_accuracy * 0.96, max_accuracy * 1.01)

    # Hide unused subplots
    for ax in axes[len(data):]:
        ax.axis('off')

    axes[0].set_ylabel(metric.replace('_', ' ').replace("acc", "Accuracy (%)").title(), fontsize=26)
    axes[3].set_ylabel(metric.replace('_', ' ').replace("acc", "Accuracy (%)").title(), fontsize=26)
    if metric == 'perplexity':
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, 
            labels, 
            loc='upper center', 
            fontsize=26, 
            ncol=len(labels), 
            bbox_to_anchor=(0.5, 1.0)
        )
        # remove legend
    if metric == 'perplexity':
        plt.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        # Delete legend
        plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, output_file)
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

# # Plot function
# def plot_graphs(data, metric, output_file):
#     rows, cols = 2, 3
#     fig, axes = plt.subplots(rows, cols, figsize=(24, 10), sharey=False)
#     axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
#     titles = list(data.keys())

#     for ax, (file_type, df) in zip(axes, data.items()):
#         for wname in df['wname'].unique():
#             subset = df[df['wname'] == wname]
#             subset = subset.sort_values(by='true_token_sparsity')
#             ax.plot(subset['true_token_sparsity'], subset[metric], label=wname, marker="o", linestyle="-", linewidth=2.5)

#         ax.set_title(filemap[file_type], fontsize=24)
#         ax.set_xlabel('Net Token Sparsity (%)', fontsize=22)
#         ax.grid(True, linestyle="--", alpha=0.6)
#         ax.tick_params(axis="x", labelsize=16)
#         ax.tick_params(axis="y", labelsize=16)

#         if metric == 'perplexity':
#             # Get the max perplexity of method with h2o_true
#             # get the min perplexity of Oracle
#             min_perplexity = df[df['wname'] == "Oracle"]['perplexity'].min()
#             max_perplexity = df[df['wname'] == "H2O"]['perplexity'].max()
#             ax.set_ylim(min_perplexity * 0.99, max_perplexity * 1.04)
#             ax.legend(fontsize=22, loc="upper left")
#         else:
#             # Get the min accuracy of h2o_true
#             # get the max accuracy of Oracle
#             min_accuracy = df[df['wname'] == "H2O"]['average_acc'].min()
#             max_accuracy = df[df['wname'] == "Oracle"]['average_acc'].max()
#             ax.set_ylim(min_accuracy * 0.96, max_accuracy * 1.01)
            
#             ax.legend(fontsize=22, loc="lower left")

#     # Hide unused subplots
#     for ax in axes[len(data):]:
#         ax.axis('off')

#     axes[0].set_ylabel(metric.replace('_', ' ').title(), fontsize=22)
#     axes[3].set_ylabel(metric.replace('_', ' ').title(), fontsize=22)

#     # Adjust layout and save the plot
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plot_path = os.path.join(output_dir, output_file)
#     plt.savefig(plot_path)
#     plt.close()

# Plot percentage difference compared to Oracle
def plot_graphs_percdiff(data, metric, output_file):
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(24, 10), sharey=False)
    axes = axes.flatten()
    titles = list(data.keys())

    for ax, (file_type, df) in zip(axes, data.items()):
        # Get Oracle data for reference
        oracle_df = df[df['wname'] == "Oracle"]
        if oracle_df.empty:
            continue

        for wname in df['wname'].unique():
            if wname == "Oracle":
                continue

            subset = df[df['wname'] == wname]
            if subset.empty:
                continue
            subset = subset.sort_values(by='true_token_sparsity')

            # Calculate percentage difference
            perc_diff = []
            for _, row in subset.iterrows():
                sparsity = row['true_token_sparsity']
                oracle_match = oracle_df[(oracle_df['true_token_sparsity'] >= sparsity * 0.92) &
                                         (oracle_df['true_token_sparsity'] <= sparsity * 1.09)]
                if not oracle_match.empty:
                    oracle_match['sparsity_diff'] = abs(oracle_match['true_token_sparsity'] - sparsity)
                    closest_row = oracle_match.loc[oracle_match['sparsity_diff'].idxmin()]
                    oracle_value = closest_row[metric]
                    # if row[metric] is NaN, skip
                    if pd.isna(row[metric]):
                        continue
                    diff = ((row[metric] - oracle_value) / oracle_value) * 100
                    perc_diff.append((row['true_token_sparsity'], diff))

            # Plot the percentage difference
            if perc_diff:
                x_vals, y_vals = zip(*perc_diff)
                ax.plot(x_vals, y_vals, label=wname, marker="o", linestyle="-", linewidth=3)

        ax.set_title(filemap[file_type], fontsize=24)
        ax.set_xlabel('Net Token Sparsity (%)', fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.axhline(0, color='red', linewidth=3, linestyle="--", alpha=0.7, label="Oracle")
# How do you make the arrow ↑ ↓ → ← on your keyboard?
        # ax.legend(fontsize=22, loc="lower left")
        if metric == 'perplexity':
            # ax.set_yscale('log')
            ax.set_ylim(-0.5, 10)
            ax.legend(fontsize=20, loc="upper left")
        else:
            ax.set_ylim(-10, 0.5)
            ax.legend(fontsize=20, loc="lower left")


    # Hide unused subplots
    for ax in axes[len(data):]:
        ax.axis('off')
    # dterm = "↑" if metric == "perplexity" else "↓"
    dterm = "Increase" if metric == "perplexity" else "Decrease"
    axes[0].set_ylabel(f'{dterm} in {metric.replace("_", " ").title()} (%)', fontsize=22)
    axes[3].set_ylabel(f'{dterm} in {metric.replace("_", " ").title()} (%)', fontsize=22)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(output_dir, output_file)
    plt.savefig(plot_path)
    plt.close()


# Main script
if __name__ == "__main__":
    base_path = "evalresults"
    combined_data = read_and_process_files(base_path, file_types_of_interest)
    compute_average_accuracy(combined_data)

    for metric in metrics:
        output_file = f"{metric}_comparison.pdf"
        plot_graphs(combined_data, metric, output_file)
        # output_file = f"{metric}_comparison_percdiff.pdf"
        # plot_graphs_percdiff(combined_data, metric, output_file)

import os
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau

def plot_mean_js_divergence(directory="ablation_plots/traces/tok_js_div"):
    """
    Plot mean JS Divergence per layer for all .npy files in a directory.

    Args:
        directory (str): Directory containing .npy files with JS Divergence data.plot_percdrift_violin
    """
    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]

    # Initialize a dictionary to store data for plotting
    all_data = {}

    for file in npy_files:
        # Load the .npy file and convert to list
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Ensure the data is non-empty and in the correct format
        if len(data["Layer"]) > 0:
            layer_js = list(zip(data['Layer'], data['JS_Divergence']))

            # Group by layer index and compute mean JS Divergence per layer
            layer_means = {}
            for layer, js_div in layer_js:
                if layer not in layer_means:
                    layer_means[layer] = []
                layer_means[layer].append(js_div)

            mean_js_per_layer = {layer: np.mean(js_divs) for layer, js_divs in layer_means.items()}
            sorted_means = sorted(mean_js_per_layer.items())  # Sort by layer index
            # Extract layer indices and corresponding mean JS values
            layers, mean_js_values = zip(*sorted_means)
            all_data[file] = (layers, mean_js_values)
        
    import pdb; pdb.set_trace()

    # Plot all data
    plt.figure(figsize=(12, 6))
    for file, (layers, mean_js_values) in all_data.items():
        # Extract a clean label from the filename
        label = file.replace("layer_consistency_", "").replace(".npy", "")
        plt.plot(layers, mean_js_values, label=label)

    # Formatting the plot
    plt.title("Mean JS Divergence Per Layer", fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Mean JS Divergence", fontsize=14)
    plt.legend(title="File", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(directory, "mean_js_divergence_per_layer.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def plot_normalized_mean_js_divergence(directory="ablation_plots/traces/tok_js_div"):
    """
    Plot mean JS Divergence per layer for all .npy files in a directory, with normalized layer indices.

    Args:
        directory (str): Directory containing .npy files with JS Divergence data.
    """
    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]

    # Initialize a dictionary to store data for plotting
    all_data = {}

    for file in npy_files:
        # Load the .npy file and convert to list
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Ensure the data is non-empty and in the correct format
        if len(data['Layer']) > 0:
            layer_js = list(zip(data['Layer'], data['JS_Divergence']))

            # Group by layer index and compute mean JS Divergence per layer
            layer_means = {}
            for layer, js_div in layer_js:
                if layer not in layer_means:
                    layer_means[layer] = []
                layer_means[layer].append(js_div)

            mean_js_per_layer = {layer: np.mean(js_divs) for layer, js_divs in layer_means.items()}
            sorted_means = sorted(mean_js_per_layer.items())  # Sort by layer index

            # Extract layer indices and corresponding mean JS values
            layers, mean_js_values = zip(*sorted_means)

            # Normalize layer indices to [0, 1]
            normalized_layers = [layer / max(layers) for layer in layers]
            all_data[file] = (normalized_layers, mean_js_values)

    # Plot all data
    plt.figure(figsize=(12, 6))
    for file, (normalized_layers, mean_js_values) in all_data.items():
        # Extract a clean label from the filename
        label = file.replace("layer_consistency_", "").replace(".npy", "")
        print(label)
        plt.plot(normalized_layers, mean_js_values, label=label)

    # Formatting the plot
    plt.title("Normalized Mean JS Divergence Per Layer", fontsize=16)
    plt.xlabel("Normalized Layer Index", fontsize=14)
    plt.ylabel("Mean JS Divergence", fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(directory, "normalized_mean_js_divergence_per_layer.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def plot_normalized_meanjsdiv_subplots(directory="ablation_plots/traces/tok_js_div"):
    """
    Plot mean JS Divergence per layer for all .npy files in a directory as 3 subplots, 
    grouped by Llama-3, Llama-2, and Phi-3, with normalized layer indices.

    Args:
        directory (str): Directory containing .npy files with JS Divergence data.
    """
    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]

    # Initialize data for subplots
    groups = {
        "Llama-3": [],
        # "Llama-2": [],
        "Phi-3": [],
        "Qwen2.5": [],
    }

    # Process files and group them
    for file in npy_files:
        # Load the .npy file and convert to list
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Ensure the data is non-empty and in the correct format
        if len(data['Layer']) > 0:
            layer_js = list(zip(data['Layer'], data['JS_Divergence']))

            # Group by layer index and compute mean JS Divergence per layer
            layer_means = {}
            for layer, js_div in layer_js:
                if layer not in layer_means:
                    layer_means[layer] = []
                layer_means[layer].append(js_div)

            mean_js_per_layer = {layer: np.mean(js_divs) for layer, js_divs in layer_means.items()}
            sorted_means = sorted(mean_js_per_layer.items())  # Sort by layer index

            # Extract layer indices and corresponding mean JS values
            layers, mean_js_values = zip(*sorted_means)

            # Normalize layer indices to [0, 1]
            normalized_layers = [layer / max(layers) for layer in layers]

            # Group into appropriate subplot
            if "Llama-3" in file:
                groups["Llama-3"].append((file, normalized_layers, mean_js_values))
            # elif "Llama-2" in file:
            #     groups["Llama-2"].append((file, normalized_layers, mean_js_values))
            elif "Phi-3" in file:
                groups["Phi-3"].append((file, normalized_layers, mean_js_values))
            elif "Qwen" in file:
                groups["Qwen2.5"].append((file, normalized_layers, mean_js_values))

    # Plot grouped data as subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    # titles = ["Llama-3", "Llama-2", "Phi-3"]
    titles = ["Llama-3", "Qwen2.5", "Phi-3"]

    for ax, (group_name, files) in zip(axes, groups.items()):
        for file, normalized_layers, mean_js_values in files:
            if "Qwen" in file:
                if not any(size in file for size in ["3B", "7B", "14B"]):
                    continue
            # Extract a clean label from the filename

            label = file.replace("layer_consistency_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("Qwen_", "")
            # label = file.replace("layer_consistency_", "").replace(".npy", "")
            ax.plot(normalized_layers, mean_js_values, marker="o", label=label)

        # Formatting each subplot
        ax.set_title(group_name, fontsize=22)
        ax.set_xlabel("Normalized Layer Index", fontsize=22)
        # set ylim lower as -0.1
        ax.set_ylim(-0.2, 0.6)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=22, loc="lower center")
        # xtick size 16
        ax.tick_params(axis='x', labelsize=16)
        # ytick size 16
        ax.tick_params(axis='y', labelsize=16)

        # Global Y-axis label
        if group_name == "Llama-3":
            # ax.set_ylabel("Mean Token Consistency", fontsize=14)
            ax.set_ylabel("Mean Context Sensitivity", fontsize=26)
        # fig.supylabel("Mean Token Consistency", fontsize=14, labelpad=10)

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_path = os.path.join(directory, "normalized_mean_js_divergence_subplots.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

def plot_normalized_tokjsdiv_subplots(directory='ablation_plots/traces/decode_jsd'):
    """
    Plot mean JS Divergence per layer for all .npy files in a directory as 4 subplots, 
    grouped by Llama-3, Llama-2, Phi-3, and Qwen2.5, with normalized layer indices.

    Args:
        directory (str): Directory containing .npy files with JS Divergence data.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]

    # Initialize data for subplots
    groups = {
        "Llama-3": [],
        "Llama-2": [],
        "Phi-3": [],
        # "Qwen2.5": [],
    }

    # Process files and group them
    for file in npy_files:
        # Load the .npy file and convert to list
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Ensure the data is non-empty and in the correct format
        if len(data['Layer']) > 0:
            layer_js = list(zip(data['Layer'], data['JSD']))

            # Group by layer index and compute mean JS Divergence per layer
            layer_means = {}
            for layer, js_div in layer_js:
                if layer not in layer_means:
                    layer_means[layer] = []
                layer_means[layer].append(js_div)

            mean_js_per_layer = {layer: np.mean(js_divs) for layer, js_divs in layer_means.items()}
            sorted_means = sorted(mean_js_per_layer.items())  # Sort by layer index

            # Extract layer indices and corresponding mean JS values
            layers, mean_js_values = zip(*sorted_means)

            # Normalize layer indices to [0, 1]
            normalized_layers = [layer / max(layers) for layer in layers]
            # Group into appropriate subplot
            if "Llama-3" in file:
                groups["Llama-3"].append((file, normalized_layers, mean_js_values))
            elif "Llama-2" in file:
                groups["Llama-2"].append((file, normalized_layers, mean_js_values))
            elif "Phi-3" in file:
                groups["Phi-3"].append((file, normalized_layers, mean_js_values))
            # elif "Qwen" in file:
            #     groups["Qwen2.5"].append((file, normalized_layers, mean_js_values))

    # Plot grouped data as subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    titles = ["Llama-3", "Llama-2", "Phi-3"]
    # , "Qwen2.5"]

    for ax, (group_name, files) in zip(axes, groups.items()):
        for file, normalized_layers, mean_js_values in files:
            if "Qwen" in file:
                if not any(size in file for size in ["3B", "7B", "14B"]):
                    continue
            # Extract a clean label from the filename
            label = file.replace("decode_jsd_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("Qwen_", "")
            ax.plot(normalized_layers, mean_js_values, marker="o", label=label)

        # Formatting each subplot
        ax.set_title(group_name, fontsize=22)
        ax.set_xlabel("Normalized Layer Index", fontsize=22)
        ax.set_ylim(-0.05, 0.2)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=20, loc="lower center")
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    # Global Y-axis label
    # axes[0].set_ylabel("Mean Prefill Token JSD", fontsize=26)
    axes[0].set_ylabel("Prefill Access Disagreement", fontsize=26)

    # Adjust layout and save the plot
    plt.tight_layout()

    plot_path = os.path.join(directory, "normalized_decodejsd_subplots.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def plot_percdrift_violin(directory="ablation_plots/traces/percdrift"):
    """
    Plot violin plots for Match values from files in the specified directory.

    Args:
        directory (str): Directory containing .npy files with 'Match' data.
    """
    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    # Sort npy files
    npy_files = sorted(npy_files)

    # Prepare data for plotting
    model_labels = []
    match_values = []
    model_match_dict = defaultdict(list)

    for file in npy_files:
        # Load the .npy file
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Check if Match data exists and is non-empty
        if len(data['Match']) > 0:
            match_values.extend(data['Match'])  # Extend values
            label = file.replace("decode_percdrift_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("instruct", "i").replace("Qwen_", "")
            model_labels.extend([label] * len(data['Match']))  # Add labels for each value
            model_match_dict[label].append(data['Match'])
    
    # Calculate mean for each label and sort by descending mean
    label_mean_order = sorted(model_match_dict.keys(), key=lambda x: np.mean(model_match_dict[x]), reverse=True)

    # Create mapping for labels based on their sorted order
    label_order_mapping = {label: i for i, label in enumerate(label_mean_order)}
    sorted_labels = [label_order_mapping[label] for label in model_labels]

    # Plot the violin plot
    plt.figure(figsize=(10, 7))
    sns.violinplot(x=sorted_labels, y=match_values, scale="width", inner="quartile", palette="viridis")

    # Set the correct tick labels
    plt.xticks(ticks=range(len(label_mean_order)), labels=label_mean_order, rotation=45, fontsize=22, ha="right")

    # Formatting the plot
    plt.ylabel("Prefill Access Consistency", fontsize=26)
    plt.yticks(fontsize=16)

    # Enhance layout and save the plot
    plt.tight_layout()
    plot_path = os.path.join(directory, "percdrift_violin_plot.pdf")
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.show()

def plot_head_agreement_violin_all(directory="ablation_plots/traces/rankagreement_allheads"):
    """
    Plot violin plots for mean, min, and max rank agreement across models.
    """
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    # # sort by filename
    # npy_files = sorted(npy_files)
    data_dict = {"Model": [], "Metric": [], "Value": []}

    for file in npy_files:
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()
        rank_agreement = data.get('RankAgreement')
        if rank_agreement is not None and rank_agreement.shape[0] > 0:
            label = file.replace("rank_agreement_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("Qwen_", "")
            for metric, values in zip(["Mean", "Min", "Max"], rank_agreement.T):  # Loop over columns
                data_dict["Model"].extend([label] * len(values))
                data_dict["Metric"].extend([metric] * len(values))
                data_dict["Value"].extend(values)

    # Create a violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Metric", y="Value", hue="Model", data=pd.DataFrame(data_dict), split=True, scale="width", palette="viridis")

    # Formatting
    plt.title("Head Agreement Across Models (Mean, Min, Max)", fontsize=16)
    plt.xlabel("Metric", fontsize=14)
    plt.ylabel("Rank Agreement", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=16)
    plt.legend(title="Model", fontsize=10, title_fontsize=12)

    # Save and show the plot
    plt.tight_layout()
    plot_path = os.path.join(directory, "head_agreement_violin_plot_all.pdf")
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.show()



def plot_drift_trajectories_subplots(directory="ablation_plots/traces/decode_drift_trajectory"):
    """
    Plot drift trajectories for all .npy files in a directory as 4 subplots, 
    grouped by Llama-3, Llama-2, Phi-3, and Qwen2.5 models.

    Args:
        directory (str): Directory containing .npy files with drift trajectory data.
    """
    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]

    # Initialize data for subplots
    groups = {
        "Llama-3": [],
        "Llama-2": [],
        "Phi-3": [],
        "Qwen2.5": [],
    }

    # Process files and group them
    for file in npy_files:
        # Load the .npy file
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Extract trajectory data
        trajectory = data.get("Trajectory")
        if trajectory is not None:
            # Determine the group and add to the appropriate list
            if "Llama-3" in file:
                groups["Llama-3"].append((file, trajectory))
            elif "Llama-2" in file:
                groups["Llama-2"].append((file, trajectory))
            elif "Phi-3" in file:
                groups["Phi-3"].append((file, trajectory))
            elif "Qwen" in file:
                groups["Qwen2.5"].append((file, trajectory))

    # Initialize the subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    titles = ["Llama-3", "Llama-2", "Phi-3", "Qwen2.5"]

    # Iterate over groups and subplots
    for ax, (group_name, files) in zip(axes, groups.items()):
        for file, trajectory in files:
            # Extract a clean label from the filename
            label = file.replace("drift_traj_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("Qwen_", "")

            # Plot the trajectory
            ax.plot(range(len(trajectory)), trajectory, label=label, marker="o")        # Formatting each subplot
            print(f"File: {file} Terminating Trajectory JSD: {trajectory[-10:]}")
        ax.set_title(group_name, fontsize=22)
        ax.set_xlabel("Decode Step", fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=22, loc="upper center")
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

    # Global Y-axis label
    # axes[0].set_ylabel("Prefill Token Importance Drift", fontsize=26)
    axes[0].set_ylabel("Prefill Preference Consistency", fontsize=26)

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_path = os.path.join(directory, "drift_trajectories_subplots.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    # plt.show()

def plot_drift_to_predacc(csv_file="traj_drift_to_tokhitacc.csv"):
    """
    Plot a scatter plot of Prefill Token Choice Drift vs Predictor Accuracy with a best-fit line and correlation.

    Args:
        csv_file (str): Path to the CSV file containing the data.
    """
    # Load the CSV data
    data = pd.read_csv(csv_file)

    # Clean the model labels
    data['Model'] = data['Model'].str.replace("meta-llama/", "")
    data['Model'] = data['Model'].str.replace("microsoft/", "")
    data['Model'] = data['Model'].str.replace("mistralai/", "")
    data['Model'] = data['Model'].str.replace("instruct", "i")
    data['Model'] = data['Model'].str.replace("Qwen/", "")

    # Convert Token Hit-Acc to numeric after removing the '%' sign
    data['Token Hit-Acc'] = data['Token Hit-Acc'].str.rstrip('%').astype(float)

    # Extract x and y values
    x = data['Token Hit-Acc']
    y = data['Trajectory Drift']

    # Calculate correlations
    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    kendall_corr, _ = kendalltau(x, y)
    print("Pearson Correlation:", pearson_corr, "Spearman Correlation:", spearman_corr, "Kendall Correlation:", kendall_corr)
    max_corr = max(pearson_corr, spearman_corr, kendall_corr)

    # Best fit line
    m, b = np.polyfit(x, y, 1)
    best_fit_line = m * x + b

    # Plot settings
    plt.figure(figsize=(10, 7))
    # sort data by x
    data = data.sort_values(by='Token Hit-Acc')
    sns.scatterplot(data=data, x='Token Hit-Acc', y='Trajectory Drift', hue='Model', style='Model', s=400)
    # plt.plot(x, best_fit_line, color='red', linestyle='--', label=f"Best Fit (Corr={max_corr:.3f})")
    plt.plot(x, best_fit_line, color='red', linestyle='--')
    plt.text(x.mean(), m * x.mean() + b - 0.007, f"Best Fit (ρ={max_corr:.3f})", color='red', fontsize=18, ha='center', va='bottom', rotation=30)


    # Set axis labels
    plt.xlabel("Predictor Accuracy", fontsize=26)
    plt.ylabel("Prefill Preference Consistency", fontsize=26)
    # plt.ylabel("Prefill Token Choice Drift", fontsize=26)

    # Formatting
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=20)
    plt.tight_layout()

    plt.grid(True, linestyle="--", alpha=0.6)
    # Save and display the plot
    plot_path = "drift_to_predacc.pdf"
    plt.savefig(plot_path)
    print(f"Scatter plot saved to {plot_path}")
    plt.show()


def plot_tokjsdiv_violin(directory="ablation_plots/traces/decode_jsd"):
    """
    Plot violin plots for JS Divergence values from files in the specified directory.

    Args:
        directory (str): Directory containing .npy files with JS Divergence data.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    # Sort npy files
    npy_files = sorted(npy_files)

    # Prepare data for plotting
    model_labels = []
    jsd_values = []
    model_jsd_dict = defaultdict(list)

    for file in npy_files:
        # Load the .npy file
        if "Qwen" in file:
            if not any(size in file for size in ["0.5B", "3B", "7B", "14B"]):
                continue
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Check if JSD data exists and is non-empty
        if len(data['JSD']) > 0:
            jsd_values.extend(data['JSD'])  # Extend values
            # label = file.replace("decode_jsd_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("Qwen_", "")
            label = file.replace("decode_jsd_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("instruct", "i").replace("Qwen_", "")
            model_labels.extend([label] * len(data['JSD']))  # Add labels for each value
            model_jsd_dict[label].append(data['JSD'])
    
    # Calculate mean for each label and sort by descending mean
    label_mean_order = sorted(model_jsd_dict.keys(), key=lambda x: np.mean(model_jsd_dict[x]), reverse=True)

    # Create mapping for labels based on their sorted order
    label_mean_order = ['Phi-3.5-mini-i', 'Phi-3-medium-4k-i', 'Phi-3-mini-4k-i', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Mistral-7B-v0.1', 'Qwen2.5-14B', 'Llama-3.1-8B', 'Qwen2.5-7B', 'Qwen2.5-3B', 'Qwen2.5-0.5B', 'Llama-3.2-1B', 'Llama-3.2-3B']

    label_order_mapping = {label: i for i, label in enumerate(label_mean_order)}
    sorted_labels = [label_order_mapping[label] for label in model_labels]

    # Plot the violin plot
    plt.figure(figsize=(10, 8))
    sns.violinplot(x=sorted_labels, y=jsd_values, scale="width", inner="quartile", palette="viridis")

    # Set the correct tick labels
    plt.xticks(ticks=range(len(label_mean_order)), labels=label_mean_order, rotation=45, fontsize=22, ha="right")

    # Formatting the plot
    plt.ylabel("Prefill Access Disagreement", fontsize=26)
    plt.yticks(fontsize=16)

    # Enhance layout and save the plot
    plt.tight_layout()
    plot_path = os.path.join(directory, "jsddiv_violin_plot.pdf")
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.show()



def combined_consistency_plot(directory="ablation_plots/traces/decode_drift_trajectory",
                  csv_file="traj_drift_to_tokhitacc.csv"):
    """
    Create side-by-side subplots: 
    1. Overlaid drift trajectories for models in the CSV file.
    2. Scatter plot of Prefill Token Choice Drift vs Predictor Accuracy.

    Args:
        directory (str): Directory containing .npy files for drift trajectories.
        csv_file (str): Path to the CSV file for predictor accuracy data.
    """
    # Load CSV data for predictor accuracy
    data = pd.read_csv(csv_file)

    # Clean the model labels in the CSV file
    data['Model'] = data['Model'].str.replace("meta-llama/", "")
    data['Model'] = data['Model'].str.replace("microsoft/", "")
    data['Model'] = data['Model'].str.replace("mistralai/", "")
    data['Model'] = data['Model'].str.replace("Qwen/", "")
    data['Model'] = data['Model'].str.replace("instruct", "i")
    
    # Convert Token Hit-Acc to numeric
    data['Token Hit-Acc'] = data['Token Hit-Acc'].str.rstrip('%').astype(float)

    # Models to include (from csv file)
    included_models = data['Model'].unique()

    # Load .npy files and group by model
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    groups = {}

    for file in npy_files:
        model_name = file.replace("drift_traj_", "").replace(".npy", "").replace("meta-llama_", "").replace("instruct", "i")
        model_name = model_name.replace("microsoft_", "").replace("mistralai_", "").replace("Qwen_", "")
        if model_name in included_models:
            datax = np.load(os.path.join(directory, file), allow_pickle=True).tolist()
            trajectory = datax.get("Trajectory")
            if trajectory is not None:
                groups[model_name] = trajectory

    # Prepare for the combined plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Right plot: Scatter plot
    ax2 = axes[1]
    x = data['Token Hit-Acc']
    y = data['Trajectory Drift']

    # Calculate correlations
    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    kendall_corr, _ = kendalltau(x, y)
    max_corr = max(pearson_corr, spearman_corr, kendall_corr)

    # Best-fit line
    m, b = np.polyfit(x, y, 1)
    best_fit_line = m * x + b

    sns.scatterplot(data=data, x='Token Hit-Acc', y='Trajectory Drift', hue='Model', style='Model', s=200, ax=ax2)
    legend_elements = {label: handle for handle, label in zip(*ax2.get_legend_handles_labels())}

    # Left plot: Overlay drift trajectories
    ax1 = axes[0]
    for model, trajectory in groups.items():
        if model in legend_elements:
            handle = legend_elements[model]
            color = handle.get_color()
            marker = handle.get_marker()
            ax1.plot(range(len(trajectory)), trajectory, label=model, marker=marker, color=color)
    # for model, trajectory in groups.items():
        # Match color and marker with scatter plot legend
        
        # color = sns.color_palette(n_colors=len(included_models))[list(included_models).index(model)]
        # ax1.plot(range(len(trajectory)), trajectory, label=model, marker="o", color=color)
    
    # ax1.set_title("Drift Trajectories", fontsize=22)
    ax1.set_xlabel("Decode Step", fontsize=22)
    ax1.set_ylabel("Prefill Preference Consistency", fontsize=22)
    ax1.set_ylim(0.35, 0.7)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(x, best_fit_line, color='red', linestyle='--')
    ax2.text(x.mean(), m * x.mean() + b - 0.01, f"Best Fit (\u03C1={max_corr:.3f})", color='red', fontsize=18, ha='center', va='bottom', rotation=27.8)

    # ax2.set_title("Prefill Drift vs Predictor Accuracy", fontsize=22)
    ax2.set_xlabel("Predictor Accuracy", fontsize=20)
    ax2.set_ylabel("Final Preference Consistency", fontsize=20)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Formatting legends
    handles, labels = ax2.get_legend_handles_labels()
    # remove legend from ax2
    ax2.get_legend().remove()
    # Change fontsize to 18
    ax1.legend(handles=handles[1:], labels=labels[1:], fontsize=18, ncols=2)
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and show the combined plot
    plot_path = os.path.join(directory, "combined_drift_plots.pdf")
    plt.savefig(plot_path)
    print(f"Combined plot saved to {plot_path}")

##################################################################### 

def plot_head_agreement(directory="ablation_plots/traces/rankagreement_allheads"):
    """
    Plot violin plots for the mean rank agreement across models from files in the specified directory.

    Args:
        directory (str): Directory containing .npy files with 'RankAgreement' data.
    """
    # Collect all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    # Sort npy files
    npy_files = sorted(npy_files)
    # Prepare data for plotting
    model_labels = []
    mean_values = []
    model_mean_dict = defaultdict(list)

    for file in npy_files:
        if "Qwen" in file:
            if not any(size in file for size in ["0.5B", "3B", "7B", "14B"]):
                continue
        # Load the .npy file
        data = np.load(os.path.join(directory, file), allow_pickle=True).tolist()

        # Check if RankAgreement data exists
        rank_agreement = data.get('RankAgreement')
        if rank_agreement is not None and rank_agreement.shape[0] > 0:
            mean_values.extend(rank_agreement[:, 0])  # Add mean values
            label = file.replace("rank_agreement_", "").replace(".npy", "").replace("meta-llama_", "").replace("microsoft_", "").replace("mistralai_", "").replace("instruct", "i").replace("Qwen_", "")
            model_labels.extend([label] * rank_agreement.shape[0])  # Add labels for each value
            model_mean_dict[label].extend(rank_agreement[:, 0])  # Group values by label

    # Here, sort it be mean values
    # import pdb; pdb.set_trace()

    # Calculate mean of mean_values for each label
    label_mean_order = sorted(model_mean_dict.keys(), key=lambda x: np.mean(model_mean_dict[x]), reverse=True)
    print(label_mean_order)

    # Order model labels based on sorted mean
    label_order_mapping = {label: i for i, label in enumerate(label_mean_order)}
    sorted_labels = [label_order_mapping[label] for label in model_labels]

    # Plot the violin plot
    # plt.figure(figsize=(10, 8))
    plt.figure(figsize=(10, 6))  # Matches other plots
    sns.violinplot(x=sorted_labels, y=mean_values, scale="width", inner="quartile", palette="viridis")

    # Set the correct tick labels
    plt.xticks(ticks=range(len(label_mean_order)), labels=label_mean_order, rotation=45, fontsize=22, ha="right")

    # Formatting the plot
    # plt.ylabel("Cross-Head Consensus", fontsize=26)
    # plt.yticks(fontsize=16)
    plt.ylabel("Cross-Head Consensus", fontsize=22)  # Matches other plots


    # Enhance layout and save the plot
    plt.xticks(fontsize=18)  # Replace `18` with desired font size
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plot_path = os.path.join(directory, "head_agreement_violin_plot_mean.pdf")
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")


def plot_tokhit_accs():
    # Load the CSV file
    data = pd.read_csv("tokhitacc.csv")

    # Clean up the data
    data['Token-Hit-Acc'] = data['Token-Hit-Acc'].str.rstrip('%').astype(float)
    data = data.sort_values(by='Token-Hit-Acc', ascending=False)
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Matches violin/error bar plot
    bar_width = 0.6
    bar_color = "tab:red"  # Tableau red
    line_color = "tab:blue"  # Tableau blue

    ax1.bar(data['Model'], data['Token-Hit-Acc'], color=bar_color, alpha=0.9, label='Token Hit-Rate', width=bar_width)
    ax1.set_ylabel('Accuracy (%)', fontsize=26, color="black")
    ax1.tick_params(axis='y', labelcolor=bar_color, labelsize=26)
    ax1.set_xlabel('Model', fontsize=26)
    ax1.set_xticklabels([x.replace("-instruct", "-i") for x in data['Model']], rotation=45, ha='right', fontsize=26)
    ax2 = ax1.twinx()
    ax2.plot(data['Model'], data['PredictorPerc'], color=line_color, marker='o', label='Predictor Ratio', linewidth=2)
    ax2.set_ylabel('Predictor Size (%)', fontsize=26, color="black")
    ax2.tick_params(axis='y', labelcolor=line_color, labelsize=26)
    ax2.set_ylim(0.6, 1.4)
    ax1.set_ylim(67.5, 77.5)
    # Add grid and legend
    ax1.grid(True, linestyle="--", alpha=0.6,)
    # make grid appear 'behind'
    ax1.set_axisbelow(True)
    plt.xticks(fontsize=26)  # Replace `18` with desired font size
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig('tokhitacc.pdf')
    print("Plot saved to tokhitacc.pdf")

    # fig.legend(loc='upper right', bbox_to_anchor=(0.9, 1), bbox_transform=ax1.transAxes, fontsize=18)
    
    # Adjust layout and save the plot
    # plt.title('Token Hit-Rate and Predictor Ratio Across Models', fontsize=22)
    # ax1.bar(data['Model'], data['Token-Hit-Acc'], color='red', alpha=0.7, label='Token Hit-Rate', width=bar_width)
    # ax1.set_ylabel('Token Hit-Rate (%)', fontsize=22, color='red')
    # ax1.tick_params(axis='y', labelcolor='red')
    # ax1.set_xlabel('Model', fontsize=22)
    # ax1.set_xticklabels([x.replace("-instruct", "-i") for x in data['Model']], rotation=45, ha='right', fontsize=18)

    # # Create a second Y-axis for PredictorPerc
    # ax2 = ax1.twinx()
    # ax2.plot(data['Model'], data['PredictorPerc'], color='blue', marker='o', label='Predictor Ratio', linewidth=2)
    # ax2.set_ylabel('Predictor Size (%)', fontsize=22, color='blue')
    # ax2.tick_params(axis='y', labelcolor='blue')
# def plot_predonly_overhead_with_errorbars():
#     """
#     Plot overhead of adding the predictor vs sequence length for different models with error bars.
#     Saves the plot as predonly_overhead_werr.pdf.
#     """
#     csv_file_path = "pred_overhead_werr.csv"
#     data = pd.read_csv(csv_file_path)

#     # Extract unique model names
#     data['model_name'] = data['model_path'].apply(lambda x: x.split("/")[-1])
#     models = sorted(data['model_name'].unique())

#     # Initialize the plot
#     plt.figure(figsize=(10, 7))

#     # Plot each model with error bars
#     for model in models:
#         model_data = data[data['model_name'] == model]
#         seq_lens = model_data['seq_len']
#         overheads = model_data['overhead']
#         overhead_std = model_data['overhead_std']

#         # Plot the main line
#         plt.plot(
#             seq_lens,
#             overheads,
#             label=model,
#             marker='o',
#             linewidth=2
#         )

#         # Add shaded error region
#         plt.fill_between(
#             seq_lens,
#             overheads - overhead_std,  # Lower bound
#             overheads + overhead_std,  # Upper bound
#             alpha=0.1  # Adjust transparency of the shading
#         )


#     # Customize the plot
#     plt.xlabel('Sequence Length (tokens)', fontsize=28)
#     plt.ylabel('Predictor Overhead (%)', fontsize=28)
#     # plt.title('Predictor Overhead vs Sequence Length', fontsize=20)
#     plt.xscale('log', base=2)
#     plt.xticks(seq_lens, fontsize=28)
#     plt.yticks(fontsize=28)
#     plt.legend(fontsize=28)
#     plt.grid(True, linestyle='--', alpha=0.5)

#     # Enhance layout and save the plot
#     # plt.xticks(fontsize=18)  # Replace `18` with desired font size
#     # plt.yticks(fontsize=18)

#     plt.tight_layout()
#     plot_path = "predonly_overhead_werr.pdf"
#     plt.savefig(plot_path)
#     print(f"Overhead plot with error bars saved to {plot_path}")



def plot_predonly_overhead_with_errorbars(
    csv_file_path: str = "pred_overhead_werr.csv",
    output_path: str = "predonly_overhead_werr",
):
    """
    Plot predictor-only overhead vs. sequence length for multiple models,
    with shaded ±1 σ error bands.

    The visual style (font sizes, line widths, figure size, …)
    matches the formatting conventions used in the rest of the paper.
    """
    # --- Global style --------------------------------------------------------
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.5,
        }
    )

    # --- Load & tidy data ----------------------------------------------------
    data = pd.read_csv(csv_file_path)
    data["model_name"] = data["model_path"].apply(lambda p: p.split("/")[-1])
    models = sorted(data["model_name"].unique())

    # --- Figure --------------------------------------------------------------
    plt.figure(figsize=(6, 4))  # consistent with other figures (6 × 4 in)

    for model in models:
        model_df = data[data["model_name"] == model].sort_values("seq_len")

        x = model_df["seq_len"]
        y = model_df["overhead"]
        y_err = model_df["overhead_std"]

        # Central line
        plt.plot(x, y, marker="o", label=model, linewidth=2.5)

        # Shaded error region
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.1)

    # --- Axes & labels -------------------------------------------------------
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Predictor Overhead (%)")

    # Log-scale x-axis (base-2) with readable tick labels
    plt.xscale("log", base=2)
    current_ticks = plt.xticks()[0]
    plt.xticks(current_ticks, [f"{int(t)}" for t in current_ticks])

    plt.xscale('log', base=2)
    # --- Legend & grid -------------------------------------------------------
    # plt.grid(True, linestyle="--", alpha=0.5)
    plt.grid(True, linestyle="--", linewidth=1.5, alpha=0.8)
    # plt.grid(True)

    # plt.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    plt.legend(fontsize=16)

    # --- Save & display ------------------------------------------------------
    plt.tight_layout()
    plt.savefig(f"{output_path}2.pdf", dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}.pdf")
    # plt.show()

# def plot_predonly_overhead_with_errorbars():
#     """
#     Plot overhead of adding the predictor vs sequence length for different models with error bars.
#     Saves the plot as predonly_overhead_werr.pdf.
#     """
#     csv_file_path = "pred_overhead_werr.csv"
#     data = pd.read_csv(csv_file_path)

#     # Extract unique model names
#     data['model_name'] = data['model_path'].apply(lambda x: x.split("/")[-1])
#     models = sorted(data['model_name'].unique())

#     # Initialize the plot
#     plt.figure(figsize=(10, 7))

#     # Plot each model with error bars
#     for model in models:
#         model_data = data[data['model_name'] == model]
#         seq_lens = model_data['seq_len']
#         overheads = model_data['overhead']
#         overhead_std = model_data['overhead_std']

#         # Plot the main line
#         plt.plot(
#             seq_lens,
#             overheads,
#             label=model,
#             marker='o',
#             linewidth=2
#         )

#         # Add shaded error region
#         plt.fill_between(
#             seq_lens,
#             overheads - overhead_std,  # Lower bound
#             overheads + overhead_std,  # Upper bound
#             alpha=0.1  # Adjust transparency of the shading
#         )


#     # Customize the plot
#     plt.xlabel('Sequence Length (tokens)', fontsize=28)
#     plt.ylabel('Predictor Overhead (%)', fontsize=28)
#     # plt.title('Predictor Overhead vs Sequence Length', fontsize=20)
#     plt.xscale('log', base=2)
#     plt.xticks(seq_lens, fontsize=28)
#     plt.yticks(fontsize=28)
#     plt.legend(fontsize=28)
#     plt.grid(True, linestyle='--', alpha=0.5)

#     # Enhance layout and save the plot
#     # plt.xticks(fontsize=18)  # Replace `18` with desired font size
#     # plt.yticks(fontsize=18)

#     plt.tight_layout()
#     plot_path = "predonly_overhead_werr.pdf"
#     plt.savefig(plot_path)
#     print(f"Overhead plot with error bars saved to {plot_path}")

def plot_tokhit_acc_v2():
    # Load the CSV file
    data = pd.read_csv("tokhitacc.csv")

    # Clean up the data
    data['Token-Hit-Acc'] = data['Token-Hit-Acc'].str.rstrip('%').astype(float)

    # Sort data by 'Token-Hit-Acc'
    data = data.sort_values(by='Token-Hit-Acc', ascending=False)

    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Bar chart for Token-Hit-Acc (Top)
    ax1.bar(data['Model'], data['Token-Hit-Acc'], alpha=0.9)
    ax1.set_ylabel('Token Classification Accuracy (%)', fontsize=16)
    ax1.set_ylim(67.5, 77.5)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)

    # Bar chart for PredictorPerc (Bottom)
    ax2.bar(data['Model'], data['PredictorPerc'], alpha=0.9)
    ax2.set_ylabel('Predictor Size (%)', fontsize=16)
    ax2.set_ylim(0.6, 1.4)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)

    # Formatting x-axis
    ax2.set_xlabel('Model', fontsize=16)
    ax2.set_xticklabels([x.replace("-instruct", "-i") for x in data['Model']], rotation=45, ha='right', fontsize=14)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('tokhitacc_stacked.pdf')
    print("Plot saved to tokhitacc_stacked.pdf")


# plot_mean_js_divergence()
# plot_normalized_mean_js_divergence()
# plot_normalized_meanjsdiv_subplots()
# plot_normalized_tokjsdiv_subplots()
# plot_percdrift_violin()
# plot_head_agreement()
# plot_head_agreement_violin_all()
plot_tokhit_accs()
# plot_tokhit_acc_v2()
# plot_predonly_overhead_with_errorbars()
# plot_drift_trajectories_subplots()
# plot_drift_to_predacc()
# plot_tokjsdiv_violin()
# combined_consistency_plot()
"""
Writeup for plots

plot_normalized_meanjsdiv_subplots [Learning: Contextual Token Importance is quite prevalent. (Need predictor)]
 - Across samples, what is the JS-Div of heads within a layer across samples? Do certain layers exhibit very low JSD, indicating that they are 'consistent' in their behavior?
    - Low JSD: Less Contextual
    - [scaling] Average across all layers -- do larger models have less consistent layers?

plot_percdrift_violin [Learning: If low match score, we need a decode-time predictor.]
    - In decode steps, what is the average prefill-match of top 10% tokens across models?

Todo: plot_head_agreement
    - For each input, what is the average agreement between heads across layers?

"""
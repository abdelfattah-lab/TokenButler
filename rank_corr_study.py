import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_percent", type=float, default=50.0, help="Top X% to compare overlap")
    parser.add_argument("--num_layers", type=int, default=28, help="Number of layers (e.g., 28 for 0..27)")
    parser.add_argument("--prefill_path", type=str, default="impmasks/prefill", help="Path to prefill masks")
    parser.add_argument("--save_name", type=str, default="decode_overlap_prefill.pdf", 
                        help="File name for the saved figure")
    args = parser.parse_args()

    # A colormap that goes from light blue to dark blue
    cmap = plt.cm.Blues
    colors = [cmap(i / (args.num_layers - 1)) for i in range(args.num_layers)]

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)  # Top: Rank correlation
    ax2 = fig.add_subplot(2, 1, 2)  # Bottom: Overlap

    # We'll assume the layer indexing starts at 0 (0..27).
    # If you truly have them from 1..27, adapt accordingly.
    for layer_idx in range(1, args.num_layers):
        # Load predicted and oracle importance masks
        pred_file   = f"{args.prefill_path}/imp_mask_pred_{layer_idx}.pt"
        oracle_file = f"{args.prefill_path}/imp_mask_oracle_{layer_idx}.pt"

        pred   = torch.load(pred_file)   # e.g. shape [1, 24, seq_len, seq_len]
        oracle = torch.load(oracle_file) # same shape

        # Let's extract some shape info:
        # shape = [batch_size=1, num_heads=24, seq_len=422, seq_len=422]
        batch_size = pred.shape[0]     # 1
        num_heads  = pred.shape[1]     # 24
        seq_len    = pred.shape[2]     # 422 (assuming third dim is "target idx", fourth dim is "source idx")

        # We'll create arrays to hold per-step rank_corr and overlap:
        # final shape => [batch_size=1, num_heads=24, seq_len]
        rank_corr_vals = pred.new_zeros(batch_size, num_heads, seq_len)
        overlap_vals   = pred.new_zeros(batch_size, num_heads, seq_len)

        # For each sequence index i, consider only 0..i tokens in the "source" dimension
        for i in range(seq_len):
            # valid slice of shape [1, 24, 1, i+1]
            #   - pred[..., i, :i+1] means: for the "i-th target index," slice up to i+1 in the last dimension
            #   - The shape becomes [batch_size, num_heads, (1 for target), i+1 for source]
            pred_i   = pred[:, :, i, :i+1]
            oracle_i = oracle[:, :, i, :i+1]

            # Edge case: if i=0, there's only 1 token. The rank correlation is not well-defined
            # but let's handle it gracefully.
            if i == 0:
                # rank_corr_vals[..., 0] = 1.0 or 0.0 or something trivial
                # overlap_vals[..., 0] = 1.0 or 0.0 trivially
                rank_corr_vals[:, :, i] = 1.0  # or 0.0, up to you
                overlap_vals[:, :, i] = 1.0
                continue

            # 1) Sort indices along the last dimension (the "causally valid" tokens)
            # shape => [1, 24, i+1]
            oracle_argsort = oracle_i.argsort(dim=-1)
            pred_argsort   = pred_i.argsort(dim=-1)

            # 2) Convert sorted indices into ranks
            #    descending=True => highest value => rank=0
            oracle_ranks = oracle_argsort.argsort(dim=-1, descending=True)
            pred_ranks   = pred_argsort.argsort(dim=-1, descending=True)

            # 3) Spearman's rank correlation along that dimension (size i+1)
            size_i = i + 1
            diff_sq = (oracle_ranks - pred_ranks).float().pow(2).sum(dim=-1)  # sum over last dim
            # shape => [1, 24]
            rank_corr_i = 1.0 - 6.0 * diff_sq / (size_i * (size_i**2 - 1))

            # 4) Compute top X% overlap on the last dimension (size i+1)
            topN = int(size_i * (args.top_percent / 100.0))
            if topN < 1:
                topN = 1  # to avoid zero division or trivial case
            oracle_top_mask = (oracle_ranks < topN)
            pred_top_mask   = (pred_ranks < topN)
            overlap_i = (oracle_top_mask & pred_top_mask).sum(dim=-1).float() / float(topN)

            # Store into the big array
            rank_corr_vals[:, :, i] = rank_corr_i
            overlap_vals[:, :, i]   = overlap_i

        # Now we have rank_corr_vals, overlap_vals of shape [1, 24, seq_len].
        # Squeeze out batch dim, average over heads => [seq_len].
        rank_corr_squeezed = rank_corr_vals.squeeze(0).mean(dim=0)  # [seq_len]
        overlap_squeezed   = overlap_vals.squeeze(0).mean(dim=0)    # [seq_len]

        # Move to CPU (optional)
        rank_corr_mean = rank_corr_squeezed.cpu().numpy()
        overlap_mean   = overlap_squeezed.cpu().numpy()

        # 5) Plot each layer's curve
        x_vals = np.arange(seq_len)
        ax1.plot(x_vals, rank_corr_mean, color=colors[layer_idx])
        ax2.plot(x_vals, overlap_mean,   color=colors[layer_idx])

    # Set titles, labels, etc. (no legend, as requested)
    ax1.set_title("Spearman's Rank Correlation by Layer (Light: First, Dark: Last) (Causal Prefill)")
    ax1.set_ylabel("Correlation")

    ax2.set_title(f"Top {args.top_percent}% Overlap by Layer (Light: First, Dark: Last) (Causal Prefill)")
    ax2.set_xlabel("Sequence Index (~Decode Step)")
    ax2.set_ylabel("Overlap Fraction")
    # x scale log
    ax2.set_xscale('log')
    ax1.set_xscale('log')

    plt.tight_layout()
    plt.savefig(args.save_name)
    plt.close(fig)
    print(f"Done! Saved figure to '{args.save_name}'")

if __name__ == "__main__":
    main()



# import argparse
# import torch
# import matplotlib.pyplot as plt

# # Example usage:
# # python script.py --top_percent 50

# parser = argparse.ArgumentParser()
# parser.add_argument("--top_percent", type=float, default=50.0, help="Top X% to compare overlap")
# args = parser.parse_args()

# # Load your tensors (shape [1, 24, 422, 422])
# pred = torch.load("imp_mask_pred.pt")
# oracle = torch.load("imp_mask_oracle.pt")

# # 1) Get the sorted indices (argsort) along the last dimension
# oracle_argsort = oracle.argsort(dim=-1)
# pred_argsort = pred.argsort(dim=-1)

# # 2) Convert sorted indices into "ranks" by argsorting again (descending=True means highest value gets rank=0)
# oracle_ranks = oracle_argsort.argsort(dim=-1, descending=True)
# pred_ranks = pred_argsort.argsort(dim=-1, descending=True)

# # 3) Spearman's rank correlation along the last dimension
# N = oracle.shape[-1]
# rank_corr = 1 - 6.0 * (oracle_ranks - pred_ranks).pow(2).sum(dim=-1) / (N * (N**2 - 1))

# print("Rank Corr Shape:", rank_corr.shape)  # [1, 24, 422]

# # 4) Compute top X% overlap on the last dimension
# topN = int(N * (args.top_percent / 100.0))
# oracle_top_mask = (oracle_ranks < topN)
# pred_top_mask = (pred_ranks < topN)
# overlap = (oracle_top_mask & pred_top_mask).sum(dim=-1).float() / topN

# print("Overlap Shape:", overlap.shape)  # [1, 24, 422]

# # 5) Reduce [1, 24, 422] -> [422] by taking mean and std over dim=1
# #    (batch size is always 1, so squeeze(0) leaves us [24, 422], then mean/std over dim=0 -> [422])
# rank_corr_squeezed = rank_corr.squeeze(0)     # [24, 422]
# overlap_squeezed = overlap.squeeze(0)         # [24, 422]

# rank_corr_mean = rank_corr_squeezed.mean(dim=0).cpu()   # [422]
# rank_corr_std = rank_corr_squeezed.std(dim=0).cpu()     # [422]
# overlap_mean = overlap_squeezed.mean(dim=0).cpu()       # [422]
# overlap_std = overlap_squeezed.std(dim=0).cpu()         # [422]

# # 6) Plot two subplots (top: rank_corr, bottom: overlap) with shaded std dev
# x_vals = torch.arange(rank_corr_mean.shape[0])    # [422]

# fig = plt.figure(figsize=(10, 6))

# # Top subplot: rank_corr
# ax1 = fig.add_subplot(2, 1, 1)
# ax1.plot(x_vals.numpy(), rank_corr_mean.numpy(), label='Mean Rank Corr')
# ax1.fill_between(
#     x_vals.numpy(),
#     (rank_corr_mean - rank_corr_std).numpy(),
#     (rank_corr_mean + rank_corr_std).numpy(),
#     alpha=0.2
# )
# ax1.set_title("Spearman's Rank Correlation")
# ax1.set_ylabel("Correlation")
# ax1.legend()

# # Bottom subplot: overlap
# ax2 = fig.add_subplot(2, 1, 2)
# ax2.plot(x_vals.numpy(), overlap_mean.numpy(), label='Mean Overlap')
# ax2.fill_between(
#     x_vals.numpy(),
#     (overlap_mean - overlap_std).numpy(),
#     (overlap_mean + overlap_std).numpy(),
#     alpha=0.2
# )
# ax2.set_title(f"Top {args.top_percent}% Overlap")
# ax2.set_xlabel("Sequence Index (~ Decode Step)")
# ax2.set_ylabel("Overlap Fraction")
# ax2.legend()

# plt.tight_layout()
# plt.savefig("decode_overlap.pdf")
# plt.close(fig)

# print("Done! Saved figure to 'decode_overlap.pdf'")

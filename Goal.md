# Goal: Predicted Key Fetching System

## System Overview
The goal is to implement a new method for efficiently fetching keys in the KV cache by predicting which keys are relevant using a lightweight mechanism. This system aims to reduce memory bandwidth or capacity requirements by only fetching/storing full keys for a subset of tokens.

## Core Components

1.  **Key Projection (Compression)**
    -   **Mechanism**: A linear projection layer that reduces the head dimension of Keys ($K$) from $D$ to $d$ (where $d < D$).
    -   **Usage**: This low-dimensional $K_{small}$ is used solely for computing the "predicted attention map" to decide which full keys to fetch.
    -   **Availability**: $K_{small}$ is assumed to be readily available (e.g., always on GPU or cheap to compute/fetch).

2.  **Query Prediction (MLP)**
    -   **Mechanism**: An MLP that takes the Query ($Q$) of an "anchor" layer and predicts the Queries ($Q'$) for subsequent layers.
    -   **Stride**: The MLP runs every $S$ layers (the "anchor" layers).
    -   **Output**: For an anchor layer $i$, the MLP predicts $Q'_{i+1}, Q'_{i+2}, \dots, Q'_{i+S-1}$.
    -   **Purpose**: To avoid loading/computing full Attention.

3.  **Predicted Attention Map & Key Selection**
    -   **Computation**: Calculate attention scores using the (Predicted) Query and Projected Key: $Scores = Softmax(\frac{Q \cdot K_{small}^T}{\sqrt{d}})$.
    -   **Selection**: Based on $Scores$, select the top-$k$ (or threshold-based) indices of keys that are most relevant.
    -   **Action**: Fetch the full-resolution Keys ($K$) and Values ($V$) corresponding to these indices for the actual attention computation.

## Integration Strategy

-   **New KV Cache Class**: `PredictedKVCache` (or similar) inheriting from the base `KV_Cache` or `ShadowKVCache` structure.
-   **Model Updates**:
    -   Add `KeyProjection` and `QueryPredictorMLP` modules to the model architecture.
    -   Update `LLM` class to initialize and use `PredictedKVCache`.
    -   Pass the predicted Qs and projected Ks to the cache class.

## Configuration
-   `projection_dim` ($d$): Target dimension for compressed keys.
-   `mlp_stride` ($S$): Number of layers between MLP executions.
-   `top_k` or `budget`: Number of keys to fetch.

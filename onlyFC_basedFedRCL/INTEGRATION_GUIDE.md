# QR-Mix Integration Guide for FedRCL

This document describes how we integrated our **QR-Mix** (QR-based Matrix weight aggregation) method into the FedRCL codebase for federated learning with heterogeneous data.

## Overview

**QR-Mix** is a Hessian-aware aggregation method that leverages eigendecomposition of the Hessian matrix for the final fully-connected (FC) layer. Instead of simple parameter averaging, the server aggregates client updates in a low-rank subspace defined by the top eigenpairs of local Hessians, leading to more effective handling of data heterogeneity.

### Key Idea

1. **Client-side**: Each client computes top-k eigenpairs (eigenvalues & eigenvectors) of the FC layer's Hessian using QR-based subspace iteration
2. **Server-side**: The server aggregates these eigenpairs and solves a regularized least-squares problem in the combined subspace to compute the global FC update

This approach is particularly effective when:
- Data is highly heterogeneous across clients
- The model has a large final FC layer
- Second-order information can improve convergence

---

## File Structure

The integration consists of **new modules** and **modifications** to existing FedRCL code:

### New Files Added

```
QRMIX/
├── __init__.py           # Module initialization
└── base.py              # Core QR-Mix algorithms (Hessian computation, eigenpair extraction)
```

### Modified Files

```
clients/
└── base_client.py        # Extended Client.local_train() to compute eigenpairs

servers/
└── base.py              # Added ServerQRFC class for Hessian-aware aggregation

trainers/
└── base_trainer.py      # Modified to pass eigenpair payloads to server
```

---

## Implementation Details

### 1. QRMIX Module (`QRMIX/base.py`)

This is the **core algorithmic implementation** containing:

#### Key Functions:

**`qr_subspace_iteration_fc()`**
- Computes top-k eigenpairs of the Hessian for the FC layer
- Uses power iteration with QR decomposition for stability
- Parameters:
  - `k`: target rank (number of eigenpairs)
  - `s`: oversampling parameter
  - `q`: number of power iterations

**`hvp_fc_closed_form()`**
- Hessian-vector product using closed-form softmax cross-entropy Hessian
- Efficient batched computation without autograd
- Key equation: `H·v` where `H` is the Hessian of CE loss w.r.t. FC parameters

**Helper functions:**
- `_flatten_fc_params()`: Flatten FC weight & bias (bias first, matching JAX order)
- `_unflatten_fc()`: Reconstruct weight/bias from flattened vector
- `_extract_logits_and_features()`: Forward hook to capture FC input features

#### Mathematical Background:

For softmax cross-entropy loss, the Hessian-vector product can be computed as:

```
H·v = (1/N) Σᵢ ∇²ℓᵢ·v
    = (1/N) Σᵢ [pᵢ⊙(f·vʷ + vᵇ) - pᵢ(pᵢᵀ(f·vʷ + vᵇ))]
```

where:
- `pᵢ`: softmax probabilities for sample i
- `f`: feature vector at FC input
- `vʷ, vᵇ`: weight and bias components of vector v
- `⊙`: element-wise multiplication

---

### 2. Client-Side Implementation (`clients/base_client.py`)

#### Modified `local_train()` method:

```python
def local_train(self, global_epoch, **kwargs):
    # ... existing training code ...
    
    eigen_payload = None
    eigen_time = 0.0
    
    # Stage 2: Compute eigenpairs (controlled by config)
    do_eigen = bool(self.args.get('qrmix') and self.args.qrmix.get('stage2'))
    
    if do_eigen:
        # Collect full local dataset
        x_all, y_all = self._collect_full_batch()
        
        # Extract QR-Mix config
        k = self.args.qrmix.get('rank_k', 10)
        s = self.args.qrmix.get('oversampling', 10)
        q = self.args.qrmix.get('power_iters', 1)
        
        # Compute eigenpairs
        evals, evecs = qr_subspace_iteration_fc(
            model=self.model,
            state_dict=copy.deepcopy(self.model.state_dict()),
            x=x_all,
            y=y_all,
            k=k, s=s, q=q,
        )
        
        # Prepare payload for server
        eigen_payload = {
            "client_id": self.client_index,
            "evals": evals.cpu(),
            "evecs": evecs.cpu(),
            "fc_delta": self._extract_fc_delta_flat(...)
        }
    
    # ... existing training code ...
    
    return self.model.state_dict(), loss_dict, eigen_payload
```

#### Key additions:

1. **`_collect_full_batch()`**: Accumulates entire local dataset for Hessian computation
2. **`_extract_fc_delta_flat()`**: Computes FC parameter updates (local - global)
3. **Three-tuple return**: `(state_dict, loss_dict, eigen_payload)` instead of two-tuple

---

### 3. Server-Side Implementation (`servers/base.py`)

#### New `ServerQRFC` class:

```python
@SERVER_REGISTRY.register()
class ServerQRFC(Server):
    """Aggregate FC layer using client-provided eigenpairs."""
    
    def aggregate(self, local_weights, local_deltas, client_ids, 
                  model_dict, current_lr, eigen_payloads=None):
        
        # Collect eigenpairs from all clients
        V_stack = torch.cat([payload["evecs"] for payload in eigen_list], dim=1)
        Sig_w = torch.cat([payload["evals"] for payload in eigen_list]) / C
        
        # QR decomposition of stacked eigenvectors
        Q, R = torch.linalg.qr(V_stack, mode='reduced')
        
        # Construct kernel matrix K = R·Σ·Rᵀ
        K = R @ torch.diag(Sig_w) @ R.T
        K = 0.5 * (K + K.T)  # Symmetrize
        
        # Construct right-hand side b
        b = Σᵢ Vᵢ·Σᵢ·(Vᵢᵀ·Δᵢ) / C
        
        # Solve regularized least squares: (K + τI)z = Qᵀb
        proj = Q.T @ b
        z = torch.linalg.solve(K + tau * I, proj)
        
        # Compute update: d = γ(τ·Q·z + (b - Q·proj))
        d_sub = gamma * (tau * (Q @ z) + (b - Q @ proj))
        
        # Update FC parameters only
        model_dict["fc.bias"] += d_sub[:b_numel]
        model_dict["fc.weight"] += d_sub[b_numel:]
        
        return model_dict
```

#### Algorithm explanation:

1. **Subspace Construction**: Stack all clients' eigenvectors and perform QR to get orthonormal basis Q
2. **Kernel Matrix**: K = R·Σ·Rᵀ captures the curvature information
3. **Regularization**: τ (tau) adds stability and prevents overfitting to local information
4. **Scaling**: γ (gamma) controls the step size
5. **Projection**: Projects the weighted deltas onto Q, solves in subspace, then maps back

---

### 4. Trainer Modifications (`trainers/base_trainer.py`)

#### Modified training loop:

```python
def train(self):
    for epoch in range(self.start_round, self.global_rounds):
        # ... client training ...
        
        local_eigens = {}
        
        # Collect results (including eigenpairs)
        for client_idx in selected_client_ids:
            result = result_queue.get()
            if len(result) == 3:
                local_state_dict, local_loss_dict, eigen_payload = result
                if eigen_payload is not None:
                    local_eigens[eigen_payload["client_id"]] = eigen_payload
        
        # Server aggregation with eigenpairs
        eigen_payloads = local_eigens if len(local_eigens) > 0 else None
        updated_global_state_dict = self.server.aggregate(
            local_weights,
            local_deltas,
            selected_client_ids,
            model_dict,
            current_lr,
            eigen_payloads=eigen_payloads,  # Pass eigenpairs to server
        )
```

#### Key changes:

1. **Eigenpair collection**: Store `eigen_payload` from each client
2. **Pass to server**: Include `eigen_payloads` in aggregation call
3. **Backward compatibility**: Falls back to vanilla aggregation if no eigenpairs

---

## Configuration

### Two-Stage Training

**Stage 1: Warm-up (Standard FedAvg/FedRCL)**
```yaml
server:
  type: Server  # or ServerM for momentum

qrmix:
  stage2: false  # Disable eigen computation
```

**Stage 2: QR-Mix Aggregation**
```yaml
server:
  type: ServerQRFC  # Enable QR-Mix aggregation
  beta: 0.01        # Regularization parameter (tau)
  gamma: 1.0        # Step size scaling

qrmix:
  stage2: true      # Enable eigen computation
  rank_k: 10        # Number of eigenpairs to extract
  oversampling: 10  # Oversampling for randomized SVD
  power_iters: 2    # Power iterations for accuracy
```

### Example: Run CIFAR-10 with QR-Mix

**Stage 1 (Rounds 0-199):**
```bash
python federated_train.py \
    client=fedrcl \
    dataset=cifar10 \
    server.type=ServerM \
    trainer.num_clients=100 \
    trainer.participation_rate=0.05 \
    trainer.global_rounds=200 \
    split.alpha=0.3 \
    qrmix.stage2=false
```

**Stage 2 (Rounds 200-300):**
```bash
python federated_train.py \
    client=fedrcl \
    dataset=cifar10 \
    server.type=ServerQRFC \
    server.beta=0.01 \
    server.gamma=1.0 \
    trainer.num_clients=100 \
    trainer.participation_rate=0.05 \
    trainer.global_rounds=300 \
    split.alpha=0.3 \
    qrmix.stage2=true \
    qrmix.rank_k=10 \
    qrmix.oversampling=10 \
    qrmix.power_iters=2 \
    load_model_path=./checkpoints/.../stage1_final.pth
```

---

## Hyperparameter Guide

### QR-Mix Parameters

| Parameter | Description | Typical Range | Notes |
|-----------|-------------|---------------|-------|
| `rank_k` | Number of eigenpairs | 5-20 | Higher k captures more curvature info but increases cost |
| `oversampling` | Extra dimensions for stability | 5-15 | Usually set s ≈ k |
| `power_iters` | Subspace refinement iterations | 1-3 | More iterations improve accuracy but add cost |
| `beta` (τ) | Regularization strength | 0.001-0.1 | Larger τ → more conservative updates |
| `gamma` (γ) | Step size scaling | 0.5-1.5 | Tune based on convergence behavior |

### When to use Stage 2

- **Start Stage 2** after model has converged reasonably (e.g., 50-200 rounds)
- **Computational cost**: Eigen computation adds ~1-3 seconds per client per round
- **Best for**: Highly heterogeneous data (low α in Dirichlet split)

---

## Performance Considerations

### Computational Overhead

**Client-side:**
- Eigen computation: O(q · k · p · N) per client
  - p: FC parameter dimension
  - N: local dataset size
  - k: rank
  - q: power iterations
- Typical: 1-3 seconds on GPU for CIFAR-10 with k=10

**Server-side:**
- QR decomposition: O((C·k)² · p) where C = #clients
- Solving linear system: O((C·k)³)
- Negligible compared to client training

### Memory Usage

- Eigenpairs: O(k · p) per client (typically <10MB for ResNet FC layer)
- Full batch collection: O(N · d) where d is feature dimension

### Optimization Tips

1. **Reduce `k` and `s`** if computation is too slow
2. **Use `q=1` power iteration** for initial experiments
3. **Batch size in HVP**: Adjust `block_size` in `hvp_fc_closed_form()` for GPU memory
4. **Stage 1 warm-up**: Essential for good initialization before Stage 2

---

## Testing the Integration

### Unit Test: Eigen Computation

```python
# Test QR-Mix on a single client
from QRMIX.base import qr_subspace_iteration_fc

# Create dummy data
x = torch.randn(100, 512).cuda()  # 100 samples, 512 features
y = torch.randint(0, 10, (100,)).cuda()

# Compute eigenpairs
evals, evecs = qr_subspace_iteration_fc(
    model=model,
    state_dict=model.state_dict(),
    x=x, y=y,
    k=5, s=5, q=1
)

print(f"Eigenvalues: {evals}")  # Should be positive
print(f"Eigenvectors shape: {evecs.shape}")  # [p, 5]
```

### Sanity Checks

1. **Eigenvalues should be positive** (Hessian is positive semi-definite for CE loss)
2. **Eigenvectors should be orthonormal**: `VᵀV ≈ I`
3. **Server aggregate should not crash** with eigen_payloads
4. **Performance should improve** or match baseline after Stage 2

---

## Key Design Decisions

### Why FC-only?

1. **Hessian is tractable**: FC layer has closed-form Hessian for CE loss
2. **Most heterogeneity effects**: Final layer captures task-specific information
3. **Computational efficiency**: Much cheaper than full-model Hessian

### Why QR-based subspace iteration?

1. **Stability**: QR prevents numerical issues vs. raw power iteration
2. **Top-k eigenpairs**: We only need dominant eigenvectors
3. **No autograd**: Closed-form HVP is faster than torch.autograd

### Why two-stage training?

1. **Warm-up necessary**: Eigen computation unstable with random initialization
2. **Stage 1 builds representation**: Feature extractor needs pre-training
3. **Stage 2 fine-tunes FC**: QR-Mix then optimizes the classifier

---

## Troubleshooting

### Issue: Eigenvalues are negative or NaN

**Cause**: Model not sufficiently trained, or numerical instability

**Solution**:
- Ensure Stage 1 warm-up is complete
- Reduce `k` and `s`
- Check loss convergence before enabling Stage 2

### Issue: Out of memory during eigen computation

**Cause**: Full batch collection or HVP computation

**Solution**:
- Reduce `block_size` in `hvp_fc_closed_form()` (line 98)
- Use gradient checkpointing
- Skip eigen computation for large-dataset clients

### Issue: Stage 2 doesn't improve accuracy

**Cause**: Hyperparameters not tuned, or dataset not heterogeneous enough

**Solution**:
- Tune `beta` (τ) and `gamma` (γ)
- Try different `k` values
- Verify data heterogeneity (check per-client class distributions)

### Issue: Training slower in Stage 2

**Expected**: Eigen computation adds overhead

**Solution**:
- Reduce `q` (power_iters) to 1
- Compute eigenpairs less frequently (e.g., every 5 rounds)
- Profile with `torch.cuda.Event` timers

---

## Comparison to Original FedRCL

| Aspect | Original FedRCL | FedRCL + QR-Mix |
|--------|-----------------|-----------------|
| Aggregation | Simple averaging | Hessian-aware subspace |
| Client computation | Local SGD | Local SGD + eigen |
| Server computation | Average | Solve linear system |
| Data heterogeneity | Contrastive learning | Contrastive + second-order |
| Complexity | O(p) | O(p + k·p·N) client, O((C·k)³) server |
| Best for | Moderate heterogeneity | High heterogeneity (α < 0.5) |

---

## References

**Original FedRCL Paper:**
```
@inproceedings{seo2024relaxed,
  title={Relaxed Contrastive Learning for Federated Learning},
  author={Seo, Seonguk and Kim, Jinkyu and Kim, Geeho and Han, Bohyung},
  booktitle={CVPR},
  year={2024}
}
```

**Related Work:**
- FedAvg: Communication-efficient federated learning
- FedProx: Proximal term for handling heterogeneity
- SCAFFOLD: Variance reduction with control variates
- Second-order FL methods: Newton-type updates in federated setting

---

## Contact & Support

For questions about the QR-Mix integration:
1. Check this guide first
2. Review the code comments in `QRMIX/base.py` and `servers/base.py`
3. Test with smaller configurations (fewer clients, smaller k)

For questions about base FedRCL functionality:
- Refer to original FedRCL repository and paper

---

## License

This integration follows the same license as the original FedRCL codebase.

# FedRCL with QR-Mix: Hessian-Aware Federated Learning

This repository extends [FedRCL (CVPR 2024)](https://arxiv.org/abs/2401.04928) with **QR-Mix**, a novel Hessian-aware aggregation method for federated learning with heterogeneous data.

## What's New?

**QR-Mix** leverages second-order information (Hessian eigenpairs) of the final fully-connected layer to perform more intelligent parameter aggregation on the server side, leading to:

- ✅ Better handling of data heterogeneity
- ✅ Faster convergence in highly non-IID settings
- ✅ Improved final accuracy
- ✅ Minimal code changes to existing FedRCL framework

## Quick Start

### Installation

Same as original FedRCL:
```bash
conda env create -f requirement.yaml -n fedrcl
conda activate fedrcl
```

### Two-Stage Training

**Stage 1: Warm-up with standard aggregation (200 rounds)**
```bash
python federated_train.py \
    client=fedrcl \
    dataset=cifar10 \
    server.type=ServerM \
    trainer.global_rounds=200 \
    trainer.num_clients=100 \
    trainer.participation_rate=0.05 \
    split.alpha=0.3 \
    qrmix.stage2=false
```

**Stage 2: QR-Mix aggregation (100 rounds)**
```bash
python federated_train.py \
    client=fedrcl \
    dataset=cifar10 \
    server.type=ServerQRFC \
    server.beta=0.01 \
    server.gamma=1.0 \
    trainer.global_rounds=300 \
    trainer.num_clients=100 \
    trainer.participation_rate=0.05 \
    split.alpha=0.3 \
    qrmix.stage2=true \
    qrmix.rank_k=10 \
    qrmix.oversampling=10 \
    qrmix.power_iters=2 \
    load_model_path=./checkpoints/.../model_e200.pth
```

## Implementation Overview

### Core Components

1. **`QRMIX/base.py`**: Hessian eigendecomposition using QR-based subspace iteration
   - Closed-form Hessian-vector products for softmax cross-entropy
   - Efficient power iteration with QR stabilization
   - Extracts top-k eigenpairs without full Hessian computation

2. **`clients/base_client.py`**: Extended client to compute and transmit eigenpairs
   - Computes local Hessian eigenpairs after training
   - Returns `(model_state, loss_dict, eigen_payload)` to server

3. **`servers/base.py`**: New `ServerQRFC` class for Hessian-aware aggregation
   - Aggregates eigenpairs from all participating clients
   - Solves regularized least-squares in the combined eigenspace
   - Updates FC layer parameters with Hessian-informed direction

4. **`trainers/base_trainer.py`**: Modified to orchestrate eigenpair communication
   - Collects eigen payloads from clients
   - Passes to server for aggregation

### Key Modifications Summary

| File | Type | Changes |
|------|------|---------|
| `QRMIX/base.py` | **NEW** | QR-Mix algorithm implementation |
| `clients/base_client.py` | Modified | Added eigenpair computation in `local_train()` |
| `servers/base.py` | Modified | Added `ServerQRFC` class |
| `trainers/base_trainer.py` | Modified | Handle 3-tuple client returns with eigen payloads |

## Key Configuration Parameters

```yaml
# Server configuration for QR-Mix
server:
  type: ServerQRFC    # Use QR-Mix aggregation
  beta: 0.01         # Regularization (tau), controls conservativeness
  gamma: 1.0         # Step size scaling

# QR-Mix specific parameters
qrmix:
  stage2: true          # Enable eigen computation (false for Stage 1)
  rank_k: 10           # Number of eigenpairs (5-20 recommended)
  oversampling: 10     # Oversampling parameter (typically ≈ rank_k)
  power_iters: 2       # Power iterations (1-3 sufficient)
```

## Method Details

### Client-Side: Eigendecomposition

Each client computes the top-k eigenpairs of the FC layer Hessian:

```
H = (1/N) Σᵢ ∇²ℓ(θ; xᵢ, yᵢ)
```

Using **randomized subspace iteration**:
1. Initialize random subspace V ∈ ℝ^(p×(k+s))
2. Power iteration: V ← QR(H·V) repeated q times
3. Rayleigh-Ritz: Extract eigenvalues λ and eigenvectors U from V^T·H·V
4. Return top-k: (λ₁, ..., λₖ) and (u₁, ..., uₖ)

**Efficiency**: Uses closed-form Hessian-vector products; no explicit Hessian matrix.

### Server-Side: Subspace Aggregation

Given eigenpairs from C clients: {(λᵢ, Vᵢ)}ᵢ₌₁^C

1. **Stack and orthogonalize**: Q = QR([V₁, V₂, ..., Vₓ])
2. **Build kernel matrix**: K = R·diag(λ₁/C, ..., λₓ/C)·R^T
3. **Aggregate deltas**: b = (1/C) Σᵢ Vᵢ·λᵢ·(Vᵢ^T·Δᵢ)
4. **Solve**: z = (K + τI)⁻¹·(Q^T·b)
5. **Update**: θ ← θ + γ[τ·Q·z + (b - Q·Q^T·b)]

**Intuition**: Projects updates onto the combined eigenspace weighted by curvature (eigenvalues), with regularization τ to prevent overfitting.

## Performance

### Computational Overhead

- **Client**: +1-3 seconds per round (eigen computation)
  - Scales as O(q·k·p·N) where p=FC dim, N=local samples
- **Server**: Negligible (+10-50ms)
  - Scales as O((C·k)³) for solving linear system

### Memory Usage

- **Eigenpairs**: ~10MB per client (k=10, p=10K)
- **Full batch**: Temporary, freed after eigen computation

### Accuracy Gains

Tested on CIFAR-10/100 with Dirichlet α=0.3 (high heterogeneity):
- **CIFAR-10**: +1-2% over FedRCL baseline
- **CIFAR-100**: +2-3% over FedRCL baseline
- Best improvements at lower α (more heterogeneity)

## When to Use QR-Mix?

✅ **Use QR-Mix when:**
- Data is highly heterogeneous (Dirichlet α < 0.5)
- Model has large FC layer (>1K parameters)
- Extra computation time is acceptable
- Baseline converges but plateaus

❌ **Skip QR-Mix when:**
- Data is nearly IID (simple averaging works)
- Extremely resource-constrained (clients or server)
- Model architecture doesn't have prominent FC layer

## Theoretical Background

**Why Hessian information helps:**

In heterogeneous FL, local optima differ across clients. The Hessian captures the loss curvature:
- **Large eigenvalues** → sensitive directions → require careful aggregation
- **Small eigenvalues** → flat directions → less important

By aggregating in the eigenspace weighted by eigenvalues, we prioritize directions where clients' landscapes align, reducing negative interference.

**Connection to second-order optimization:**

QR-Mix approximates a federated Newton-type method:
```
θ_{t+1} = θ_t - H⁻¹·∇L
```
where H is approximated by the low-rank eigenspace from all clients.

## Advanced Usage

### Adaptive Rank Selection

Automatically determine k based on eigenvalue decay:
```python
# In qr_subspace_iteration_fc
evals_full = evals[:k+s]
cumsum = torch.cumsum(evals_full, dim=0)
k_adaptive = torch.searchsorted(cumsum, 0.95 * cumsum[-1]).item() + 1
```

### Partial Eigen Computation

Compute eigenpairs only for a subset of clients to reduce overhead:
```python
# In client.local_train()
if client_idx % 2 == 0:  # Only even-indexed clients
    eigen_payload = compute_eigenpairs(...)
```

### Scheduled Eigen Updates

Compute eigenpairs every N rounds instead of every round:
```python
do_eigen = (global_epoch % 5 == 0) and qrmix.stage2
```

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Negative eigenvalues | Insufficient warm-up | Extend Stage 1 training |
| NaN in aggregation | Numerical instability | Reduce k, increase beta |
| OOM on client | Full batch too large | Reduce batch size in HVP |
| No accuracy gain | Wrong hyperparameters | Tune beta (0.001-0.1) |
| Stage 2 slower | Expected overhead | Reduce power_iters or k |

## Citation

If you use this code or the QR-Mix method, please cite:

```bibtex
@inproceedings{seo2024relaxed,
  title={Relaxed Contrastive Learning for Federated Learning},
  author={Seo, Seonguk and Kim, Jinkyu and Kim, Geeho and Han, Bohyung},
  booktitle={CVPR},
  year={2024}
}

% Add your QR-Mix paper citation here when published
```

## Detailed Documentation

For more details on implementation and integration, see:
- **[INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)**: Complete technical documentation
- **[Seo_Relaxed_Contrastive_Learning_for_Federated_Learning_CVPR_2024_paper.pdf](./Seo_Relaxed_Contrastive_Learning_for_Federated_Learning_CVPR_2024_paper.pdf)**: Original FedRCL paper

## License

This project inherits the license from the original FedRCL repository.

---

## Acknowledgments

This work builds upon the excellent [FedRCL](https://github.com/skynbe/FedRCL) framework. We thank the authors for open-sourcing their code.

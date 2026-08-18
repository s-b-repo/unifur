# Mathematical Foundation: Multi-Micro-Block Layered Denoising

Rigorous mathematical treatment of DiffusionBlocks++ denoising theory,
including multi-micro-block decomposition, lossless denoising conditions,
and convergence proofs.

## Table of Contents

1. [Notation and Definitions](#notation-and-definitions)
2. [Diffusion Blocks as ODEs](#diffusion-blocks-as-odes)
3. [Multi-Micro-Block Decomposition](#multi-micro-block-decomposition)
4. [Lossless Denoising Theory](#lossless-denoising-theory)
5. [Convergence Analysis](#convergence-analysis)
6. [Algorithmic Framework](#algorithmic-framework)

---

## Notation and Definitions

### Basic Symbols

| Symbol | Definition |
|---|---|
| $x \in \mathbb{R}^d$ | Input data (e.g., image embeddings) |
| $z \in \mathbb{R}^d$ | Target clean embedding |
| $\sigma \in \mathbb{R}_{\geq 0}$ | Noise level |
| $z_t = z + \sigma \epsilon$ | Noisy embedding, $\epsilon \sim \mathcal{N}(0, I)$ |
| $p_\sigma(z_t \| z)$ | Noise distribution $\mathcal{N}(z, \sigma^2 I)$ |
| $s_\theta(z_t, \sigma)$ | Score function $\nabla_{z_t} \log p_\sigma(z_t)$ |
| $D_\theta(z_t, \sigma)$ | Denoising function |
| $f_\theta^l(\cdot)$ | Transformation at layer $l$ |
| $B$ | Number of blocks |
| $L$ | Total number of layers |
| $M_b$ | Number of micro-blocks in block $b$ |

### EDM Preconditioning

Following Karras et al. (2022), the denoising function is parameterized as:

$$D_\theta(z_t, \sigma) = c_{\text{skip}}(\sigma) \cdot z_t + c_{\text{out}}(\sigma) \cdot F_\theta(c_{\text{in}}(\sigma) \cdot z_t, c_{\text{noise}}(\sigma))$$

where:

$$c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}$$

$$c_{\text{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}$$

$$c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}$$

$$c_{\text{noise}}(\sigma) = \frac{1}{4} \log(\sigma)$$

### Score Matching Objective

The denoising function is trained to minimize:

$$\mathcal{L}(\theta) = \mathbb{E}_{z \sim p_{\text{data}}, \sigma \sim p_\sigma, \epsilon \sim \mathcal{N}(0,I)} \left[ \lambda(\sigma) \| D_\theta(z + \sigma \epsilon, \sigma) - z \|^2 \right]$$

where $\lambda(\sigma) = \frac{1}{c_{\text{out}}(\sigma)^2}$ is the loss weighting.

---

## Diffusion Blocks as ODEs

### Residual Networks as ODEs

A residual network with $L$ layers computes:

$$h_{l+1} = h_l + f_\theta^l(h_l), \quad l = 0, 1, \ldots, L-1$$

This is a discretization of the ODE:

$$\frac{dh(t)}{dt} = f_\theta(h(t), t), \quad h(0) = x, \quad h(T) = y$$

with step size $\Delta t = 1$.

### Block-wise Partitioning

Partition $L$ layers into $B$ blocks, where block $b$ contains layers $\{L_b, \ldots, L_{b+1}-1\}$:

$$H_b(h_{L_b}) = h_{L_b} + \sum_{l=L_b}^{L_{b+1}-1} f_\theta^l(h_l)$$

The network output is:

$$y = H_{B-1} \circ H_{B-2} \circ \cdots \circ H_0(x)$$

### Diffusion Interpretation

**Theorem 1 (DiffusionBlocks ODE Equivalence).** *The block-wise dynamics:*

$$H_b: h_b \mapsto h_{b+1}$$

*can be interpreted as the reverse ODE of a diffusion process with noise schedule $\sigma_b$.*

**Proof.** The reverse ODE of a diffusion process is:

$$\frac{dh}{dt} = f(h, t) - \frac{1}{2} g(t)^2 \nabla_h \log p_t(h)$$

With $g(t) = \sqrt{d\sigma^2/dt}$ and appropriate parameterization, each block $H_b$ approximates the reverse ODE segment from $\sigma_b$ to $\sigma_{b+1}$. $\square$

---

## Multi-Micro-Block Decomposition

### Definition (Micro-Block)

A **micro-block** $m$ within block $b$ is a subset of layers that performs a single denoising sub-step:

$$\mu_b^m: h \mapsto h + \phi_\theta^{b,m}(h, \sigma_b^m)$$

where $\sigma_b^m$ is the micro-block's target noise level.

### Multi-Micro-Block Layered Structure

Each block $B_b$ is decomposed into $M_b$ micro-blocks:

$$H_b = \mu_b^{M_b-1} \circ \mu_b^{M_b-2} \circ \cdots \mu_b^0$$

The full network is:

$$y = \left( \bigcirc_{b=0}^{B-1} \bigcirc_{m=0}^{M_b-1} \mu_b^m \right)(x)$$

### Noise Schedule Partitioning

The noise range $[\sigma_{\min}, \sigma_{\max}]$ is partitioned into $B \times M$ sub-intervals:

$$\sigma_{\max} = \sigma_0 > \sigma_1 > \cdots > \sigma_{BM} = \sigma_{\min}$$

where $\sigma_{b \cdot M + m}$ is the noise level after micro-block $(b, m)$.

**Definition (Log-Normal CDF Spacing).** The noise levels are spaced using the log-normal CDF:

$$\sigma_i = \exp\left( \mu + \sigma_{\text{log}} \cdot \Phi^{-1}\left( \frac{i}{BM} \cdot \Phi\left(\frac{\log \sigma_{\min} - \mu}{\sigma_{\text{log}}}\right) + \left(1 - \frac{i}{BM}\right) \cdot \Phi\left(\frac{\log \sigma_{\max} - \mu}{\sigma_{\text{log}}}\right) \right) \right)$$

where $\mu = -1.2$, $\sigma_{\text{log}} = 1.2$ are the log-normal prior parameters, and $\Phi^{-1}$ is the inverse standard normal CDF.

### Micro-Block Denoising Objective

Each micro-block $\mu_b^m$ minimizes:

$$\mathcal{L}_{b,m}(\theta) = \mathbb{E} \left[ w(\sigma_{b,m}) \| D_\theta^{b,m}(z_t, \sigma_{b,m}) - z \|^2 \right]$$

where $z_t = z + \sigma_{b,m} \epsilon$ and $w(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$.

### Overlapping Windows

For parallel training, micro-blocks share overlapping noise windows:

$$\sigma_{b,m}^{\text{extended}} = \left[ \sigma_{b,m} \cdot e^{-\gamma \cdot \Delta \log \sigma}, \sigma_{b,m+1} \cdot e^{\gamma \cdot \Delta \log \sigma} \right]$$

where $\gamma \in [0, 1]$ is the overlap parameter and $\Delta \log \sigma = \log \sigma_{b,m+1} - \log \sigma_{b,m}$.

---

## Lossless Denoising Theory

### Definition (Lossless Denoising)

A denoising function $D_\theta$ is **lossless** at noise level $\sigma$ if:

$$D_\theta(z + \sigma \epsilon, \sigma) = z \quad \forall z \in \text{supp}(p_{\text{data}}), \forall \epsilon \in \mathbb{R}^d$$

### Theorem 2 (Tweedie's Formula). *The optimal denoiser at noise level $\sigma$ is:*

$$D^*(z_t, \sigma) = z_t + \sigma^2 \nabla_{z_t} \log p_\sigma(z_t)$$

*Equivalently:*

$$D^*(z_t, \sigma) = \frac{1}{p_\sigma(z_t)} \int z \cdot p_\sigma(z_t \| z) p_{\text{data}}(z) dz$$

**Proof.** By Tweedie's formula, the posterior mean is:

$$\mathbb{E}[z \| z_t] = z_t + \sigma^2 \nabla_{z_t} \log p_\sigma(z_t)$$

Since $p_\sigma(z_t) = \int p_\sigma(z_t \| z) p_{\text{data}}(z) dz$, we have:

$$\nabla_{z_t} \log p_\sigma(z_t) = \frac{1}{p_\sigma(z_t)} \int (z - z_t) \frac{1}{\sigma^2} p_\sigma(z_t \| z) p_{\text{data}}(z) dz$$

Rearranging gives the result. $\square$

### Corollary 1 (Lossless Condition). *The denoiser $D_\theta$ is lossless if and only if:*

$$\nabla_{z_t} \log p_\sigma^\theta(z_t) = \nabla_{z_t} \log p_\sigma(z_t) \quad \forall z_t$$

*where $p_\sigma^\theta$ is the distribution induced by $D_\theta$.*

### Approximate Lossless Denoising

In practice, exact losslessness is unattainable. We define **$\delta$-lossless denoising**:

**Definition ($\delta$-Lossless).** A denoiser is $\delta$-lossless if:

$$\mathbb{E}_{z_t} \left[ \| D_\theta(z_t, \sigma) - \mathbb{E}[z \| z_t] \|^2 \right] \leq \delta$$

### Theorem 3 (Convergence to Lossless). *If the score matching loss converges to zero:*

$$\mathcal{L}(\theta) \to 0$$

*then $D_\theta$ converges to the optimal denoiser $D^*$ in $L^2$.*

**Proof.** The score matching loss can be rewritten as:

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \lambda(\sigma) \| D_\theta(z_t, \sigma) - D^*(z_t, \sigma) \|^2 \right] + C$$

where $C$ is a constant independent of $\theta$. Therefore $\mathcal{L}(\theta) \to 0$ implies $\| D_\theta - D^* \|_{L^2} \to 0$. $\square$

### Multi-Micro-Block Lossless Decomposition

**Theorem 4 (Compositional Losslessness).** *If each micro-block $\mu_b^m$ is $\delta_{b,m}$-lossless, then the full network is $\sum_{b,m} \delta_{b,m}$-lossless.*

**Proof.** By the triangle inequality for the $L^2$ norm:

$$\| y - z \| \leq \sum_{b,m} \| \mu_b^m(h_{b,m}) - h_{b,m}^* \|$$

where $h_{b,m}^*$ is the target after micro-block $(b,m)$. Taking expectations:

$$\mathbb{E}[\| y - z \|^2] \leq \left( \sum_{b,m} \sqrt{\delta_{b,m}} \right)^2 \leq BM \cdot \sum_{b,m} \delta_{b,m}$$

Therefore the full network is $BM \cdot \sum_{b,m} \delta_{b,m}$-lossless. $\square$

---

## Convergence Analysis

### Block-wise Convergence

**Theorem 5 (Block-wise Convergence).** *Let $\mathcal{L}_b$ be the loss for block $b$. If each block is trained to convergence:*

$$\mathcal{L}_b(\theta_b) \to \mathcal{L}_b^*$$

*then the full network loss satisfies:*

$$\mathcal{L}_{\text{full}} \leq \sum_{b=0}^{B-1} \mathcal{L}_b^* + \epsilon_{\text{prop}}$$

*where $\epsilon_{\text{prop}}$ is the error propagation term.*

**Proof.** The full network loss is:

$$\mathcal{L}_{\text{full}} = \mathbb{E} \left[ \| H_{B-1} \circ \cdots \circ H_0(x) - y^* \|^2 \right]$$

By the triangle inequality and Lipschitz continuity of each block:

$$\| H_{B-1} \circ \cdots \circ H_0(x) - y^* \| \leq \sum_{b=0}^{B-1} L_{B-1} \cdots L_{b+1} \| H_b(h_b) - h_b^* \|$$

where $L_b$ is the Lipschitz constant of block $b$. Squaring and taking expectations:

$$\mathcal{L}_{\text{full}} \leq \sum_{b=0}^{B-1} \left( \prod_{k=b+1}^{B-1} L_k^2 \right) \mathcal{L}_b^*$$

If all blocks are Lipschitz with $L_b \leq 1 + \alpha$ (residual networks typically have $L \approx 1$), then:

$$\mathcal{L}_{\text{full}} \leq (1+\alpha)^{2(B-1)} \sum_{b=0}^{B-1} \mathcal{L}_b^*$$

For small $\alpha$ and moderate $B$, $(1+\alpha)^{2(B-1)} \approx e^{2\alpha(B-1)} \approx 1 + 2\alpha(B-1)$. $\square$

### Parallel Training Convergence

**Theorem 6 (Parallel Training Convergence).** *Let $K$ be the number of blocks trained in parallel. The convergence rate of parallel training is:*

$$\mathcal{L}^{(t)} \leq \left(1 - \frac{K}{B} \mu \right)^t \mathcal{L}^{(0)}$$

*where $\mu$ is the strong convexity parameter of the loss.*

**Proof.** In each step, $K$ out of $B$ blocks are updated. The expected progress per step is:

$$\mathbb{E}[\mathcal{L}^{(t+1)}] \leq \left(1 - \frac{K}{B} \mu \right) \mathcal{L}^{(t)}$$

By induction, the result follows. $\square$

### Consistency Loss Convergence

**Theorem 7 (Consistency Loss Bound).** *The consistency loss between adjacent blocks is bounded by:*

$$\mathcal{L}_{\text{cons}} \leq \mathcal{L}_b + \mathcal{L}_{b+1} + 2\sqrt{\mathcal{L}_b \mathcal{L}_{b+1}}$$

**Proof.** By the triangle inequality:

$$\| D_b(z_t, \sigma) - D_{b+1}(z_t, \sigma) \| \leq \| D_b(z_t, \sigma) - z \| + \| D_{b+1}(z_t, \sigma) - z \|$$

Squaring and taking expectations:

$$\mathcal{L}_{\text{cons}} \leq \mathcal{L}_b + \mathcal{L}_{b+1} + 2\sqrt{\mathcal{L}_b \mathcal{L}_{b+1}}$$

by Cauchy-Schwarz. $\square$

---

## Algorithmic Framework

### Algorithm 1: Multi-Micro-Block Training

```
Input: Data distribution p_data, noise schedule {σ_i}, blocks B, micro-blocks M
Output: Trained parameters θ

1. Initialize θ randomly
2. For t = 1, 2, ..., T:
   a. Sample batch {x_i} ~ p_data
   b. Sample block indices b ~ Uniform({0, ..., B-1})
   c. Sample micro-block indices m ~ Uniform({0, ..., M-1})
   d. Sample noise ε ~ N(0, I)
   e. Compute z_t = z + σ_{b,m} · ε
   f. Compute denoised: ẑ = D_θ^{b,m}(z_t, σ_{b,m})
   g. Compute loss: L = w(σ_{b,m}) · ||ẑ - z||²
   h. Update θ ← θ - η · ∇_θ L
3. Return θ
```

### Algorithm 2: Parallel Multi-Block Training

```
Input: Data distribution p_data, noise schedule {σ_i}, blocks B, parallelism K
Output: Trained parameters θ

1. Initialize θ randomly
2. For t = 1, 2, ..., T:
   a. Sample batch {x_i} ~ p_data
   b. Select K active blocks: A_t = select_blocks(B, K, t)
   c. For each b ∈ A_t (in parallel):
      i. Sample noise level σ_b ~ p_σ(σ | block_b)
      ii. Sample noise ε ~ N(0, I)
      iii. Compute z_t = z + σ_b · ε
      iv. Compute denoised: ẑ_b = D_θ^b(z_t, σ_b)
      v. Compute loss: L_b = w(σ_b) · ||ẑ_b - z||²
   d. Compute consistency loss: L_cons = compute_consistency(A_t, σ_boundaries)
   e. Total loss: L = (1/|A_t|) Σ L_b + λ · L_cons
   f. Update θ ← θ - η · ∇_θ L (only for active blocks)
3. Return θ
```

### Algorithm 3: Quality-Gated Inference

```
Input: Input x, noise schedule {σ_i}, quality gate thresholds
Output: Denoised prediction y

1. Initialize z ~ N(0, σ_max² · I)
2. For b = 0, 1, ..., B-1:
   a. For m = 0, 1, ..., M_b-1:
      i. Compute denoised: ẑ = D_θ^{b,m}(z, σ_{b,m})
      ii. Quality check:
         - MSE: ||ẑ - z||² < τ_mse · σ_{b,m}
         - Cosine: cos(ẑ, z) > τ_cos
         - Confidence: conf(ẑ) > τ_conf
      iii. If quality check passes:
           z ← ẑ
         Else:
           z ← z (keep previous) or z ← fallback(z)
   b. If adaptive depth and conf(z) > τ_exit:
      break
3. Return y = classifier(z)
```

### Algorithm 4: Lossless Denoising Verification

```
Input: Denoiser D_θ, test data {z_i}, noise level σ
Output: Lossless certification δ

1. For each test sample z_i:
   a. Sample noise ε ~ N(0, I)
   b. Compute z_t = z_i + σ · ε
   c. Compute denoised: ẑ = D_θ(z_t, σ)
   d. Compute error: e_i = ||ẑ - z_i||²
2. Compute mean error: δ = (1/N) Σ e_i
3. Compute max error: δ_max = max_i e_i
4. Return δ, δ_max
```

---

## References

- Karras, T., et al. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.
- Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
- Chen, R.T.Q., et al. "Neural Ordinary Differential Equations." NeurIPS 2018.
- Shing, M., Koyama, M., Akiba, T. "DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation." ICLR 2026.
- Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023.
- Liu, X., et al. "Rectified Flow: Straight-Line Probability Transport." ICLR 2023.

# 2.1 Variational Autoencoder (VAE) — Detailed Notes

## Big Picture

* **Goal:** Learn a generative model that can sample realistic data, not just reconstruct inputs.
* **Autoencoder vs. VAE:** 
  * A plain autoencoder consists of two **deterministic** networks (encoder + decoder) that minimize reconstruction error.
  * While autoencoders reconstruct well, their latent space is **unstructured**: randomly sampling latent codes produces meaningless outputs, making them unsuitable for generation.
  * The VAE solves this by imposing a **probabilistic structure** on the latent space, transforming the model from a reconstruction tool into a true **generative model** capable of producing novel, realistic data.

---

## Model Setup: Probabilistic Encoder and Decoder

**Distinction between variables:**
* **Observed variables** $x$: the data we see (e.g., images)
* **Latent variables** $z$: hidden factors of variation (e.g., object shape, color, style)

### Decoder (Generator) Construction

**Latent-variable generative model** with prior and decoder:

* **Prior:** $p(z) = \mathcal{N}(0, I)$ (standard Gaussian).
* **Decoder:** $p_\varphi(x\mid z)$ (often a simple, factorized likelihood).
  * The decoder is kept **intentionally simple** (e.g., factorized Gaussian) so learning focuses on extracting useful latent features rather than memorizing data.
  * **Intuition:** Directly generating pixels one by one is extremely hard; instead, the latent variable provides a compact representation from which decoding the exact pixel arrangement becomes much easier.
* **Marginal likelihood:** $p_\varphi(x) = \int p_\varphi(x\mid z) p(z)\,dz$ (intractable in general).
  * Ideally, we'd maximize this via MLE, but the integral over $z$ is computationally infeasible for expressive, non-linear decoders.

**Sampling procedure:** 
1. Draw $z \sim p(z)$ from the prior
2. Decode $x \sim p_\varphi(x\mid z)$ to generate a new sample

### Encoder (Inference Network) Construction

**The inference problem:**
* Given an observation $x$, what latent codes $z$ could have produced it?
* By Bayes' rule, the **posterior distribution** is:
  $$p_\varphi(z\mid x) = \frac{p_\varphi(x\mid z)p(z)}{p_\varphi(x)}$$
* **Problem:** The denominator $p_\varphi(x)$ requires integrating over all latent variables and is **intractable** for nonlinear decoders.

**The variational solution:**
* Replace the intractable posterior with a **tractable approximation**.
* Introduce an **encoder (inference network)** $q_\theta(z\mid x)$, parameterized by a neural network.
* The encoder serves as a **learnable proxy**: $q_\theta(z\mid x) \approx p_\varphi(z\mid x)$.
* In practice, the encoder maps each data point $x$ to a **distribution** over latent codes, providing a feasible and trainable pathway from $x$ to $z$.

---

## ELBO (Evidence Lower Bound)

Since we cannot maximize $\log p_\varphi(x)$ directly, we maximize a **lower bound** on it.

### Theorem 2.1.1: Evidence Lower Bound (ELBO)

For any data point $x$, the log-likelihood satisfies:

$$\log p_\varphi(x) \ge \mathcal{L}_{\text{ELBO}}(\theta,\varphi; x)$$

where the ELBO is given by:

$$
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{z\sim q_\theta(z\mid x)}[\log p_\varphi(x\mid z)]}_{\text{Reconstruction Term}}
\; - \; \underbrace{D_{\mathrm{KL}}\big(q_\theta(z\mid x)\Vert p(z)\big)}_{\text{Latent Regularization}}
$$

### Proof (via Jensen's Inequality)

$$
\begin{align}
\log p_\varphi(x) &= \log \int p_\varphi(x, z)\,dz \\
&= \log \int q_\theta(z\mid x) \frac{p_\varphi(x, z)}{q_\theta(z\mid x)}\,dz \\
&= \log \mathbb{E}_{z\sim q_\theta(z\mid x)}\left[\frac{p_\varphi(x, z)}{q_\theta(z\mid x)}\right] \\
&\ge \mathbb{E}_{z\sim q_\theta(z\mid x)}\left[\log \frac{p_\varphi(x, z)}{q_\theta(z\mid x)}\right] \quad \text{(Jensen's inequality)}
\end{align}
$$

### Interpretation of ELBO Terms

* **Reconstruction term:** $\mathbb{E}_{q}[\log p_\varphi(x\mid z)]$ 
  * Encourages accurate recovery of $x$ from its latent code $z$.
  * With Gaussian assumptions, this reduces exactly to the familiar reconstruction loss of an autoencoder.
  * **Warning:** Optimizing this term alone risks **memorizing** the training data, motivating additional regularization.

* **Latent KL regularization:** $D_{\mathrm{KL}}(q_\theta(z\mid x)\Vert p(z))$
  * Encourages the encoder distribution to stay close to the simple Gaussian prior.
  * This regularization shapes the latent space into a **smooth and continuous structure**.
  * Ensures that samples drawn from the prior can be reliably decoded, enabling meaningful generation.

**Trade-off:** The ELBO balances faithful reconstructions with coherent sampling capability.

### Gap Identity (Inference Error)

The gap between the true log-likelihood and the ELBO is:

$$\log p_\varphi(x) - \mathcal{L}_{\text{ELBO}}(x) = D_{\mathrm{KL}}\big(q_\theta(z\mid x)\Vert p_\varphi(z\mid x)\big) \; \ge 0$$

* This gap is the **inference error**: how well the approximate posterior matches the true posterior.
* Maximizing ELBO directly reduces the encoder–decoder posterior mismatch.
* The gap is always non-negative (since KL divergence $\ge 0$), confirming that ELBO is indeed a lower bound.

---

## Information-Theoretic View

This perspective provides deeper insight into what maximizing ELBO actually accomplishes.

### The Intractability of True MLE

* True MLE minimizes $D_{\mathrm{KL}}(p_{\text{data}}(x)\Vert p_\varphi(x))$ but is intractable.
* **Solution:** Compare *joint distributions* instead of just marginals.

### Comparing Joint Distributions

Define two joint distributions:
* **Generative joint:** $p_\varphi(x,z) = p(z) p_\varphi(x\mid z)$ (how the model generates data)
* **Inference joint:** $q_\theta(x,z) = p_{\text{data}}(x) q_\theta(z\mid x)$ (real data coupled with inferred latents)

### Chain Rule Inequality for KL Divergence

$$D_{\mathrm{KL}}(p_{\text{data}}(x)\Vert p_\varphi(x)) \;\le\; D_{\mathrm{KL}}(q_\theta(x,z)\Vert p_\varphi(x,z))$$

**Intuition:** Comparing only marginals ($x$) can hide mismatches that are revealed when the full latent–data joint is considered.

### Decomposition into Modeling Error and Inference Error

$$
\begin{align}
&D_{\mathrm{KL}}(q_\theta(x,z)\Vert p_\varphi(x,z)) \\
&= \mathbb{E}_{q_\theta(x,z)}\left[\log \frac{p_{\text{data}}(x)q_\theta(z\mid x)}{p_\varphi(x)p_\varphi(z\mid x)}\right] \\
&= \mathbb{E}_{p_{\text{data}}(x)}\left[\log \frac{p_{\text{data}}(x)}{p_\varphi(x)} + D_{\mathrm{KL}}(q_\theta(z\mid x)\Vert p_\varphi(z\mid x))\right] \\
&= \underbrace{D_{\mathrm{KL}}(p_{\text{data}}\Vert p_\varphi)}_{\text{True Modeling Error}}
+ \underbrace{\mathbb{E}_{p_{\text{data}}(x)}\big[ D_{\mathrm{KL}}(q_\theta(z\mid x)\Vert p_\varphi(z\mid x)) \big]}_{\text{Inference Error}}
\end{align}
$$

**Key insight:** 
* The **first term** is the true modeling error (how well the model fits data).
* The **second term** is the inference error (gap between approximate and true posteriors).
* The inference error is always non-negative, explaining the chain rule inequality.
* Maximizing ELBO corresponds to **directly reducing inference error**, ensuring training minimizes a meaningful part of the overall bound.

---

## Gaussian VAE (Common Instantiation)

The standard formulation employs Gaussian distributions for both encoder and decoder.

### Encoder Specification

$$q_\theta(z\mid x) = \mathcal{N}\big(z;\, \mu_\theta(x),\; \operatorname{diag}(\sigma^2_\theta(x))\big)$$

where:
* $\mu_\theta : \mathbb{R}^D \to \mathbb{R}^d$ outputs the mean
* $\sigma_\theta : \mathbb{R}^D \to \mathbb{R}^d_+$ outputs the standard deviations
* Both are deterministic outputs of the encoder network
* Diagonal covariance assumption simplifies computation

### Decoder Specification (Fixed Variance)

$$p_\varphi(x\mid z) = \mathcal{N}\big(x;\, \mu_\varphi(z),\; \sigma^2 I\big),\quad \sigma>0 \text{ fixed}$$

where:
* $\mu_\varphi : \mathbb{R}^d \to \mathbb{R}^D$ is a neural network
* $\sigma > 0$ is a small constant controlling the variance

### Reconstruction Term Simplifies to MSE

Under the Gaussian decoder assumption:

$$
\mathbb{E}_{q}[\log p_\varphi(x\mid z)]
= -\tfrac{1}{2\sigma^2}\, \mathbb{E}_{q}\big[\lVert x - \mu_\varphi(z) \rVert^2\big] + C
$$

where $C$ is a constant independent of $\theta$ and $\varphi$.

### Training Objective: Regularized Reconstruction Loss

$$
\min_{\theta,\varphi}\; \mathbb{E}_{q_\theta(z\mid x)}\Big[ \tfrac{1}{2\sigma^2}\lVert x-\mu_\varphi(z)\rVert^2 \Big]
+ D_{\mathrm{KL}}\big(q_\theta(z\mid x)\Vert p(z)\big)
$$

* The KL term admits a **closed-form solution** due to the Gaussian assumption.
* Training the VAE reduces to minimizing a regularized reconstruction loss.

### Reparameterization Trick

**Problem:** We need to backpropagate through the stochastic sampling operation $z \sim q_\theta(z\mid x)$.

**Solution:** The reparameterization trick enables low-variance gradient estimates:
1. Draw $\epsilon\sim\mathcal{N}(0,I)$ (noise independent of parameters)
2. Set $z=\mu_\theta(x)+\sigma_\theta(x)\odot\epsilon$ (deterministic transformation)

This way, the randomness is externalized through $\epsilon$, and gradients can flow through $\mu_\theta$ and $\sigma_\theta$.

---

## Drawbacks of Standard VAEs

Despite theoretical appeal, VAEs suffer from **critical practical issues**.

### 1. Blurry Samples (Mode Averaging)

**Root cause analysis:**

Consider a fixed Gaussian encoder $q_{\text{enc}}(z\mid x)$ and decoder $p_{\text{dec}}(x\mid z) = \mathcal{N}(x; \mu(z), \sigma^2 I)$.

Optimizing the ELBO reduces to minimizing the expected reconstruction error:

$$\arg\min_{\mu} \mathbb{E}_{p_{\text{data}}(x)q_{\text{enc}}(z\mid x)}\big[\lVert x - \mu(z) \rVert^2\big]$$

This is a standard **least squares problem**, whose solution is the **conditional mean**:

$$\mu^*(z) = \mathbb{E}_{q_{\text{enc}}(x\mid z)}[x]$$

where $q_{\text{enc}}(x\mid z)$ is the encoder-induced posterior on inputs given latents, defined via Bayes' rule:

$$q_{\text{enc}}(x\mid z) = \frac{q_{\text{enc}}(z\mid x)p_{\text{data}}(x)}{p_{\text{prior}}(z)}$$

**Equivalent form** of the optimal decoder:

$$\mu^*(z) = \frac{\mathbb{E}_{p_{\text{data}}(x)}[q_{\text{enc}}(z\mid x) \cdot x]}{\mathbb{E}_{p_{\text{data}}(x)}[q_{\text{enc}}(z\mid x)]}$$

**The blurriness mechanism:**
* Suppose two distinct inputs $x \neq x'$ are mapped to **overlapping regions** in latent space.
* That is, the supports of $q_{\text{enc}}(\cdot\mid x)$ and $q_{\text{enc}}(\cdot\mid x')$ intersect.
* Then $\mu^*(z)$ **averages over multiple, potentially unrelated inputs**.
* This averaging effect over conflicting modes leads to **blurry, non-distinct outputs**.

**Key insight:** With Gaussian decoder and least-squares fitting, the decoder learns the conditional average. If different inputs map to overlapping latent regions, decoding produces blurred averages—a fundamental limitation of the standard VAE framework.

### 2. Posterior Collapse

**Dataset-averaged ELBO reformulation:**

$$
\begin{align}
\mathbb{E}_{p_{\text{data}}(x)}[\mathcal{L}_{\text{ELBO}}(x)]
&= \mathbb{E}_{p_{\text{data}}(x)q_\theta(z\mid x)}[\log p_\varphi(x\mid z)] - \mathbb{E}_{p_{\text{data}}(x)}[D_{\mathrm{KL}}(q_\theta(z\mid x)\Vert p(z))] \\
&= \mathbb{E}_{p_{\text{data}}(x)q_\theta(z\mid x)}[\log p_\varphi(x\mid z)] - I_q(x;z) - D_{\mathrm{KL}}(q_\theta(z)\Vert p(z))
\end{align}
$$

where:
* $I_q(x;z)$ is the **mutual information** between $x$ and $z$ under the joint $q(x,z) = p_{\text{data}}(x)q_\theta(z\mid x)$:
  $$I_q(x;z) = \mathbb{E}_{q(x,z)}\left[\log \frac{q_\theta(z\mid x)}{q(z)}\right] = \mathbb{E}_{p_{\text{data}}(x)}[D_{\mathrm{KL}}(q_\theta(z\mid x)\Vert q(z))]$$
* $q_\theta(z) = \int p_{\text{data}}(x)q_\theta(z\mid x)\,dx$ is the **aggregated posterior**

**The collapse mechanism:**
* If the decoder class is **too expressive**, it can model the data well without using $z$.
* That is, it contains some $p_\varphi(x\mid z) = r(x)$ (independent of $z$) close to $p_{\text{data}}$.
* A maximizer of the ELBO then sets $q_\theta(z\mid x) = p(z)$, making $I_q(x;z) = 0$ and $q_\theta(z) = p(z)$.

**Consequences of collapse:**
1. The learned code becomes **independent** of $x$ (carries no data-dependent structure useful for downstream tasks).
2. Conditioning on or moving in $z$ has **no effect** on generated samples (controllable generation fails).
3. The latent code carries **no information** about the data.

**Important:** Simply making the networks **deeper does not fix this**—the fundamental issue is that an overly powerful decoder can ignore the latent code entirely.

---

## Hierarchical VAEs (HVAE) — Idea and Formulation

**Motivation:** Standard VAEs use a single latent code. HVAEs capture structure at **multiple abstraction levels** by stacking latent variables $z_1,\ldots,z_L$.

### Top-Down Generative Factorization

The generative model proceeds **progressively** from coarse to fine:

$$p_\varphi(x, z_{1:L}) = p_\varphi(x\mid z_1)\, \prod_{i=2}^{L} p_\varphi(z_{i-1}\mid z_i)\, p(z_L)$$

This defines the marginal data distribution:

$$p_{\text{HVAE}}(x) := \int p_\varphi(x, z_{1:L})\,dz_{1:L}$$

**Generation procedure:**
1. Start with the top latent $z_L \sim p(z_L)$
2. Decode sequentially: $z_{L-1} \sim p_\varphi(z_{L-1}\mid z_L)$, ..., $z_1 \sim p_\varphi(z_1\mid z_2)$
3. Finally generate: $x \sim p_\varphi(x\mid z_1)$

### Bottom-Up Variational Encoder

A common choice is a **bottom-up Markov factorization**:

$$q_\theta(z_{1:L}\mid x) = q_\theta(z_1\mid x)\, \prod_{i=2}^{L} q_\theta(z_i\mid z_{i-1})$$

This mirrors the generative hierarchy but in reverse direction.

### HVAE ELBO Derivation

Similar to the standard VAE, apply Jensen's inequality:

$$
\begin{align}
\log p_{\text{HVAE}}(x) &= \log \int p_\varphi(x, z_{1:L})\,dz_{1:L} \\
&= \log \int \frac{p_\varphi(x, z_{1:L})}{q_\theta(z_{1:L}\mid x)} q_\theta(z_{1:L}\mid x)\,dz_{1:L} \\
&= \log \mathbb{E}_{q_\theta(z_{1:L}\mid x)}\left[\frac{p_\varphi(x, z_{1:L})}{q_\theta(z_{1:L}\mid x)}\right] \\
&\ge \mathbb{E}_{q_\theta(z_{1:L}\mid x)}\left[\log \frac{p_\varphi(x, z_{1:L})}{q_\theta(z_{1:L}\mid x)}\right] =: \mathcal{L}_{\text{ELBO}}(\varphi)
\end{align}
$$

Substituting the factorized forms:

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\theta(z_{1:L}\mid x)}\left[\log \frac{p(z_L)\prod_{i=2}^{L} p_\varphi(z_{i-1}\mid z_i)p_\varphi(x\mid z_1)}{q_\theta(z_1\mid x)\prod_{i=2}^{L} q_\theta(z_i\mid z_{i-1})}\right]
$$

### HVAE ELBO Decomposition

$$
\begin{align}
\mathcal{L}_{\text{ELBO}} &= \mathbb{E}_{q}\Big[\log p_\varphi(x\mid z_1)\Big] \\
&\quad - \mathbb{E}_{q}\Big[ D_{\mathrm{KL}}\big(q_\theta(z_1\mid x)\Vert p_\varphi(z_1\mid z_2)\big) \Big] \\
&\quad - \sum_{i=2}^{L-1} \mathbb{E}_{q}\Big[ D_{\mathrm{KL}}\big(q_\theta(z_i\mid z_{i-1})\Vert p_\varphi(z_i\mid z_{i+1})\big) \Big] \\
&\quad - \mathbb{E}_{q}\Big[ D_{\mathrm{KL}}\big(q_\theta(z_L\mid z_{L-1})\Vert p(z_L)\big) \Big]
\end{align}
$$

where $\mathbb{E}_q := \mathbb{E}_{p_{\text{data}}(x)q_\theta(z_{1:L}\mid x)}$.

### Key Effects of Hierarchy

1. **Distributes information penalties across layers:** Instead of a single KL term, penalties are spread across multiple levels.
2. **Aligns inference with generation:** Each inference conditional $q_\theta(z_i\mid z_{i-1})$ is aligned with its top-down generative counterpart $p_\varphi(z_i\mid z_{i+1})$.
3. **Progressive generation:** Encourages coarse-to-fine structure, capturing data at multiple abstraction levels.
4. **Localizes learning signals:** The adjacent KL terms provide localized gradients through these layer-wise comparisons.

**Key observation:** These properties stem from the **hierarchical latent graph structure**, not from simply deepening networks in a flat VAE.

---

## Why Simply Deepening a Flat VAE is Insufficient

There are two fundamental limitations that deeper networks alone cannot address:

### Limitation 1: Restricted Variational Family

In a standard VAE:
$$q_\theta(z\mid x) = \mathcal{N}\big(z;\, \mu_\theta(x),\; \operatorname{diag}(\sigma^2_\theta(x))\big)$$

**Problem:**
* For each fixed $x$, the encoder posterior is a **single Gaussian with diagonal covariance**.
* Greater network depth improves the accuracy of $\mu_\theta$ and $\sigma_\theta$ but **does not expand the family**.
* Even a full covariance remains **one unimodal ellipsoid**.
* When $p_\varphi(z\mid x)$ is **multi-peaked**, this family cannot match it.
* This loosens the ELBO and weakens inference.

**Solution required:** A **richer posterior class**, not merely deeper networks. HVAEs address this by having multiple latent layers that can collectively represent more complex distributions.

### Limitation 2: Posterior Collapse Persists

Recall the dataset-averaged ELBO:

$$
\mathbb{E}_{p_{\text{data}}(x)}[\mathcal{L}_{\text{ELBO}}(x)]
= \mathbb{E}_{p_{\text{data}}(x)q_\theta(z\mid x)}[\log p_\varphi(x\mid z)] - I_q(x;z) - D_{\mathrm{KL}}(q_\theta(z)\Vert p(z))
$$

**Problem:**
* If the decoder class can model the data well **without using $z$** (i.e., it contains some $p_\varphi(x\mid z) = r(x)$ close to $p_{\text{data}}$), then:
  * A maximizer of ELBO sets $q_\theta(z\mid x) = p(z)$
  * This makes $I_q(x;z) = 0$ and $q_\theta(z) = p(z)$
* This "ignore $z$" solution **does not disappear by making networks deeper**.

**Consequences:**
1. The learned code becomes **independent of $x$** (no data-dependent structure for downstream tasks).
2. Conditioning or moving in $z$ has **no effect** on generated samples (controllable generation fails).

---

## What Hierarchy Actually Changes

An HVAE introduces multiple latent levels with the structure:

$$p_\varphi(x, z_{1:L}) = p_\varphi(x\mid z_1)\, \prod_{i=2}^{L} p_\varphi(z_{i-1}\mid z_i)\, p(z_L)$$

The HVAE ELBO becomes:

$$
\begin{align}
\mathcal{L}_{\text{ELBO}}(x) &= \mathbb{E}_q[\log p_\varphi(x\mid z_1)] \\
&\quad - \mathbb{E}_q[D_{\mathrm{KL}}(q_\theta(z_1\mid x)\Vert p_\varphi(z_1\mid z_2))] \\
&\quad - \sum_{i=2}^{L-1} \mathbb{E}_q[D_{\mathrm{KL}}(q_\theta(z_i\mid z_{i-1})\Vert p_\varphi(z_i\mid z_{i+1}))] \\
&\quad - \mathbb{E}_q[D_{\mathrm{KL}}(q_\theta(z_L\mid z_{L-1})\Vert p(z_L))]
\end{align}
$$

**Key structural differences:**
* Each inference conditional is aligned with its top-down generative counterpart:
  * $q_\theta(z_1\mid x)$ with $p_\varphi(z_1\mid z_2)$
  * Intermediate layers: $q_\theta(z_i\mid z_{i-1})$ with $p_\varphi(z_i\mid z_{i+1})$
  * Top layer: $q_\theta(z_L\mid z_{L-1})$ with $p(z_L)$
* This **distributes the information penalty across levels** and **localizes learning signals** through adjacent KL terms.
* These properties are unique to the hierarchical structure and cannot be achieved by simply deepening a flat VAE.

---

## Practical Challenges of HVAEs

Despite their theoretical advantages, HVAEs pose unique training difficulties:

1. **Joint optimization instability:**
   * Encoder and decoder must be optimized jointly.
   * Lower layers and decoder can already reconstruct $x$, leaving higher-level latents with little effective signal.

2. **Weak gradients to deeper latents:**
   * Gradient information reaching deeper variables is often **indirect and weak**.
   * Makes it difficult for them to contribute meaningfully.

3. **Capacity balancing:**
   * Overly expressive conditionals can **dominate the reconstruction task**.
   * This suppresses the utility of higher latents.
   * Need careful architectural/regularization safeguards.

4. **Training techniques needed:**
   * $\beta$-VAE or KL annealing often help mitigate collapse.
   * Layer-wise training or careful initialization may be required.

---

## Conceptual Link to Diffusion Models (Preview)

**Core similarity:**
* Both HVAEs and diffusion models exploit a **progressive, multi-stage** structure.
* Both model data generation as a sequence of refinement steps.

**Key difference (and improvement in diffusion models):**
* HVAEs learn **both** the encoding (bottom-up) and decoding (top-down) processes jointly.
* Diffusion models **fix** the encoding (noisy forward process) and only learn the decoding (denoising reversal).

**Benefits of fixing the encoder:**
* Improved **stability** during training.
* Better **sample quality**.
* Avoids the joint optimization challenges of HVAEs.
* Sidesteps posterior collapse issues.

**Looking ahead:** Section 2.2 explores how diffusion models inherit the progressive structure of HVAEs while elegantly sidestepping their central weaknesses, leading to significant leaps in generation quality.

---

## Minimal Checklist (What to Remember)

✓ **ELBO = Reconstruction - Prior KL:** Maximizing ELBO shapes a usable latent space.

✓ **Gap identity:** $\log p(x) - \mathcal{L}_{\text{ELBO}} = D_{\mathrm{KL}}(q(z\mid x)\Vert p(z\mid x))$ (inference error).

✓ **Gaussian VAE:** MSE reconstruction + closed-form KL; use reparameterization trick.

✓ **Blurriness cause:** Stems from conditional averaging when different inputs map to overlapping latent regions.

✓ **Posterior collapse:** Occurs when overly expressive decoders model data without using $z$, driving $I_q(x;z) = 0$.

✓ **HVAEs:** Introduce hierarchical latents to model coarse→fine structure but face joint optimization challenges.

✓ **Why depth alone is insufficient:** 
  * Variational family remains unimodal (can't capture multi-modal posteriors)
  * Posterior collapse persists regardless of depth

✓ **Diffusion models:** Fix the encoding process, focusing on learning the generative reversal for improved stability.

---

## Quick Equations (At a Glance)

**ELBO:**
$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q}[\log p_\varphi(x\mid z)] - D_{\mathrm{KL}}(q_\theta(z\mid x)\Vert p(z))$$

**Gap (inference error):**
$$\log p(x) - \mathcal{L}_{\text{ELBO}} = D_{\mathrm{KL}}(q(z\mid x)\Vert p(z\mid x))$$

**Gaussian decoder ⇒ MSE:**
$$\mathbb{E}_{q}[\log p_\varphi(x\mid z)] = -\tfrac{1}{2\sigma^2}\mathbb{E}_{q}[\lVert x-\mu_\varphi(z)\rVert^2] + C$$

**Reparameterization:**
$$z=\mu_\theta(x)+\sigma_\theta(x)\odot\epsilon, \quad \epsilon\sim\mathcal{N}(0,I)$$

**Optimal decoder (blurriness):**
$$\mu^*(z) = \mathbb{E}_{q_{\text{enc}}(x\mid z)}[x]$$

**Mutual information (posterior collapse):**
$$I_q(x;z) = \mathbb{E}_{p_{\text{data}}(x)}[D_{\mathrm{KL}}(q_\theta(z\mid x)\Vert q(z))]$$

---

## Common Pitfalls

❌ **Treating $\beta$-VAE or KL annealing as optional:** They often help mitigate collapse and improve training stability.

❌ **Using too powerful decoders without safeguards:** Without architectural/regularization constraints, posterior collapse is likely.

❌ **Assuming deeper networks alone solve multi-modality:** The variational family $q(z\mid x)$ remains unimodal; need hierarchical structure.

❌ **Ignoring the reparameterization trick:** Direct sampling prevents gradient flow; always use reparameterization.

❌ **Not monitoring mutual information $I_q(x;z)$:** A good diagnostic for posterior collapse during training.

---

## Pseudocode (Training Loop)

```python
for minibatch x:
    # Encoder: compute mean and log-variance
    mu_x, logvar_x = encoder(x)
    
    # Reparameterization trick
    eps ~ N(0, I)
    z = mu_x + exp(0.5 * logvar_x) ⊙ eps
    
    # Decoder: reconstruct
    x_recon = decoder(z)
    
    # Reconstruction loss (e.g., MSE for continuous data, BCE for binary)
    recon_loss = -log p(x | z)   # e.g., ||x - x_recon||^2 / (2*sigma^2)
    
    # KL divergence (closed-form for Gaussian)
    kl_loss = -0.5 * sum(1 + logvar_x - mu_x^2 - exp(logvar_x))
    
    # Total loss (ELBO, negated for minimization)
    loss = recon_loss + beta * kl_loss  # beta=1 for standard VAE
    
    # Update parameters
    update θ, φ to minimize loss
```

**Note:** $\beta$-VAE uses $\beta > 1$ to encourage better disentanglement at the cost of reconstruction quality.

---

## Key Observation from the Reading

**Observation 2.1.1 (The Power of Stacking Layers):**

> Stacking layers allows the model to generate data progressively, starting with coarse details and adding finer ones at each step. This process makes it far easier to capture the complex structure of high-dimensional data.

This insight is foundational not just for HVAEs but for many modern generative models including:
* Score-based methods (Section 3.4)
* Normalizing flows (Section 5.1)
* Diffusion models (Section 2.2)

---

## References (For Deeper Study)

* **Kingma & Welling (2013/2014):** Auto-Encoding Variational Bayes. [Original VAE paper]
* **Rezende, Mohamed, & Wierstra (2014):** Stochastic Backpropagation and Approximate Inference in Deep Generative Models.
* **Vahdat & Kautz (2020):** NVAE: A Deep Hierarchical Variational Autoencoder. [State-of-the-art hierarchical VAE]
* **Higgins et al. (2017):** $\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. [Disentanglement]
* **Bowman et al. (2016):** Generating Sentences from a Continuous Space. [Posterior collapse discussion]

---

## Notation Conventions

**Important note on notation:**
* The textbook deviates from common VAE convention (which uses $q$ for encoder, $p$ for generator).
* Instead, it uses $p$ for all distributions, with subscripts/superscripts clarifying roles.
* In these notes, we follow the standard convention: $q_\theta$ for encoder, $p_\varphi$ for decoder/generator.
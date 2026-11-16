# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Yutaka Shimizu's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/purewater0901.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Apply One Policy-Value Head Training Step | Update linear policy and value heads in PyTorch from fixed features, search-policy targets, and bounded outcome targets. | https://www.tensortonic.com/problems/alphago-l05-policy-value-training-step |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/problems/cs336-l02-einsum-attention-scores |
| Causal Grouped-Query Attention | Compute causal scaled dot-product attention in which each contiguous group of query heads shares one key/value head. | https://www.tensortonic.com/problems/cs336-l03-causal-grouped-query-attention |
| Rotary Query and Key Embeddings | Rotate each adjacent coordinate pair of query and key vectors by a position-dependent angle. | https://www.tensortonic.com/problems/cs336-l03-rotary-query-key-embeddings |
| Decode Attention with a KV Cache | Append one key and value position to a KV cache and compute grouped-query attention for the next token. | https://www.tensortonic.com/problems/cs336-l10-kv-cache-decode-step |
| Transformer Decoder Block | Implement a NumPy Transformer decoder block with masked self-attention, cross-attention, and feed-forward layers. | https://www.tensortonic.com/problems/dl-decoder-block |
| Transformer Encoder Block | Implement a NumPy Transformer encoder block with self-attention, residual connections, normalization, and feed-forward layers. | https://www.tensortonic.com/problems/dl-encoder-block |
| Vision Transformer Patch Embedding | Convert an image into non-overlapping Vision Transformer patch embeddings with a learned linear projection. | https://www.tensortonic.com/problems/dl-vision-transformer |
| ε-Greedy Action Selection | Select a reinforcement-learning action with epsilon-greedy exploration using action values and controlled randomness. | https://www.tensortonic.com/problems/epsilon-greedy |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Apply 4×4 Homogeneous Transform | Apply a 4x4 homogeneous transformation matrix to 3D points using rotation, translation, and homogeneous coordinates. | https://www.tensortonic.com/problems/homogeneous-transform |
| Implement Autoregressive Decoding with a KV Cache | Decode autoregressively with an append-only key-value cache so previously processed tokens are never recomputed. | https://www.tensortonic.com/problems/inference-autoregressive-kv-cache |
| Implement Grouped-Query Attention (GQA) | Implement grouped-query attention by mapping query heads onto fewer key-value heads with validated head divisibility. | https://www.tensortonic.com/problems/inference-grouped-query-attention |
| Calculate KV Cache Memory for MHA, MQA, GQA, and MLA | Compute the total KV-cache memory, in bytes, for a full sequence under four attention variants: MHA, MQA, GQA, and MLA. | https://www.tensortonic.com/problems/inference-kv-cache-memory |
| Implement MoE Token Dispatch and Expert Aggregation | Run each expert's feed-forward computation only on the tokens routed to it, and scatter-add the routing-weighted results back into token order. | https://www.tensortonic.com/problems/inference-moe-dispatch-aggregation |
| K-Means Clustering | Implement K-means clustering with nearest-centroid assignments, centroid updates, convergence checks, and stable labels. | https://www.tensortonic.com/problems/ml-kmeans |
| Policy Gradient Loss | Compute policy-gradient loss from selected action probabilities and advantage estimates with stable logarithms. | https://www.tensortonic.com/problems/policy-gradient-loss |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions, with the formula and tests in the browser. | https://www.tensortonic.com/problems/positional-encoding |
| Tabular Q-Learning (Single Update) | Perform one tabular Q-learning update from reward, discount, learning rate, and the best next-state value. | https://www.tensortonic.com/problems/q-learning-update |
| Expected SARSA | Apply Expected SARSA updates using the epsilon-greedy expectation over next-state action values. | https://www.tensortonic.com/problems/rl-expected-sarsa |
| Generalized Advantage Estimation (GAE) | Compute generalized advantage estimates by recursively combining rewards, value predictions, discounts, and GAE decay. | https://www.tensortonic.com/problems/rl-generalized-advantage-estimation |
| GRPO Group-Relative Advantage | Normalize rewards within each GRPO group using its mean and standard deviation, including zero-variance handling. | https://www.tensortonic.com/problems/rl-grpo-group-relative-advantage |
| Monte Carlo Policy Evaluation | Estimate policy state values from complete sampled returns without requiring an environment transition model. | https://www.tensortonic.com/problems/rl-monte-carlo-policy-evaluation |
| Policy Iteration | Policy iteration solves a finite Markov Decision Process by alternating two steps until the greedy policy stops changing. | https://www.tensortonic.com/problems/rl-policy-iteration |
| REINFORCE Gradient | Implement REINFORCE Gradient, and return the scalar loss as a float rounded to 4 decimals. | https://www.tensortonic.com/problems/rl-reinforce-gradient |
| REINFORCE with Baseline | Implement REINFORCE with Baseline, and return the scalar loss as a Python float, rounded to 4 decimals. | https://www.tensortonic.com/problems/rl-reinforce-with-baseline |
| PPO with Reference-Policy KL | Compute a PPO policy objective with reference-policy KL regularization for stable RLHF optimization. | https://www.tensortonic.com/problems/rl-rlhf-ppo-kl-loss |
| Softmax (Boltzmann) Exploration | Convert action values into a numerically stable Boltzmann exploration distribution controlled by temperature. | https://www.tensortonic.com/problems/rl-softmax-exploration |
| Value Iteration | Run value iteration to convergence for a finite MDP, then extract the greedy optimal policy with stable tie handling. | https://www.tensortonic.com/problems/rl-value-iteration |
| SARSA Update | Perform one on-policy SARSA action-value update from the observed reward and next selected action. | https://www.tensortonic.com/problems/sarsa-update |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement a Simple CNN Layer (NumPy) | Implement a NumPy CNN layer forward pass with batched valid convolution across channels and bias addition. | https://www.tensortonic.com/problems/simple-cnn-layer |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Token and Learned Position Embeddings | Build GPT-2 input representations by summing learned token embeddings with learned positional embeddings. | https://www.tensortonic.com/research/gpt2/gpt2-embedding |
| Greedy Autoregressive Decoding | Implement GPT-2 greedy decoding by repeatedly selecting the highest-logit next token from the model output. | https://www.tensortonic.com/research/gpt2/gpt2-greedy-decode |
| Temperature and Top-k Sampling | Implement GPT-2 top-k sampling with temperature scaling, probability normalization, and seeded token selection. | https://www.tensortonic.com/research/gpt2/gpt2-topk-sampling |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/purewater0901)
<!-- tensortonic:end -->

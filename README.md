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
| ε-Greedy Action Selection | Select a reinforcement-learning action with epsilon-greedy exploration using action values and controlled randomness. | https://www.tensortonic.com/problems/epsilon-greedy |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Implement Grouped-Query Attention (GQA) | Implement grouped-query attention by mapping query heads onto fewer key-value heads with validated head divisibility. | https://www.tensortonic.com/problems/inference-grouped-query-attention |
| Policy Gradient Loss | Compute policy-gradient loss from selected action probabilities and advantage estimates with stable logarithms. | https://www.tensortonic.com/problems/policy-gradient-loss |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions, with the formula and tests in the browser. | https://www.tensortonic.com/problems/positional-encoding |
| Tabular Q-Learning (Single Update) | Perform one tabular Q-learning update from reward, discount, learning rate, and the best next-state value. | https://www.tensortonic.com/problems/q-learning-update |
| Expected SARSA | Apply Expected SARSA updates using the epsilon-greedy expectation over next-state action values. | https://www.tensortonic.com/problems/rl-expected-sarsa |
| GRPO Group-Relative Advantage | Normalize rewards within each GRPO group using its mean and standard deviation, including zero-variance handling. | https://www.tensortonic.com/problems/rl-grpo-group-relative-advantage |
| Monte Carlo Policy Evaluation | Estimate policy state values from complete sampled returns without requiring an environment transition model. | https://www.tensortonic.com/problems/rl-monte-carlo-policy-evaluation |
| REINFORCE with Baseline | Implement REINFORCE with Baseline, and return the scalar loss as a Python float, rounded to 4 decimals. | https://www.tensortonic.com/problems/rl-reinforce-with-baseline |
| Softmax (Boltzmann) Exploration | Convert action values into a numerically stable Boltzmann exploration distribution controlled by temperature. | https://www.tensortonic.com/problems/rl-softmax-exploration |
| SARSA Update | Perform one on-policy SARSA action-value update from the observed reward and next selected action. | https://www.tensortonic.com/problems/sarsa-update |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/purewater0901)
<!-- tensortonic:end -->

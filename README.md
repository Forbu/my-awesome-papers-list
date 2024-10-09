## My personal papers list 

[Deep learning general](#Deep-learning-general)

[Reinforcement learning](#Reinforcement-learning)

[Generative model](#Generative-model)

[LLMs](#LLMs)

[GNNs](#GNNs)

[DL for Physics](#DL-for-Physics)

[Symetry in NN](#Symetry-group-in-DL)

[Datasets](#Datasets)

[Reasoning](#Reasoning)

[Training Big Model](#Training-bigger-model)

[Deep potential Arc](#Deep-Potential)


### Deep learning general

- Loss of Plasticity in Deep Continual Learning : https://arxiv.org/abs/2306.13812
  
Discussion about the issue of loss of ability to learn in NN
New algorithm to improve continual learning 

- [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)
Interesting theorical approach that said that there is a lot of useless symetries in NN

- [End-to-end Algorithm Synthesis with Recurrent Networks: Logical...](https://arxiv.org/abs/2202.05826)
Using recurrent modeliing to do OOD generalization

- https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.5.043252
Interesting take on forecasting chaotic time series

- [An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162) Choosing the batch size (to read)

- [Learning to (Learn at Test Time):
RNNs with Expressive Hidden States](https://arxiv.org/pdf/2407.04620) : a new way to approach sequence problem with DL / ML

### Reinforcement learning 

- Sample-efficient reinforcement learningby breaking the replay ratio barrier [https://openreview.net/pdf?id=4GBGwVIEYJ](https://openreview.net/pdf?id=4GBGwVIEYJ "https://openreview.net/pdf?id=4GBGwVIEYJ")

-  Mastering Diverse Domains through World Models : dreamerv3 https://arxiv.org/pdf/2301.04104.pdf
Interesting Symlog Predictions for reward scaling

- [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/abs/2305.19452)
Extensive use of shrink and reset (continual learning)

- Decision diffuser (inverse RL with diffusion) :
https://arxiv.org/pdf/2211.15657.pdf
Take into acount the stochastic RL env

- DO TRANSFORMER WORLD MODELS GIVE BETTER POLICY GRADIENTS?
https://arxiv.org/pdf/2402.05290.pdf
Good world model approach but a transformer achitecture for RL with world model.

- In the same transformer-like architecture : TRANSFORMER-BASED WORLD MODELS ARE HAPPY
WITH 100K INTERACTIONS : https://arxiv.org/pdf/2303.07109

- Ranked Reward: Enabling Self-Play Reinforcement
Learning for Combinatorial Optimization https://arxiv.org/pdf/1807.01672

### Generative model 

- MeshGPT : https://nihalsid.github.io/mesh-gpt/static/MeshGPT.pdf#page=9&zoom=100,412,724
  Similar to VAEGAN : first VAE then autoregressive sampling from latent space.

- Stretching Each Dollar: Diffusion Training from Scratch on
a Micro-Budget : https://arxiv.org/abs/2407.15811
Clever use of token masking to reduce training flops

- Movie Gen: A Cast of Media Foundation Models , a meta ai architecture for image/movie generation
https://ai.meta.com/static-resource/movie-gen-research-paper

**Matching flow papers** :

- https://arxiv.org/pdf/2210.02747.pdf
Simply explain matching flow for continuous variable
  
- https://arxiv.org/pdf/2302.00482.pdf matching flow with OT optim. Nice explaination of CMF with some OT extension.
  
- https://arxiv.org/pdf/2403.03206.pdf SB3 with matching flow. An example of CMF implementation.

- https://arxiv.org/pdf/2404.19739v1 MF with categorical variable

- https://arxiv.org/abs/2309.06380 : instaflow, "straighten" flow trajectories => faster inference

- https://arxiv.org/abs/2209.03003 : Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, improve flow speed

- https://arxiv.org/pdf/2405.20320 : improved maching flow training

- https://arxiv.org/pdf/2407.15595 : discret matching flow for language generation

**Graph Generation**

- https://arxiv.org/pdf/2312.11529.pdf
Graph generation : multi level graph generation thanks to Coarsening

**Generative other**

- https://arxiv.org/pdf/2404.09562 : sigma GPT autoregressive but with permutation training (allow better sampling). 



### LLMs 

- Reinforced Self-Training (ReST) for Language Modeling : https://arxiv.org/pdf/2308.08998.pdf
Methodo to improve LLM if you have a reward function at your disposal

- [Direct Preference Optimization: Your Language Model is Secretly a R...](https://arxiv.org/abs/2305.18290)
A good preference optimization scheme (if you have a preference dataset)

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) How to make your LLM moral version Anthropic AI 

- Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences :
		https://arxiv.org/abs/2404.03715

- https://arxiv.org/pdf/2309.07864.pdf#page=67&zoom=100,144,670
  toolformer, LLM with tools

- https://arxiv.org/pdf/2306.00637.pdf
Stable cascade : a efficient way to train image generation with NLP instruct

- https://arxiv.org/abs/2405.12130 : MoRa, better efficient finetuning than LoRa (higher memory capacity)

- Finetuning LLMs : https://arxiv.org/pdf/2402.02868 Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation
Problem

- How to create synthetic data for training : https://www.youtube.com/watch?v=yBL7J0kgldU

### GNNs 

- https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf (EXPHORMER: Sparse Transformers for Graphs)

### DL for Physics

- LAGRANGIAN NEURAL NETWORKS : enforce conservation law with a special loss (https://arxiv.org/pdf/2003.04630)

- Quantum simulation : https://arxiv.org/pdf/2308.16848

- Neural Network Potentials: A Concise Overview of Methods: https://arxiv.org/pdf/2107.03727

- PhAST: Physics-Aware, Scalable, and Task-Specific GNNs for
Accelerated Catalyst Design https://jmlr.org/papers/volume25/23-0680/23-0680.pdf

- FLOW MATCHING FOR ACCELERATED SIMULATION OF ATOMIC TRANSPORT IN MATERIALS (https://arxiv.org/pdf/2410.01464)



### Symetry group in DL

- https://arxiv.org/pdf/2203.06153
Symmetry Group Equivariant Architectures for Physics

- https://arxiv.org/pdf/1802.08219
  Equivariant learning (just use sperical harmonics)

- https://arxiv.org/abs/2006.10503
  SE(3) transformer, include invariance in transformer to accelerate training.

- Harmonics of Learning:
Universal Fourier Features
Emerge in Invariant Networks
https://arxiv.org/pdf/2312.08550

### Datasets

- 3D objects dataset : https://arxiv.org/pdf/2307.05663

### Constraints into Neural Net

- https://arxiv.org/pdf/2403.14404 : physics informed generative modelling

- https://arxiv.org/pdf/2402.14009 : geometric informed generative modeling

### Reasoning

- https://arxiv.org/pdf/2406.11179 : Reasoning with diffusion and energy based model

- https://arxiv.org/pdf/2409.12917 Training Language Models to Self-Correct via
Reinforcement Learning improving reasoning in LLM

- Scaling LLM Test-Time Compute Optimally can
be More Effective than Scaling Model Parameters : https://arxiv.org/pdf/2408.03314

### Training bigger model

- https://arxiv.org/pdf/2203.03466 : Tensor Programs V:
Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer

The idea is to tune an HP (like lr) with a small model and then we can transfert this HP to a bigger model without tuning specific this bigger model.

- https://arxiv.org/abs/2310.17813 : "A Spectral Condition for Feature Learning" instead of muP we simply a spectral normalization 

- https://blog.eleuther.ai/mutransfer/ good summarized on why and how to implement muP parametrization

### Other

- https://proceedings.mlr.press/v206/bertrand23a/bertrand23a.pdf
Other thing than elo to rank player

- Quantum physics a theorical minimum : [pdf](https://github.com/markf94/QML_Thesis/blob/master/Books_and_Resources/Quantum%20Mechanics%20-%20The%20Theoretical%20Minimum.pdf)

### Deep Potential

Current reading on deep potential :

- https://arxiv.org/pdf/2203.00393 (survey paper)

- https://arxiv.org/pdf/1707.01478 (Deep Potential: a general representation of a many-body
potential energy surface)

- https://arxiv.org/pdf/1707.09571 (Deep Potential Molecular Dynamics: a scalable model with the
accuracy of quantum mechanics)

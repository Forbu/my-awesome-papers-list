## My personal papers list 

[Deep learning general](#Deep-learning-general)

[Reinforcement learning](#Reinforcement-learning)

[Generative model](#Generative-model)

[LLMs](#LLMs)

[GNNs](#GNNs)

[Symetry in NN](#Symetry-group-in-DL)

[Datasets](#Datasets)

[Reasoning](#Reasoning)


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

### Generative model 

- MeshGPT : https://nihalsid.github.io/mesh-gpt/static/MeshGPT.pdf#page=9&zoom=100,412,724
  Similar to VAEGAN : first VAE then autoregressive sampling from latent space.

**Machting flow papers** :

- https://arxiv.org/pdf/2210.02747.pdf
Simply explain matching flow for continuous variable
  
- https://arxiv.org/pdf/2302.00482.pdf matching flow with OT optim. Nice explaination of CMF with some OT extension.
  
- https://arxiv.org/pdf/2403.03206.pdf SB3 with matching flow. An example of CMF implementation.

- https://arxiv.org/pdf/2404.19739v1 MF with categorical variable

- https://arxiv.org/abs/2309.06380 : instaflow, "straighten" flow trajectories => faster inference

- https://arxiv.org/pdf/2405.20320 : improved maching flow training

**Graph Generation**

- https://arxiv.org/pdf/2312.11529.pdf
Graph generation : multi level graph generation thanks to Coarsening

**Generatiev other**

- https://arxiv.org/pdf/2404.09562 : sigma GPT autoregressive but with permutation training (allow better sampling). 



### LLMs 

- Reinforced Self-Training (ReST) for Language Modeling : https://arxiv.org/pdf/2308.08998.pdf
Methodo to improve LLM if you have a reward function at your disposal

- [Direct Preference Optimization: Your Language Model is Secretly a R...](https://arxiv.org/abs/2305.18290)
A good preference optimization scheme (if you have a preference dataset)

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

- Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences :
		https://arxiv.org/abs/2404.03715

- https://arxiv.org/pdf/2309.07864.pdf#page=67&zoom=100,144,670
  toolformer, LLM with tools

- https://arxiv.org/pdf/2306.00637.pdf
Stable cascade : a efficient way to train image generation with NLP instruct

- https://arxiv.org/abs/2405.12130 : MoRa, better efficient finetuning than LoRa (higher memory capacity)

### GNNs 

- https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf (EXPHORMER: Sparse Transformers for Graphs)

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

### Other

- https://proceedings.mlr.press/v206/bertrand23a/bertrand23a.pdf
Other thing than elo to rank player


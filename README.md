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

[Transformer](#Transformer-architecure)

[Training Big Model](#Training-bigger-model)

[Deep potential Arc](#Deep-Potential)


### Deep learning general

- Loss of Plasticity in Deep Continual Learning : [![arXiv](https://img.shields.io/badge/arXiv-2306.13812-b31b1b.svg)](https://arxiv.org/abs/2306.13812)
  
Discussion about the issue of loss of ability to learn in NN
New algorithm to improve continual learning 

- [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)
[![arXiv](https://img.shields.io/badge/arXiv-2209.04836-b31b1b.svg)](https://arxiv.org/abs/2209.04836)
Interesting theorical approach that said that there is a lot of useless symetries in NN

- [End-to-end Algorithm Synthesis with Recurrent Networks: Logical...](https://arxiv.org/abs/2202.05826)
[![arXiv](https://img.shields.io/badge/arXiv-2202.05826-b31b1b.svg)](https://arxiv.org/abs/2202.05826)
Using recurrent modeliing to do OOD generalization

- https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.5.043252
Interesting take on forecasting chaotic time series

- [An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162)
[![arXiv](https://img.shields.io/badge/arXiv-1812.06162-b31b1b.svg)](https://arxiv.org/pdf/1812.06162) Choosing the batch size (to read)

- TABM: ADVANCING TABULAR DEEP LEARNING WITH PARAMETER-EFFICIENT ENSEMBLING (tabular data model) :
  https://arxiv.org/pdf/2410.24210


### Reinforcement learning 

- Sample-efficient reinforcement learningby breaking the replay ratio barrier [https://openreview.net/pdf?id=4GBGwVIEYJ](https://openreview.net/pdf?id=4GBGwVIEYJ "https://openreview.net/pdf?id=4GBGwVIEYJ")

-  Mastering Diverse Domains through World Models : dreamerv3 https://arxiv.org/pdf/2301.04104.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2301.04104-b31b1b.svg)](https://arxiv.org/pdf/2301.04104.pdf)
Interesting Symlog Predictions for reward scaling

- [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/abs/2305.19452)
[![arXiv](https://img.shields.io/badge/arXiv-2305.19452-b31b1b.svg)](https://arxiv.org/abs/2305.19452)
Extensive use of shrink and reset (continual learning)

- Decision diffuser (inverse RL with diffusion) :
https://arxiv.org/pdf/2211.15657.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2211.15657-b31b1b.svg)](https://arxiv.org/pdf/2211.15657.pdf)
Take into acount the stochastic RL env

- DO TRANSFORMER WORLD MODELS GIVE BETTER POLICY GRADIENTS?
https://arxiv.org/pdf/2402.05290.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2402.05290-b31b1b.svg)](https://arxiv.org/pdf/2402.05290.pdf)
Good world model approach but a transformer achitecture for RL with world model.

- In the same transformer-like architecture : TRANSFORMER-BASED WORLD MODELS ARE HAPPY
WITH 100K INTERACTIONS : https://arxiv.org/pdf/2303.07109
[![arXiv](https://img.shields.io/badge/arXiv-2303.07109-b31b1b.svg)](https://arxiv.org/pdf/2303.07109)

- Ranked Reward: Enabling Self-Play Reinforcement
Learning for Combinatorial Optimization https://arxiv.org/pdf/1807.01672
[![arXiv](https://img.shields.io/badge/arXiv-1807.01672-b31b1b.svg)](https://arxiv.org/pdf/1807.01672)

- Embodied Intelligence
Through World Models : https://tspace.library.utoronto.ca/bitstream/1807/140956/2/Hafner_Danijar_202411_PhD_thesis.pdf
A nice summary of world model technics from Danijar Hafner

- OpenAI o1 algorithm : https://arxiv.org/pdf/2310.04363
[![arXiv](https://img.shields.io/badge/arXiv-2310.04363-b31b1b.svg)](https://arxiv.org/pdf/2310.04363) (AMORTIZING INTRACTABLE INFERENCE
IN LARGE LANGUAGE MODELS)

- RL for LLM (deepseek R1) : https://arxiv.org/abs/2402.03300
[![arXiv](https://img.shields.io/badge/arXiv-2402.03300-b31b1b.svg)](https://arxiv.org/abs/2402.03300) (group PPO)

- ppo implementation detail https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

- https://rlhfbook.com/ a big pdf on a lot of technics around RL in NLP

- https://dapo-sia.github.io/static/pdf/dapo_paper.pdf improved GRPO (DAPO)

### Generative model 

- MeshGPT : https://nihalsid.github.io/mesh-gpt/static/MeshGPT.pdf#page=9&zoom=100,412,724
  Similar to VAEGAN : first VAE then autoregressive sampling from latent space.

- Stretching Each Dollar: Diffusion Training from Scratch on
a Micro-Budget : https://arxiv.org/abs/2407.15811
[![arXiv](https://img.shields.io/badge/arXiv-2407.15811-b31b1b.svg)](https://arxiv.org/abs/2407.15811)
Clever use of token masking to reduce training flops

- Movie Gen: A Cast of Media Foundation Models , a meta ai architecture for image/movie generation
https://ai.meta.com/static-resource/movie-gen-research-paper

**Matching flow papers** :

- https://arxiv.org/pdf/2210.02747.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2210.02747-b31b1b.svg)](https://arxiv.org/pdf/2210.02747.pdf)
Simply explain matching flow for continuous variable
  
- https://arxiv.org/pdf/2302.00482.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2302.00482-b31b1b.svg)](https://arxiv.org/pdf/2302.00482.pdf) matching flow with OT optim. Nice explaination of CMF with some OT extension.
  
- https://arxiv.org/pdf/2403.03206.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2403.03206-b31b1b.svg)](https://arxiv.org/pdf/2403.03206.pdf) SB3 with matching flow. An example of CMF implementation.

- https://arxiv.org/pdf/2404.19739v1
[![arXiv](https://img.shields.io/badge/arXiv-2404.19739v1-b31b1b.svg)](https://arxiv.org/pdf/2404.19739v1) MF with categorical variable

- https://arxiv.org/abs/2309.06380
[![arXiv](https://img.shields.io/badge/arXiv-2309.06380-b31b1b.svg)](https://arxiv.org/abs/2309.06380) : instaflow, "straighten" flow trajectories => faster inference

- https://arxiv.org/abs/2209.03003
[![arXiv](https://img.shields.io/badge/arXiv-2209.03003-b31b1b.svg)](https://arxiv.org/abs/2209.03003) : Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, improve flow speed

- https://arxiv.org/pdf/2405.20320
[![arXiv](https://img.shields.io/badge/arXiv-2405.20320-b31b1b.svg)](https://arxiv.org/pdf/2405.20320) : improved maching flow training

- One Step Diffusion via Shortcut Models another (better) improved speed matching flow (https://arxiv.org/abs/2410.12557)
[![arXiv](https://img.shields.io/badge/arXiv-2410.12557-b31b1b.svg)](https://arxiv.org/abs/2410.12557)

- https://arxiv.org/pdf/2407.15595
[![arXiv](https://img.shields.io/badge/arXiv-2407.15595-b31b1b.svg)](https://arxiv.org/pdf/2407.15595) : discret matching flow for language generation

- Flow Matching Guide and Code : a nice summary from meta https://arxiv.org/pdf/2412.06264
[![arXiv](https://img.shields.io/badge/arXiv-2412.06264-b31b1b.svg)](https://arxiv.org/pdf/2412.06264)

- A General Framework for Inference-time Scaling and
Steering of Diffusion Models (refine the generation at test time to match reward model) https://arxiv.org/pdf/2501.06848
[![arXiv](https://img.shields.io/badge/arXiv-2501.06848-b31b1b.svg)](https://arxiv.org/pdf/2501.06848)

- Decentralized Diffusion Models : https://arxiv.org/pdf/2501.05450
[![arXiv](https://img.shields.io/badge/arXiv-2501.05450-b31b1b.svg)](https://arxiv.org/pdf/2501.05450) (nice setup for decentralized training)

- https://arxiv.org/pdf/2502.09616
[![arXiv](https://img.shields.io/badge/arXiv-2502.09616-b31b1b.svg)](https://arxiv.org/pdf/2502.09616) : better rectified flow with Variational Rectified Flow Matching

- Rolling Diffusion Models : https://arxiv.org/abs/2402.09470 improved diffusion for temporal-like data

- Gaussian Mixture Flow Matching Models https://arxiv.org/pdf/2504.05304

- Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective : https://arxiv.org/abs/2412.03487
 


**Generative other**

- https://arxiv.org/pdf/2404.09562
[![arXiv](https://img.shields.io/badge/arXiv-2404.09562-b31b1b.svg)](https://arxiv.org/pdf/2404.09562) : sigma GPT autoregressive but with permutation training (allow better sampling). 



### LLMs 

- Reinforced Self-Training (ReST) for Language Modeling : https://arxiv.org/pdf/2308.08998.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2308.08998-b31b1b.svg)](https://arxiv.org/pdf/2308.08998.pdf)
Methodo to improve LLM if you have a reward function at your disposal

- [Direct Preference Optimization: Your Language Model is Secretly a R...](https://arxiv.org/abs/2305.18290)
[![arXiv](https://img.shields.io/badge/arXiv-2305.18290-b31b1b.svg)](https://arxiv.org/abs/2305.18290)
A good preference optimization scheme (if you have a preference dataset)

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
[![arXiv](https://img.shields.io/badge/arXiv-2212.08073-b31b1b.svg)](https://arxiv.org/abs/2212.08073) How to make your LLM moral version Anthropic AI 

- Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences :
		https://arxiv.org/abs/2404.03715
[![arXiv](https://img.shields.io/badge/arXiv-2404.03715-b31b1b.svg)](https://arxiv.org/abs/2404.03715)

- https://arxiv.org/pdf/2309.07864.pdf#page=67&zoom=100,144,670
[![arXiv](https://img.shields.io/badge/arXiv-2309.07864-b31b1b.svg)](https://arxiv.org/pdf/2309.07864.pdf#page=67&zoom=100,144,670)
  toolformer, LLM with tools

- https://arxiv.org/pdf/2306.00637.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2306.00637-b31b1b.svg)](https://arxiv.org/pdf/2306.00637.pdf)
Stable cascade : a efficient way to train image generation with NLP instruct

- https://arxiv.org/abs/2405.12130
[![arXiv](https://img.shields.io/badge/arXiv-2405.12130-b31b1b.svg)](https://arxiv.org/abs/2405.12130) : MoRa, better efficient finetuning than LoRa (higher memory capacity)

- Finetuning LLMs : https://arxiv.org/pdf/2402.02868
[![arXiv](https://img.shields.io/badge/arXiv-2402.02868-b31b1b.svg)](https://arxiv.org/pdf/2402.02868) Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation
Problem

- How to create synthetic data for training : https://www.youtube.com/watch?v=yBL7J0kgldU

- GaLore : full finetune LLM as a lower cost, Memory-Efficient LLM Training by Gradient Low-Rank Projection 
https://arxiv.org/pdf/2403.03507
[![arXiv](https://img.shields.io/badge/arXiv-2403.03507-b31b1b.svg)](https://arxiv.org/pdf/2403.03507)

- Agent Skill Acquisition for Large Language Models via CycleQD : https://arxiv.org/abs/2410.14735
[![arXiv](https://img.shields.io/badge/arXiv-2410.14735-b31b1b.svg)](https://arxiv.org/abs/2410.14735)
A new kind of optimization algorithm for LLM (with mutation and merge)

Chinese best AI :

- KIMI K1.5: SCALING REINFORCEMENT LEARNING WITH LLMS https://arxiv.org/abs/2501.12599
[![arXiv](https://img.shields.io/badge/arXiv-2501.12599-b31b1b.svg)](https://arxiv.org/abs/2501.12599) : 

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
https://arxiv.org/abs/2501.12948
[![arXiv](https://img.shields.io/badge/arXiv-2501.12948-b31b1b.svg)](https://arxiv.org/abs/2501.12948)

### GNNs 

- https://arxiv.org/pdf/2303.06147
[![arXiv](https://img.shields.io/badge/arXiv-2303.06147-b31b1b.svg)](https://arxiv.org/pdf/2303.06147) (EXPHORMER: Sparse Transformers for Graphs)
- https://arxiv.org/pdf/1905.11136
[![arXiv](https://img.shields.io/badge/arXiv-1905.11136-b31b1b.svg)](https://arxiv.org/pdf/1905.11136) (Provably Powerful Graph Networks :powerful 3-WL expressive)

**Graph Generation**

- https://arxiv.org/pdf/2312.11529.pdf
[![arXiv](https://img.shields.io/badge/arXiv-2312.11529-b31b1b.svg)](https://arxiv.org/pdf/2312.11529.pdf)
Graph generation : multi level graph generation thanks to Coarsening

- Graph Generative Pre-trained Transformer (https://arxiv.org/pdf/2501.01073)
[![arXiv](https://img.shields.io/badge/arXiv-2501.01073-b31b1b.svg)](https://arxiv.org/pdf/2501.01073)
Graph generation AND how to finetune the model using rejection sampling or RL


### DL for Physics

- LAGRANGIAN NEURAL NETWORKS : enforce conservation law with a special loss (https://arxiv.org/pdf/2003.04630)
[![arXiv](https://img.shields.io/badge/arXiv-2003.04630-b31b1b.svg)](https://arxiv.org/pdf/2003.04630)

- Quantum simulation : https://arxiv.org/pdf/2308.16848
[![arXiv](https://img.shields.io/badge/arXiv-2308.16848-b31b1b.svg)](https://arxiv.org/pdf/2308.16848)

- Neural Network Potentials: A Concise Overview of Methods: https://arxiv.org/pdf/2107.03727
[![arXiv](https://img.shields.io/badge/arXiv-2107.03727-b31b1b.svg)](https://arxiv.org/pdf/2107.03727)

- PhAST: Physics-Aware, Scalable, and Task-Specific GNNs for
Accelerated Catalyst Design https://jmlr.org/papers/volume25/23-0680/23-0680.pdf

- FLOW MATCHING FOR ACCELERATED SIMULATION OF ATOMIC TRANSPORT IN MATERIALS (https://arxiv.org/pdf/2410.01464)
[![arXiv](https://img.shields.io/badge/arXiv-2410.01464-b31b1b.svg)](https://arxiv.org/pdf/2410.01464)

- All-atom Diffusion Transformers: Unified
generative modelling of molecules and materials : https://arxiv.org/pdf/2503.03965

### Symetry group in DL

- https://arxiv.org/pdf/2203.06153
[![arXiv](https://img.shields.io/badge/arXiv-2203.06153-b31b1b.svg)](https://arxiv.org/pdf/2203.06153)
Symmetry Group Equivariant Architectures for Physics

- https://arxiv.org/pdf/1802.08219
[![arXiv](https://img.shields.io/badge/arXiv-1802.08219-b31b1b.svg)](https://arxiv.org/pdf/1802.08219)
  Equivariant learning (just use sperical harmonics)

- https://arxiv.org/abs/2006.10503
[![arXiv](https://img.shields.io/badge/arXiv-2006.10503-b31b1b.svg)](https://arxiv.org/abs/2006.10503)
  SE(3) transformer, include invariance in transformer to accelerate training.

- Harmonics of Learning:
Universal Fourier Features
Emerge in Invariant Networks
https://arxiv.org/pdf/2312.08550
[![arXiv](https://img.shields.io/badge/arXiv-2312.08550-b31b1b.svg)](https://arxiv.org/pdf/2312.08550)

### Datasets

- 3D objects dataset : https://arxiv.org/pdf/2307.05663
[![arXiv](https://img.shields.io/badge/arXiv-2307.05663-b31b1b.svg)](https://arxiv.org/pdf/2307.05663)

### Constraints into Neural Net

- https://arxiv.org/pdf/2403.14404
[![arXiv](https://img.shields.io/badge/arXiv-2403.14404-b31b1b.svg)](https://arxiv.org/pdf/2403.14404) : physics informed generative modelling

- https://arxiv.org/pdf/2402.14009
[![arXiv](https://img.shields.io/badge/arXiv-2402.14009-b31b1b.svg)](https://arxiv.org/pdf/2402.14009) : geometric informed generative modeling

### Reasoning

- https://arxiv.org/pdf/2406.11179
[![arXiv](https://img.shields.io/badge/arXiv-2406.11179-b31b1b.svg)](https://arxiv.org/pdf/2406.11179) : Reasoning with diffusion and energy based model

- https://arxiv.org/pdf/2409.12917
[![arXiv](https://img.shields.io/badge/arXiv-2409.12917-b31b1b.svg)](https://arxiv.org/pdf/2409.12917) Training Language Models to Self-Correct via
Reinforcement Learning improving reasoning in LLM

- Scaling LLM Test-Time Compute Optimally can
be More Effective than Scaling Model Parameters : https://arxiv.org/pdf/2408.03314
[![arXiv](https://img.shields.io/badge/arXiv-2408.03314-b31b1b.svg)](https://arxiv.org/pdf/2408.03314)

- Searching Latent Program Spaces (https://arxiv.org/pdf/2411.08706)
[![arXiv](https://img.shields.io/badge/arXiv-2411.08706-b31b1b.svg)](https://arxiv.org/pdf/2411.08706) optimization in the latent space to fit every program in the in context setup.

### Transformer architecture 

- MLM-U algorithm : better than next token prediction : The Factorization Curse: Which Tokens You Predict
Underlie the Reversal Curse and More https://arxiv.org/pdf/2406.05183
[![arXiv](https://img.shields.io/badge/arXiv-2406.05183-b31b1b.svg)](https://arxiv.org/pdf/2406.05183)

- Transformers Can Navigate Mazes With Multi-Step
Prediction : https://arxiv.org/pdf/2412.05117
[![arXiv](https://img.shields.io/badge/arXiv-2412.05117-b31b1b.svg)](https://arxiv.org/pdf/2412.05117) an interesting takes on transformer next token prediction vs masked prediction (multi step prediction enable planning not so for next token prediction)

- Large Language Diffusion Models
https://arxiv.org/abs/2502.09992
[![arXiv](https://img.shields.io/badge/arXiv-2502.09992-b31b1b.svg)](https://arxiv.org/abs/2502.09992) (seems to be working fine with just masking AND clever sampling technics) (llaDa)

- improved dLLM with RL : https://dllm-reasoning.github.io/

- REMOE: FULLY DIFFERENTIABLE MIXTURE-OFEXPERTS WITH RELU ROUTING : mixture of expert but better (https://arxiv.org/pdf/2412.14711)
[![arXiv](https://img.shields.io/badge/arXiv-2412.14711-b31b1b.svg)](https://arxiv.org/pdf/2412.14711)

- softmax issues : Scalable-Softmax Is Superior for Attention https://arxiv.org/pdf/2501.19399

### Training bigger model

- https://arxiv.org/pdf/2203.03466
[![arXiv](https://img.shields.io/badge/arXiv-2203.03466-b31b1b.svg)](https://arxiv.org/pdf/2203.03466) : Tensor Programs V:
Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer

The idea is to tune an HP (like lr) with a small model and then we can transfert this HP to a bigger model without tuning specific this bigger model.

- https://arxiv.org/abs/2310.17813
[![arXiv](https://img.shields.io/badge/arXiv-2310.17813-b31b1b.svg)](https://arxiv.org/abs/2310.17813) : "A Spectral Condition for Feature Learning" instead of muP we simply use spectral normalization 

- https://blog.eleuther.ai/mutransfer/ good summarized on why and how to implement muP parametrization

- https://huggingface.co/spaces/nanotron/ultrascale-playbook : playbook on the different parallelization setup

### Other

- https://proceedings.mlr.press/v206/bertrand23a/bertrand23a.pdf
Other thing than elo to rank player

- Quantum physics a theorical minimum : [pdf](https://github.com/markf94/QML_Thesis/blob/master/Books_and_Resources/Quantum%20Mechanics%20-%20The%20Theoretical%20Minimum.pdf)

### Deep Potential

Current reading on deep potential :

- https://arxiv.org/pdf/2203.00393
[![arXiv](https://img.shields.io/badge/arXiv-2203.00393-b31b1b.svg)](https://arxiv.org/pdf/2203.00393) (survey paper)

- https://arxiv.org/pdf/1707.01478
[![arXiv](https://img.shields.io/badge/arXiv-1707.01478-b31b1b.svg)](https://arxiv.org/pdf/1707.01478) (Deep Potential: a general representation of a many-body
potential energy surface)

- https://arxiv.org/pdf/1707.09571
[![arXiv](https://img.shields.io/badge/arXiv-1707.09571-b31b1b.svg)](https://arxiv.org/pdf/1707.09571) (Deep Potential Molecular Dynamics: a scalable model with the
accuracy of quantum mechanics)

## video generation

- One-Minute Video Generation with Test-Time Training https://arxiv.org/pdf/2504.05298

- Seaweed-7B: Cost-Effective Training of Video Generation Foundation Model
 https://arxiv.org/abs/2504.08685

- https://static.magi.world/static/files/MAGI_1.pdf 

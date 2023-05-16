# Experiential Explanations for Reinforcement Learning 
> This repo contains the code of the experiential explanation framework described in the paper 
Experiential Explanations for  Reinforcement Learning. 

## Table of Contents
* [General Info](#general-information)
* [Getting Started](#getting-started)


## General Information
### Problem 
Reinforcement Learning systems can be complex and non-interpretable, making it challenging for non-AI experts to understand or intervene in their decisions. This is due, in part, to the sequential nature of RL in which actions are chosen because of future rewards. However, RL agents discard the qualitative features of their training, making it hard to recover user-understandable information for ``why'' an action is chosen.

### Overview of proposed project
We propose a technique, <em> Experiential Explanations</em> in which counterfactual explanations are generated by training <em> influence predictors</em>, models that train alongside the RL policy and learn a model of how different sources of reward affect the agent in different states, thus restoring information about how the policy reflects the environment.

### External code: 
- The Minigrid Library - [https://minigrid.farama.org](https://minigrid.farama.org) (forked and modified, with new costum environment at ```src/gym_minigrid```.)
- RL Agents for Minigrid worlds - [https://github.com/lcswillems/rl-starter-files](https://github.com/lcswillems/rl-starter-files). (forked and modified, with updated training and influence predcitors at ```stc/rl_minigrid```.)

## Getting Started
## Setup
This project requires the following Python version:
Python >= 3.10.2
You can install the project dependencies with:

```environment.yml```: 
```
conda env create -f environment.yml
```

## Usage
### Example explanations
The notebook ```src/scripts/experiential_explanations.ipynb```shows an example interaction of a proposed counterfactual and the resulted explanation. 

### Training 
To train the agent and the influence predictors, run the following command
```
python src/scripts/train_offline.py 
```

### Simulation Generation
Then run the ```src/scripts/observation_data_generation.py``` to get the simulations of the agent's actual paths in multiple environments.
```
python src/scripts/observation_data_generation.py
```
### Explanation Generation
Then run the notebook ```src/scripts/experiential_explanations.ipynb``` to experiment with different counterfactual explanations. 



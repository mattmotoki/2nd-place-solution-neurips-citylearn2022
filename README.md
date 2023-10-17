# [NeurIPS 2022 Citylearn Challenge](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge)


This repository contains the 2nd place solution for the NeurIPS 2022 Citylearn Challenge.

#  Competition Overview
[The CityLearn Challenge 2022](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge) focuses on the opportunity brought on by home battery storage devices and photovoltaics. It leverages [CityLearn](https://github.com/intelligent-environments-lab/CityLearn/tree/citylearn_2022), a Gym Environment for building distributed energy resource management and demand response. The challenge utilizes 1 year of operational electricity demand and PV generation data from 17 single-family buildings in the Sierra Crest home development in Fontana, California, that were studied for _Grid integration of zero net energy communities_.

Participants will develop energy management agent(s) and their reward function for battery charge and discharge control in each building with the goals of minimizing the monetary cost of electricity drawn from the grid, and the CO<sub>2</sub> emissions when electricity demand is satisfied by the grid.

# Solution Overview

* [Conference Paper](https://proceedings.mlr.press/v220/nweye22a/nweye22a.pdf)
* [Video Presentation](https://www.youtube.com/watch?v=Yel5zybmvwg&t=2314s)

## Phase 1
During phase 1, we solve the single-agent optimization problem using dynamic programming (DP). The main difficulty is that the grid costs are noncausal. We instead optimize with respect to the following proxy cost function

```
cost = power_price[time_step + 1] / base_power_cost * np.clip(net_energy, 0, None)
cost += emission_price[time_step + 1] / base_emission_cost * np.clip(net_energy, 0, None)
cost += proxy_weight1 * np.abs(net_energy) / base_proxy_cost1
cost += proxy_weight2 * np.abs(net_energy ** 2) / base_proxy_cost2
cost += proxy_weight3 * np.abs(net_energy ** 3) / base_proxy_cost3
```
where base_power_cost, base_emission_cost, base_proxy_cost1, base_proxy_cost2, base_proxy_cost3, are the costs associated with baseline control (a=0). We treat l1_penalty, l2_penalty, proxy_weight1, proxy_weight2, proxy_weight3 as hyperparameters which we tune with random search. 

We improve the single-agent actions using a neural network policy which we train with CMA-ES. The input to the policy network consists of the single-agent DP actions, future costs, and future energy usage. Moreover, we calculate the energy usage for a given action using the battery code from the CityLearn environment and use that as additional input to the policy network.

## Phase 2 and 3
Our approach is similar to the multiagent policy trained in phase 2. Our policy network has two stages. The first stage takes in single-agent information and outputs a single action. We collect these actions and calculate their energy usage. The single-agent actions and their energy usages are aggregated and passed into the second stage of the policy network which learns to adjust the single-agent actions. Ultimately, the only input to the policy network is future costs and forecasts of future energy usage. We use l2 regularization to try to improve generalization.

The policy network is trained with CMA-ES. There are only a few hyperparameters to tune--the strength of the l2 regularization, the number of iterations for CMA-ES, and the initial standard deviation for CMA-ES. We adjusted these parameters by hand using 6-fold time-series cross-validation to evaluate performance. 


## Recommended Hardware Specifications
It is recommended to have a computer with
* at least 64 GB of RAM
* a CPU with at least 64 cores
* a GPU with at least 12 GB of VRAM


## Installation
Create a conda env and install the necessary packages. 
```
conda env create -f environment.yml
pip install git+https://github.com/intelligent-environments-lab/CityLearn.git@3c12f30ce7b5ed0ac75a551d4e62c380f77ade0d
```

Activate the environment.
```
conda activate citylearn
```

Install torch (torchvision and torchaudio are not necessary) following the directions at [pytorch.org](https://pytorch.org/).

Add the `citylearnutils` directory to your `PYTHONPATH`. 
```
export PYTHONPATH="${PYTHONPATH}:/path/to/dir/citylearnutils"
```

## Run Training

Run the preprocessing and training scripts.
```
python scripts/1_initialize.py
python scripts/2_create_demand_model.py
python scripts/3_create_solar_model.py
python scripts/4_create_training_single_agents.py
python scripts/5_create_training_multi_agents.py
python scripts/6_policy_optimization.py
```

The rough timing of these scripts is as follows:

| Script | Timing |
|---|---|
| 1_initialize.py | 5s |
| 2_create_demand_model.py | 20 minutes |
| 3_create_solar_model.py | 5 minutes |
| 4_create_training_single_agents.py | 15 minutes |
| 5_create_training_multi_agents.py | 1.5 days |
| 6_policy_optimization.py | 2.5 days |



import cma
import numpy as np
import pandas as pd
import multiprocessing
from citylearnutils.simulator import BatterySimulator
from citylearnutils.evaluation import evaluate_agent
from citylearnutils.orderenforcingwrapper import OrderEnforcingAgent

DATA_PATH = "./data"
MODEL_PATH = f"{DATA_PATH}/citylearn_challenge_2022_phase_1/models"

class MultiAgent:

    def __init__(self, params):

        # model0 (30 features, 15 hidden, 1 output)
        self.weights0 = params[:450].reshape(15, 30)
        self.weights1 = params[450:465]

        self.n_leads = 4

        self.mu = np.array([
            *[0.1565307] * self.n_leads,  # emission leads
            *[0.2731312] * self.n_leads,  # price leads
            *[0.3679884] * self.n_leads,  # local forecast,
            *[1.8399420] * self.n_leads,  # group forecast,
            0.0, 0.0, 1.445, 0.2890
        ])

        self.sig = np.array([
            *[0.035367] * self.n_leads,  # emission leads
            *[0.117795] * self.n_leads,  # price leads
            *[1.041145] * self.n_leads,  # local forecast
            *[5.205725] * self.n_leads,  # group forecast
            1.0, 1.0, 8.325, 1.6650
        ])

        # load external data
        self.emission = pd.read_csv(
            f"{DATA_PATH}/citylearn_challenge_2022_phase_1/carbon_intensity.csv").values.flatten()
        self.price = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/pricing.csv").iloc[:, 0].values

        self.emission = np.append(self.emission, np.zeros(100))
        self.price = np.append(self.price, np.zeros(100))

        observations = pd.read_csv(f"{DATA_PATH}/external/observations.csv")
        self.dp_action = np.load(f"{DATA_PATH}/external/single_agent_dp.npy")

        # load forecasts
        self.local_forecasts = []
        for _, df in observations.groupby("building_num"):
            x = (df.non_shiftable_load - df.solar_generation).values
            self.local_forecasts.append(np.append(x, np.zeros(100)))
        self.group_forecasts = np.mean(self.local_forecasts, 0)

        # initialize empty lookups
        self.step = 0
        self.simulators = {}
        self.action_space = {}
        self.state_history = {}
        self.action_history = []
        self.observation_history = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def reset_agent(self, agent_id, observation):
        self.simulators[agent_id] = BatterySimulator()
        self.state_history[agent_id] = dict(soc=[0.0], capacity=[6.4], net_energy=[0.0])
        self.observation_history[agent_id] = []
        self.step = 0

    def update_action(self, action, features):
        hidden = np.dot(self.weights0, features).clip(0)
        action += np.tanh(self.weights1.dot(hidden))
        return action

    def compute_action(self, observation_list):

        # store observations
        for agent_id, observation in enumerate(observation_list):
            self.observation_history[agent_id].append(observation[20:23])

        # stage 0 actions
        action_list = []
        next_net_energy_list = []
        prev_net_energy_list = []
        for agent_id in range(len(observation_list)):

            # lookup action
            action = self.dp_action[agent_id, self.step]
            action_list.append(action)

            # simulate action
            next_soc, next_capacity, _, battery_energy = self.simulators[agent_id].fast_simulate(
                action=np.clip(action, -1, 1),
                current_soc=self.state_history[agent_id]["soc"][-1],
                current_capacity=self.state_history[agent_id]["capacity"][-1]
            )

            # calculate total energy consumption
            net_energy = battery_energy
            net_energy += self.observation_history[agent_id][-1][0]  # non_shiftable_load
            net_energy -= self.observation_history[agent_id][-1][1]  # solar_generation
            prev_net_energy_list.append(self.state_history[agent_id]["net_energy"][-1])
            next_net_energy_list.append(net_energy)

        # stage 1 actions
        prev_group_net_energy = sum(prev_net_energy_list)
        next_group_net_energy = sum(next_net_energy_list)
        g_inc = (next_group_net_energy > prev_group_net_energy)
        g_dec = (next_group_net_energy < prev_group_net_energy)
        g_pos = (next_group_net_energy > 0)
        g_neg = (next_group_net_energy < 0)

        actions = []
        for agent_id, (action, prev_agent_net_energy, next_agent_net_energy) in enumerate(zip(
                action_list, prev_net_energy_list, next_net_energy_list
        )):
            # feature engineering
            a_inc = (next_agent_net_energy > prev_agent_net_energy)
            a_dec = (next_agent_net_energy < prev_agent_net_energy)
            a_pos = (next_agent_net_energy > 0)
            a_neg = (next_agent_net_energy < 0)

            I_inc_inc = (g_inc and a_inc)
            I_inc_dec = (g_inc and a_dec)
            I_dec_inc = (g_dec and a_inc)
            I_dec_dec = (g_dec and a_dec)

            I_pos_pos = (g_pos and a_pos)
            I_pos_neg = (g_pos and a_neg)
            I_neg_pos = (g_neg and a_pos)
            I_neg_neg = (g_neg and a_neg)

            additional_features = np.array([
                1, action,
                I_inc_inc, I_inc_dec, I_dec_inc, I_dec_dec,
                I_pos_pos, I_pos_neg, I_neg_pos, I_neg_neg,
            ])

            features = np.concatenate([
                self.emission[self.step: self.step + self.n_leads],
                self.price[self.step: self.step + self.n_leads],
                self.local_forecasts[agent_id][self.step: self.step + self.n_leads],
                self.group_forecasts[self.step: self.step + self.n_leads],
                np.array([
                    next_group_net_energy - prev_group_net_energy,
                    next_agent_net_energy - prev_agent_net_energy,
                    next_group_net_energy,
                    next_agent_net_energy
                ])
            ])

            features = (features - self.mu) / self.sig
            features = np.append(additional_features, features)
            action = self.update_action(action, features)

            # simulate action
            action = np.clip(action, -1, 1)
            actions.append(np.array([action]))
            self.action_history.append(action)

            next_soc, next_capacity, _, battery_energy = self.simulators[agent_id].fast_simulate(
                action=action,
                current_soc=self.state_history[agent_id]["soc"][-1],
                current_capacity=self.state_history[agent_id]["capacity"][-1]
            )

            # calculate total energy consumption
            net_energy = battery_energy
            net_energy += self.observation_history[agent_id][-1][0]  # non_shiftable_load
            net_energy -= self.observation_history[agent_id][-1][1]  # solar_generation

            # update state
            self.state_history[agent_id]["soc"].append(next_soc)
            self.state_history[agent_id]["capacity"].append(next_capacity)
            self.state_history[agent_id]["net_energy"].append(net_energy)

        self.step += 1

        return actions


def objective(params):
    agent = OrderEnforcingAgent(MultiAgent(params))
    env_costs = evaluate_agent(agent)[0]
    return np.mean(env_costs)


if __name__ == "__main__":

    # optimize model
    n_jobs = min(50, multiprocessing.cpu_count())
    es = cma.CMAEvolutionStrategy(x0=np.zeros(465), sigma0=0.005, inopts={"seed": 2022, "popsize": 50})
    es.optimize(objective, iterations=10000, verb_disp=1, n_jobs=n_jobs)

    # extract actions
    agent = OrderEnforcingAgent(MultiAgent(es.best.x))
    agent = evaluate_agent(agent)[2]

    # save actions
    best_actions = np.array(agent.agent.action_history).reshape(5, -1)
    np.save(f"{MODEL_PATH}/multi_agent_dp", best_actions)

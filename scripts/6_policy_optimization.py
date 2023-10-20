import cma
import numpy as np
import pandas as pd
import multiprocessing
from citylearnutils.utils import seed_everything
from citylearnutils.simulator import BatterySimulator
from citylearnutils.evaluation import evaluate_agent
from citylearnutils.orderenforcingwrapper import OrderEnforcingAgent

DATA_PATH = "./data"
MODEL_PATH = f"{DATA_PATH}/citylearn_challenge_2022_phase_1/models"


class UserAgent:

    def __init__(self, params):

        # model0 (18 features, 5 hidden, 1 output)
        self.bias = params[:5]
        self.time_bias = params[5:29]
        self.weights00 = params[29:119].reshape(5, 18)
        self.weights01 = params[119:124].reshape(1, 5)

        self.n_leads = 5

        self.mu0 = np.array([
            1.06639756e+00, 6.99354999e-01, 0.00000000e+00,
            *[0.1565307] * self.n_leads,  # emission leads
            *[0.2731312] * self.n_leads,  # price leads
            *[0.3679884] * self.n_leads,  # local forecast
        ])

        self.sig0 = np.array([
            8.89049950e-01, 1.01712690e+00, 1.00000000e+00,
            *[0.035367] * self.n_leads,  # emission leads
            *[0.117795] * self.n_leads,  # price leads
            *[1.041145] * self.n_leads,  # local forecast
        ])

        # model1a (5 features, 5 hidden, 1 output)
        self.weights10a = params[124:149].reshape(5, 5)
        self.weights11a = params[149:154].reshape(1, 5)

        # model1b (5 features, 5 hidden, 1 output)
        self.weights10b = params[154:179].reshape(5, 5)
        self.weights11b = params[179:184].reshape(1, 5)

        # normalization
        self.mu1a = np.array([0.0, 0.0, 0.0, 0.1565307, 0.273131])
        self.sig1a = np.array([1.0, 1.0, 1.0, 0.035367, 0.117795])

        self.mu1b = np.array([0.0, 1.445, 0.2890, 0.1565307, 0.273131])
        self.sig1b = np.array([1.0, 8.325, 1.6650, 0.035367, 0.117795])

        # load external data
        self.emission = pd.read_csv(
            f"{DATA_PATH}/citylearn_challenge_2022_phase_1/carbon_intensity.csv").values.flatten()
        self.price = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/pricing.csv").iloc[:, 0].values

        self.emission = np.append(self.emission, np.zeros(100))
        self.price = np.append(self.price, np.zeros(100))

        # load forecasts
        self.local_forecasts = []
        for i in 1 + np.arange(5):
            noise = 0.2 * np.random.randn(8760, 10)
            solar_forecast = np.load(f"{DATA_PATH}/external/forecasts/solar/pred_{i}.npy")
            demand_forecast = np.load(f"{DATA_PATH}/external/forecasts/demand/pred_{i}.npy")
            self.local_forecasts.append(demand_forecast - solar_forecast + noise)

        # initialize empty lookups
        self.simulators = {}
        self.action_space = {}
        self.state_history = {}
        self.action_history = []
        self.observation_history = {}
        self.step = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def reset_agent(self, agent_id, observation):
        self.simulators[agent_id] = BatterySimulator()
        self.state_history[agent_id] = dict(soc=[0.0], capacity=[6.4], net_energy=[0.0])
        self.observation_history[agent_id] = []
        self.step[agent_id] = 0

    def agent_action_0(self, observation, features):
        bias = self.bias * self.time_bias[observation[2] - 1]
        hidden = (bias + np.dot(self.weights00, features)).clip(0)
        action = self.weights01.dot(hidden)[0]
        return action

    def agent_action_1a(self, action, features):
        hidden = np.dot(self.weights10a, features).clip(0)
        action += np.sign(action) * np.tanh(self.weights11a.dot(hidden)[0])
        return action

    def agent_action_1b(self, action, features):
        hidden = np.dot(self.weights10b, features).clip(0)
        action += np.sign(action) * np.tanh(self.weights11b.dot(hidden)[0])
        return action

    def compute_action(self, observation_list):

        # store observations
        for agent_id, observation in enumerate(observation_list):
            self.observation_history[agent_id].append(observation[20:23])

        # stage 0 actions
        action_list = []
        next_net_energy_list = []
        prev_net_energy_list = []
        for agent_id, observation in enumerate(observation_list):

            # extract features
            features = np.concatenate([
                np.array(observation[20:23]),
                self.emission[self.step[agent_id]: self.step[agent_id] + self.n_leads],
                self.price[self.step[agent_id]: self.step[agent_id] + self.n_leads],
                self.local_forecasts[agent_id][self.step[agent_id]][:self.n_leads],
            ])

            features = (features - self.mu0) / self.sig0

            # simulate action
            action = self.agent_action_0(observation, features)
            action_list.append(action)

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

            if (g_inc and a_inc) or (g_dec and a_dec):
                features = np.abs([
                    action,
                    next_group_net_energy - prev_group_net_energy,
                    next_agent_net_energy - prev_agent_net_energy,
                    self.emission[self.step[agent_id] + 1],
                    self.price[self.step[agent_id] + 1],
                ])

                features = (features - self.mu1a) / self.sig1a

                action = self.agent_action_1a(action, features)

            if (g_pos and a_pos) or (g_neg and a_neg):
                features = np.abs([
                    action,
                    next_group_net_energy,
                    next_agent_net_energy,
                    self.emission[self.step[agent_id] + 1],
                    self.price[self.step[agent_id] + 1],
                ])

                features = (features - self.mu1b) / self.sig1b

                action = self.agent_action_1b(action, features)

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
            self.step[agent_id] += 1

        return actions


def objective(params):
    agent = OrderEnforcingAgent(UserAgent(params))
    env_costs = evaluate_agent(agent)[0]
    l2_reg = np.sum(params[29:] ** 2)
    return np.mean(env_costs) + 0.01 * l2_reg


if __name__ == "__main__":

    model_count = 0

    for seed in range(5):

        # optimize model
        seed_everything(seed)
        n_jobs = min(50, multiprocessing.cpu_count())
        es = cma.CMAEvolutionStrategy(x0=np.zeros(184), sigma0=0.05, inopts={"seed": seed, "popsize": 50})
        es.optimize(objective, iterations=3000, verb_disp=1, n_jobs=n_jobs)

        # extract model parameters
        lines = open("./outcmaes/xrecentbest.dat", "r").readlines()[1:]
        param_list = np.array([np.array(x.split(" ")[5:], dtype=np.float64) for x in lines])

        # save multiple checkpoints
        for params in param_list[[2500, 2600, 2700, 2800, 2900, -1]]:
            np.save(f"{MODEL_PATH}/params{model_count}", params)
            model_count += 1

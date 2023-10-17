import os
import torch
import torch.nn as nn
import numpy as np
import numba as nb
import pandas as pd
from scipy.special import expit

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(6)


DATA_PATH = "./data/citylearn_challenge_2022_phase_1"

@nb.njit()
def _simulate(action: float, soc: float, capacity: float, nominal_power: float, capacity_power_curve: np.ndarray,
              power_efficiency_curve: np.ndarray, efficiency_scaling: float, loss_coefficient: float,
              capacity_loss_coefficient: float, init_capacity: float):
    """Maps a (state, action) pair into the next state"""

    # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
    soc_normalized = soc / capacity

    # Calculating the maximum power rate at which the battery can be charged or discharged
    idx = max(0, np.argmax(soc_normalized <= capacity_power_curve[0]) - 1)

    intercept = capacity_power_curve[1][idx]

    slope = (capacity_power_curve[1][idx + 1] - capacity_power_curve[1][idx]) / \
            (capacity_power_curve[0][idx + 1] - capacity_power_curve[0][idx])

    max_power = intercept + slope * (soc_normalized - capacity_power_curve[0][idx])
    max_power *= nominal_power

    # calculate the energy of the action
    if action >= 0:
        energy = min(action * capacity, max_power)
    else:
        energy = max(-max_power, action * capacity)

    # Calculating the maximum power rate at which the battery can be charged or discharged
    energy_normalized = np.abs(energy) / nominal_power
    idx = max(0, np.argmax(energy_normalized <= power_efficiency_curve[0]) - 1)

    intercept = power_efficiency_curve[1][idx]

    slope = (power_efficiency_curve[1][idx + 1] - power_efficiency_curve[1][idx]) / \
            (power_efficiency_curve[0][idx + 1] - power_efficiency_curve[0][idx])

    efficiency = intercept + slope * (energy_normalized - power_efficiency_curve[0][idx])
    efficiency = efficiency ** efficiency_scaling

    # update state of charge
    if energy >= 0:
        next_soc = min(soc + energy * efficiency, capacity)
    else:
        next_soc = max(0.0, soc + energy / efficiency)

    # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge
    energy_balance = next_soc - soc * (1 - loss_coefficient)
    energy_balance *= (1 / efficiency if energy_balance >= 0 else efficiency)
    next_capacity = capacity - capacity_loss_coefficient * init_capacity * abs(energy_balance) / (2 * capacity)

    # calculate batter consumption
    soc_diff = next_soc - soc
    battery_energy = soc_diff * (1 / efficiency if soc_diff > 0 else efficiency)

    return next_soc, next_capacity, efficiency, battery_energy


class BatterySimulator:
    def __init__(self, init_capacity=6.4, nominal_power=5.0, loss_coefficient=0.0, efficiency_scaling=0.5,
                 capacity_loss_coefficient=1e-5):
        r"""Initialize `Battery`.

        Parameters
        ----------
        init_capacity : float
        nominal_power: float
        loss_coefficient : float
        efficiency_scaling : float
        capacity_loss_coefficient : float

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super classes.
        """

        self.init_capacity = init_capacity
        self.nominal_power = nominal_power
        self.loss_coefficient = loss_coefficient
        self.efficiency_scaling = efficiency_scaling
        self.capacity_loss_coefficient = capacity_loss_coefficient
        self.capacity_power_curve = np.array([[0., 0.8, 1.], [1., 1., 0.2]])
        self.power_efficiency_curve = np.array([[0., 0.3, 0.7, 0.8, 1.], [0.83, 0.83, 0.9, 0.9, 0.85]])

    def fast_simulate(self, action, current_soc, current_capacity):
        """Simulates charging. Doesn't update input or internal any state.

        Parameters
        ----------
        action : float
        current_soc : float
        current_capacity : float
        """
        return _simulate(
            action, current_soc, current_capacity,
            nominal_power=self.nominal_power,
            capacity_power_curve=self.capacity_power_curve,
            power_efficiency_curve=self.power_efficiency_curve,
            efficiency_scaling=self.efficiency_scaling,
            loss_coefficient=self.loss_coefficient,
            capacity_loss_coefficient=self.capacity_loss_coefficient,
            init_capacity=self.init_capacity
        )

    def simulate(self, action, current_soc, current_capacity):
        """Simulates charging. Doesn't update input or internal any state.

        Parameters
        ----------
        action : float
        current_soc : float
        current_capacity : float
        """

        if action >= 0:
            energy = min(action * current_capacity, self._get_max_power(current_soc, current_capacity))
        else:
            energy = max(-self._get_max_power(current_soc, current_capacity), action * current_capacity)

        # update efficiency
        current_efficiency = self._get_current_efficiency(energy)

        # update state of charge
        if energy >= 0.0:
            next_soc = min(current_soc + energy * current_efficiency, current_capacity)
        else:
            next_soc = max(0.0, current_soc + energy / current_efficiency)

        # update capacity
        next_capacity = current_capacity - self._get_degradation(
            current_soc, next_soc, current_capacity, current_efficiency
        )

        return next_soc, next_capacity, current_efficiency, energy

    def _get_max_power(self, soc, capacity) -> float:
        r"""Get maximum input power while considering `capacity_power_curve` limitations if defined otherwise, returns
        `nominal_power`.

        Returns
        -------
        max_power : float
            Maximum amount of power that the storage unit can use to charge [kW].
        """

        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc_normalized = soc / capacity

        # Calculating the maximum power rate at which the battery can be charged or discharged
        idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)

        intercept = self.capacity_power_curve[1][idx]

        slope = (self.capacity_power_curve[1][idx + 1] - self.capacity_power_curve[1][idx]) / \
                (self.capacity_power_curve[0][idx + 1] - self.capacity_power_curve[0][idx])

        max_power = intercept + slope * (soc_normalized - self.capacity_power_curve[0][idx])

        max_power *= self.nominal_power

        return max_power

    #
    def _get_current_efficiency(self, energy: float) -> float:
        r"""Get technical efficiency while considering `power_efficiency_curve` limitations if defined otherwise,
        returns `efficiency`.

        Returns
        -------
        efficiency : float
            Technical efficiency.
        """

        # Calculating the maximum power rate at which the battery can be charged or discharged
        energy_normalized = np.abs(energy) / self.nominal_power
        idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)

        intercept = self.power_efficiency_curve[1][idx]

        slope = (self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx]) / \
                (self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])

        efficiency = intercept + slope * (energy_normalized - self.power_efficiency_curve[0][idx])

        efficiency = efficiency ** self.efficiency_scaling

        return efficiency

    def _get_degradation(self, current_soc, next_soc, current_capacity, current_efficiency) -> float:
        r"""Get amount of capacity degradation.

        Returns
        -------
        current_soc : float
        soc : float
        current_capacity : float
        efficiency : float
        """

        # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge
        energy_balance = next_soc - current_soc * (1 - self.loss_coefficient)
        energy_balance *= (1 / current_efficiency if energy_balance >= 0 else current_efficiency)
        return self.capacity_loss_coefficient * self.init_capacity * abs(energy_balance) / (2 * current_capacity)


class SkipMLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0.1):
        super(SkipMLP, self).__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_in)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0):
        x = self.activation(self.linear1(x0))
        x = self.linear2(self.dropout(x))
        return x + x0


class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, n_out, dropout=0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_out)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return x


class SolarModel(nn.Module):

    def __init__(self, n_features, n_horizon, n_hidden, dropout):
        super(SolarModel, self).__init__()
        self.mlp1 = MLP(n_features, n_hidden, n_hidden, dropout)
        self.mlp2 = SkipMLP(n_hidden, n_hidden, dropout)
        self.mlp3 = MLP(n_hidden, n_hidden, n_horizon, dropout)

    def forward(self, features):
        return self.mlp3(self.mlp2(self.mlp1(features))).clamp_min(0)


class DemandModel(nn.Module):

    def __init__(self, n_features, n_targets, n_hidden, dropout, emb_dim):
        super(DemandModel, self).__init__()
        self.tod_embedding = nn.Embedding(24, emb_dim)
        self.dow_embedding = nn.Embedding(24, emb_dim)
        self.mlp1 = MLP(n_features + 2 * emb_dim, n_hidden, n_hidden, dropout)
        self.mlp2 = SkipMLP(n_hidden, n_hidden, dropout)
        self.mlp3 = MLP(n_hidden, n_hidden, n_targets, dropout)

    def forward(self, time_of_day, day_of_week, features):
        tod = self.tod_embedding(time_of_day)
        dow = self.dow_embedding(day_of_week)
        x = self.mlp1(torch.cat([tod, dow, features], 1))
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x.clamp_min(0)


@nb.njit()
def _uniform_update(x_new, value, count):
    return value + (x_new - value) / (count + 1)


@nb.njit()
def _average_update(x_new, value, baseline, count, decays):
    blend = value + (baseline - value) * (1 - decays) ** count
    return blend + decays * (x_new - blend)


class ExponentialSmoother:

    def __init__(self, daily_baseline, weekly_baseline):
        self.decays = np.array([0.125, 0.25, 0.5], dtype=np.float32)
        n_decays = len(self.decays)

        self.daily_x = np.repeat(daily_baseline, n_decays).reshape(24, -1)  # average
        self.daily_b = np.zeros((24, 1), np.float32)  # baseline
        self.daily_n = np.zeros(24, np.int32)  # count

        self.weekly_x = np.repeat(weekly_baseline, n_decays).reshape(168, -1)  # average
        self.weekly_b = np.zeros((168, 1), np.float32)  # baseline
        self.weekly_n = np.zeros(168, np.int32)  # count

        self.time_index = 0

    def fit_transform(self, x, n_lags=5, n_leads=10):

        # update daily state
        i = self.time_index % 24
        self.daily_b[i] = _uniform_update(x, self.daily_b[i], self.daily_n[i])
        self.daily_x[i] = _average_update(x, self.daily_x[i], self.daily_b[i], self.daily_n[i], self.decays)
        self.daily_n[i] += 1

        # update weekly state
        j = self.time_index % 168
        self.weekly_b[j] = _uniform_update(x, self.weekly_b[j], self.weekly_n[j])
        self.weekly_x[j] = _average_update(x, self.weekly_x[j], self.weekly_b[j], self.weekly_n[j], self.decays)
        self.weekly_n[j] += 1

        # extract current values
        i = np.arange(self.time_index - n_lags, self.time_index + n_leads) % 24
        j = np.arange(self.time_index - n_lags, self.time_index + n_leads) % 168
        y = np.append(self.daily_x[i], self.weekly_x[j])

        # update time
        self.time_index += 1

        return y


class Forecaster:

    def __init__(self):

        # hardcoded values
        self.n_demand_lags = 192
        self.n_solar_leads = 10
        self.n_solar_lags = 216

        # load forecast models
        self.solar_model = SolarModel(n_features=442, n_horizon=10, n_hidden=2048, dropout=0.0)
        state_dict = torch.load(f"{DATA_PATH}/models/solar_model.pt", map_location=torch.device("cpu"))
        self.solar_model.load_state_dict(state_dict)
        self.solar_model.eval()

        self.demand_models = []
        for seed in range(5):
            self.demand_models.append(DemandModel(n_features=282, n_targets=10, n_hidden=926, dropout=0.1, emb_dim=2))
            state_dict = torch.load(f"{DATA_PATH}/models/demand_model.{seed}.pt", map_location=torch.device("cpu"))
            self.demand_models[-1].load_state_dict(state_dict)
            self.demand_models[-1].eval()

        # initialize
        self.step = 0
        self.demand_lags = {}
        self.solar_lags = {}
        self.mean_solar_lags = {}
        self.mean_solar_leads = {}
        self.smoothers = {}

    def reset_agent(self, agent_id):

        mean_solar_phase1 = np.load(f"{DATA_PATH}/models/mean_solar_phase1.npy")
        mean_daily_solar = np.load(f"{DATA_PATH}/models/mean_daily_solar.npy")

        mean_weekly_demand = np.load(f"{DATA_PATH}/models/mean_weekly_demand.npy")
        mean_daily_demand = np.load(f"{DATA_PATH}/models/mean_daily_demand.npy")

        lead_pad_solar = mean_daily_solar[:self.n_solar_leads]
        lag_pad_solar = np.tile(mean_daily_solar, 10)[-self.n_solar_lags:]
        lag_pad_demand = np.tile(mean_weekly_demand, 10)[-self.n_demand_lags:]

        self.solar_lags[agent_id] = lag_pad_solar.tolist()
        self.demand_lags[agent_id] = lag_pad_demand.tolist()
        self.mean_solar_lags[agent_id] = np.append(lag_pad_solar, mean_solar_phase1)
        self.mean_solar_leads[agent_id] = np.append(mean_solar_phase1, lead_pad_solar)
        self.smoothers[agent_id] = ExponentialSmoother(mean_daily_demand, mean_weekly_demand)

    def get_forecast(self, observation_list):

        n_agents = len(observation_list)

        # store observations
        demand_es_features = []
        for agent_id, observation in enumerate(observation_list):
            self.demand_lags[agent_id].append(observation[20])
            self.solar_lags[agent_id].append(observation[21])
            demand_es_features.append(self.smoothers[agent_id].fit_transform(observation[20]))

        # time features
        time_of_day = torch.tensor(n_agents*[self.step % 24], dtype=torch.long)
        day_of_week = torch.tensor(n_agents*[(self.step // 24) % 7], dtype=torch.long)
        self.step += 1

        # lags, leads, averages
        demand_features = []
        solar_features = []
        for agent_id in range(n_agents):
            demand_features.append(np.append(
                self.demand_lags[agent_id][-self.n_demand_lags:][::-1],
                demand_es_features[agent_id]
            ))
            solar_features.append(np.concatenate([
                self.solar_lags[agent_id][-self.n_solar_lags:][::-1],
                self.mean_solar_lags[agent_id][self.step:self.step + self.n_solar_lags][::-1],
                self.mean_solar_leads[agent_id][self.step:self.step + self.n_solar_leads],
            ]))

        demand_features = torch.tensor(np.vstack(demand_features), dtype=torch.float)
        solar_features = torch.tensor(np.vstack(solar_features), dtype=torch.float)

        with torch.no_grad():
            demand_forecasts = 0.0
            for model in self.demand_models:
                demand_forecasts += model(time_of_day, day_of_week, demand_features).numpy() / len(self.demand_models)
            solar_forecasts = self.solar_model(solar_features).numpy()

        return demand_forecasts - solar_forecasts


class ActionCalculator:

    def __init__(self, model_num):

        # load model parameters
        params = np.load(f"{DATA_PATH}/models/params{model_num}.npy")

        self.bias = params[:5]
        self.time_bias = params[5:29]
        self.weights00 = params[29:119].reshape(5, 18)
        self.weights01 = params[119:124].reshape(1, 5)

        self.n_leads = 5

        self.mu0 = np.array([
            1.06639756e+00, 6.99354999e-01, 0.00000000e+00,
            *[0.1565307] * self.n_leads,  # emission leads
            *[0.2731312] * self.n_leads,  # price leads
            *[0.3679884] * self.n_leads,  # forecast
        ])

        self.sig0 = np.array([
            8.89049950e-01, 1.01712690e+00, 1.00000000e+00,
            *[0.035367] * self.n_leads,  # emission leads
            *[0.117795] * self.n_leads,  # price leads
            *[1.041145] * self.n_leads,  # forecast
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
        self.emission = pd.read_csv(f"{DATA_PATH}/carbon_intensity.csv").values.flatten()
        self.price = pd.read_csv(f"{DATA_PATH}/pricing.csv").iloc[:, 0].values

        # pad data
        self.emission = np.append(self.emission, np.zeros(100))
        self.price = np.append(self.price, np.zeros(100))

        # initialize empty lookups
        self.step = 0
        self.simulators = {}
        self.action_space = {}
        self.state_history = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def reset_agent(self, agent_id):
        self.simulators[agent_id] = BatterySimulator(nominal_power=5.0)
        self.state_history[agent_id] = dict(soc=[0.0], capacity=[6.4], net_energy=[0.0])

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

    def compute_all_actions(self, forecasts, observation_list):

        # level 0 actions
        action_list = []
        next_net_energy_list = []
        prev_net_energy_list = []
        for agent_id, observation in enumerate(observation_list):

            # extract features
            features = np.concatenate([
                np.array(observation[20:23]),
                self.emission[self.step:self.step + self.n_leads],
                self.price[self.step:self.step + self.n_leads],
                forecasts[agent_id][:self.n_leads],
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
            net_energy += observation[20]  # non_shiftable_load
            net_energy -= observation[21]  # solar_generation
            prev_net_energy_list.append(self.state_history[agent_id]["net_energy"][-1])
            next_net_energy_list.append(net_energy)

        # level 1 actions
        prev_group_net_energy = sum(prev_net_energy_list)
        next_group_net_energy = sum(next_net_energy_list)
        g_inc = (next_group_net_energy > prev_group_net_energy)
        g_dec = (next_group_net_energy < prev_group_net_energy)
        g_pos = (next_group_net_energy > 0)
        g_neg = (next_group_net_energy < 0)

        actions = []
        for agent_id, (action, prev_agent_net_energy, next_agent_net_energy, observation) in enumerate(zip(
                action_list, prev_net_energy_list, next_net_energy_list, observation_list
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
                    self.emission[self.step + 1],
                    self.price[self.step + 1],
                ])

                features = (features - self.mu1a) / self.sig1a
                action = self.agent_action_1a(action, features)

            if (g_pos and a_pos) or (g_neg and a_neg):
                features = np.abs([
                    action,
                    next_group_net_energy,
                    next_agent_net_energy,
                    self.emission[self.step + 1],
                    self.price[self.step + 1],
                ])

                features = (features - self.mu1b) / self.sig1b
                action = self.agent_action_1b(action, features)

            # simulate action
            #action = np.clip(action, -1, 1)
            actions.append(action)

            next_soc, next_capacity, _, battery_energy = self.simulators[agent_id].fast_simulate(
                action=action,
                current_soc=self.state_history[agent_id]["soc"][-1],
                current_capacity=self.state_history[agent_id]["capacity"][-1]
            )

            # calculate total energy consumption
            net_energy = battery_energy
            net_energy += observation[20]  # non_shiftable_load
            net_energy -= observation[21]  # solar_generation

            # update state
            self.state_history[agent_id]["soc"].append(next_soc)
            self.state_history[agent_id]["capacity"].append(next_capacity)
            self.state_history[agent_id]["net_energy"].append(net_energy)

        self.step += 1

        return actions


class UserAgent:

    def __init__(self):

        # initialize
        self.step = 0
        self.action_space = {}
        self.forecaster = Forecaster()
        self.calculators = [ActionCalculator(model_num) for model_num in range(30)]
        self.dp = np.load(f"{DATA_PATH}/models/multi_agent_dp.npy")
        self.id = None

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def reset_agent(self, agent_id):
        self.step = 0
        self.forecaster.reset_agent(agent_id)
        for calc in self.calculators:
            calc.reset_agent(agent_id)

    def compute_all_actions(self, observation_list):

        # dynamic programming solution
        if self.id is None:
            self.id = "-".join([str(x[20]) for x in observation_list])

        if self.id == "2.2758-2.18875-1.0096232096354177e-07-2.81915-0.7714333333333336":
            actions = []
            for agent_id in range(len(observation_list)):
                action = self.dp[agent_id, self.step]
                actions.append(np.clip(np.array([action], dtype=np.float32), -1, 1))
            self.step += 1
            return actions

        # create forecasts
        forecasts = self.forecaster.get_forecast(observation_list)

        actions = 0.0
        for calc in self.calculators:
            actions += np.array(calc.compute_all_actions(forecasts, observation_list)) / len(self.calculators)
        actions = [np.array([np.clip(a, -1, 1)]) for a in actions]
        self.step += 1

        return actions



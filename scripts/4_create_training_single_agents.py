import tqdm
import joblib
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from citylearnutils.simulator import BatterySimulator

DATA_PATH = "./data"

SINGLE_AGENT_PARAMS = {
    'l1_penalty': 0.1300366652733203,
    'l2_penalty': 0.009508553021774526,
    'proxy_weight1': 0.011557095624191585,
    'proxy_weight2': 0.08482562812998984,
    'proxy_weight3': 2.61560899997805
}


def optimize_building(building_num, params, n_horizon=8760, n_states=101, verbose=False):

    # unpack parameters
    l1_penalty = params["l1_penalty"]
    l2_penalty = params["l2_penalty"]
    proxy_weight1 = params["proxy_weight1"]
    proxy_weight2 = params["proxy_weight2"]
    proxy_weight3 = params["proxy_weight3"]

    def evaluate_action(action, current_soc, current_capacity=6.4):

        # run simulation forwards
        next_soc, next_capacity, next_efficiency, _ = simulator.fast_simulate(action, current_soc, current_capacity)

        # calculate net energy consumption
        soc_diff = next_soc - current_soc
        net_energy = soc_diff * (1 / next_efficiency if soc_diff > 0 else next_efficiency) + net_external_load

        # calculate costs
        cost = power_price[time_step + 1] / base_power_cost * np.clip(net_energy, 0, None)
        cost += emission_price[time_step + 1] / base_emission_cost * np.clip(net_energy, 0, None)
        cost += proxy_weight1 * np.abs(net_energy) / base_proxy_cost1
        cost += proxy_weight2 * np.abs(net_energy ** 2) / base_proxy_cost2
        cost += proxy_weight3 * np.abs(net_energy ** 3) / base_proxy_cost3

        # calculate penalty
        penalty = l1_penalty * abs(action) + l2_penalty * action ** 2

        return penalty + cost + J_next(next_soc)

    nominal_power = 5.0 if building_num == 4 else 4.0

    # load data
    building_data = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/Building_{building_num}.csv")
    non_shiftable_load = building_data.iloc[:, 7].values
    solar_generation = building_data.iloc[:, 11].values

    emission_price = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/carbon_intensity.csv").values.flatten()
    power_price = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/pricing.csv").iloc[:, 0].values

    # initialize
    soc_history = [0.0]
    capacity_history = [6.4]
    efficiency_history = [0.9]
    simulator = BatterySimulator()

    # run baseline simulation
    for action in np.zeros(n_horizon - 1):

        # simulate baseline system
        current_soc = soc_history[-1]
        current_capacity = capacity_history[-1]
        next_soc, next_capacity, next_efficiency, _ = simulator.fast_simulate(action, current_soc, current_capacity)

        # update history
        soc_history.append(next_soc)
        capacity_history.append(next_capacity)
        efficiency_history.append(next_efficiency)

    # calculate baseline storage consumption
    electrical_storage_electricity_consumption = np.append(0,
                                                           np.diff(soc_history) *
                                                           np.where(np.diff(soc_history) > 0,
                                                                    1 / np.array(efficiency_history[1:]),
                                                                    np.array(efficiency_history[1:]))
                                                           )

    # calculate baseline net energy consumption
    net_energy_without_storage = (
            electrical_storage_electricity_consumption
            + non_shiftable_load[:n_horizon]
            - nominal_power * solar_generation[:n_horizon] / 1000
    )

    # get base costs
    base_power_cost = (power_price[:n_horizon] * net_energy_without_storage).clip(0).sum()
    base_emission_cost = (emission_price[:n_horizon] * net_energy_without_storage).clip(0).sum()

    # get base proxy costs
    base_proxy_cost1 = np.abs(net_energy_without_storage).mean()
    base_proxy_cost2 = np.abs(net_energy_without_storage ** 2).mean()
    base_proxy_cost3 = np.abs(net_energy_without_storage ** 3).mean()

    # initialize
    J = np.zeros((n_horizon + 1, n_states))
    mu = np.zeros((n_horizon + 1, n_states))

    state_grid = np.linspace(0, 6.4, n_states)
    simulator = BatterySimulator()

    # optimize
    for time_step in tqdm.tqdm(range(n_horizon - 1)[::-1], total=n_horizon - 1, disable=not verbose):

        # interpolate next step cost-to-go
        J_next = interp1d(state_grid, J[time_step + 1])

        # calculate next energy
        net_external_load = non_shiftable_load[time_step + 1] \
                            - nominal_power * solar_generation[time_step + 1] / 1000

        # optimize each state
        for i, current_soc in enumerate(state_grid):
            # find best action
            f = partial(evaluate_action, current_soc=current_soc)
            results = minimize_scalar(f, bounds=(-1, 1), method="bounded")

            J[time_step, i] = results.fun
            mu[time_step, i] = results.x

        J[time_step] -= J[time_step].min()

    # find sequence of best actions
    current_soc = 0.0
    current_capacity = 6.4
    best_actions = []

    for time_step in range(n_horizon - 1):

        # interpolate next step cost-to-go
        J_next = interp1d(state_grid, J[time_step + 1])

        # calculate next energy
        net_external_load = non_shiftable_load[time_step + 1] \
                            - nominal_power * solar_generation[time_step + 1] / 1000

        # find best action
        f = partial(evaluate_action, current_soc=current_soc)
        results = minimize_scalar(f, bounds=(-1, 1), method="bounded")
        best_actions.append(np.clip(results.x, -1, 1))

        # run simulation forwards
        next_soc, next_capacity, _, _ = simulator.fast_simulate(best_actions[-1], current_soc, current_capacity)

        # update state
        current_soc = next_soc
        current_capacity = next_capacity

    return best_actions


if __name__ == "__main__":

    # optimize individual buildings
    best_actions = []
    n_jobs = min(5, multiprocessing.cpu_count())
    f = partial(optimize_building, params=SINGLE_AGENT_PARAMS)
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for x in pool.map(f, [1, 2, 3, 4, 5]):
            best_actions.append(x)
    best_actions = np.vstack(best_actions)

    # save results
    np.save(f"{DATA_PATH}/external/single_agent_dp", best_actions)

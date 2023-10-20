import os
import numpy as np
import pandas as pd
from citylearnutils.utils import make_dir

DATA_PATH = "./data"
MODEL_PATH = f"{DATA_PATH}/citylearn_challenge_2022_phase_1/models"


if __name__ == "__main__":

    # create directory to store temporary data
    make_dir(MODEL_PATH)
    make_dir(f"{DATA_PATH}/external/forecasts/demand")
    make_dir(f"{DATA_PATH}/external/forecasts/solar")

    # store relevant observations (building_num, emission_price, power_price, non_shiftable_load, solar_generation)
    observations = []

    emission_price = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/carbon_intensity.csv").iloc[:, 0].values
    power_price = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/pricing.csv").iloc[:, 0].values

    for i in [1, 2, 3, 4, 5]:
        df = pd.read_csv(f"{DATA_PATH}/citylearn_challenge_2022_phase_1/Building_{i}.csv")
        observations.append(pd.DataFrame(dict(
            building_num=i,
            emission_price=emission_price,
            power_price=power_price,
            non_shiftable_load=df["Equipment Electric Power [kWh]"].values,
            solar_generation=df["Solar Generation [W/kW]"].values * (5.0 if i == 4 else 4.0) / 1000
        )))

    observations = pd.concat(observations, ignore_index=True)
    observations.to_csv(f"{DATA_PATH}/external/observations.csv", index=False)

    # average solar value
    mean_solar_phase1 = observations.groupby(np.tile(np.arange(8760), 5)).solar_generation.mean().values
    np.save(f"{MODEL_PATH}/mean_solar_phase1", mean_solar_phase1)

    # create baseline forecast
    observations["time_of_day"] = np.tile(np.arange(24), 365 * 5)
    observations["day_of_week"] = np.concatenate([np.tile(np.repeat(np.arange(7), 24), 53)[:8760]] * 5)

    # create baseline forecasts
    mean_weekly_demand = observations.groupby(["day_of_week", "time_of_day"]).non_shiftable_load.mean().values
    mean_daily_demand = observations.groupby(["time_of_day"]).non_shiftable_load.mean().values
    mean_daily_solar = observations.groupby(["time_of_day"]).solar_generation.mean().values

    np.save(f"{MODEL_PATH}/mean_weekly_demand", mean_weekly_demand)
    np.save(f"{MODEL_PATH}/mean_daily_demand", mean_daily_demand)
    np.save(f"{MODEL_PATH}/mean_daily_solar", mean_daily_solar)

import time
import warnings
import numpy as np
import numba as nb
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error as mse_func, mean_absolute_error as mae_func
from citylearnutils.utils import seed_everything

N_DAYS = 8
N_TARGETS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "./data"
FORECAST_PATH = f"{DATA_PATH}/external/forecasts/demand"
MODEL_PATH = f"{DATA_PATH}/citylearn_challenge_2022_phase_1/models"

DEMAND_MODEl_PARAMS = {
    "training": {
        "batch_size": 256,
        "n_epochs": 25,
        "grad_clip": 0.1,
        "seed": 0
    },
    "scheduler": {
        "max_lr": 1.54e-05,
        "pct_start": 0.2,
        "div_factor": 1,
        "final_div_factor": 10000
    },
    "optimizer": {
        "weight_decay": 1e-07
    },
    "model": {
        "n_features": 282,
        "n_targets": 10,
        "n_hidden": 926,
        "dropout": 0.1,
        "emb_dim": 2
    }
}


@nb.njit()
def _uniform_update(x_new, value, count):
    return value + (x_new - value) / (count + 1)


@nb.njit()
def _average_update(x_new, value, baseline, count, decays):
    blend = value + (baseline - value) * (1 - decays) ** count
    return blend + decays * (x_new - blend)


class ExponentialSmoother():

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

    def fit_transform(self, x, n_lags=5, n_leads=N_TARGETS):
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


class DemandDataset(Dataset):

    def __init__(self, time_of_day, day_of_week, features, targets, device):
        super(DemandDataset, self).__init__()
        self.time_of_day = torch.tensor(time_of_day, dtype=torch.long, device=device)
        self.day_of_week = torch.tensor(day_of_week, dtype=torch.long, device=device)
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        time_of_day = self.time_of_day[index]
        day_of_week = self.day_of_week[index]
        features = self.features[index]
        targets = self.targets[index]
        return time_of_day, day_of_week, features, targets


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


if __name__ == "__main__":

    # preprocess data
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        # load data
        data = pd.read_csv(f"{DATA_PATH}/external/observations.csv")

        # create baseline forecast
        data["time_of_day"] = np.tile(np.arange(24), 365 * 5)
        data["day_of_week"] = np.concatenate([np.tile(np.repeat(np.arange(7), 24), 53)[:8760]] * 5)

        daily_baseline = data.groupby(["time_of_day"]).non_shiftable_load.mean().values
        weekly_baseline = data.groupby(["day_of_week", "time_of_day"]).non_shiftable_load.mean().values

        # lags
        lags = np.arange(N_DAYS * 24)
        lag_cols = [f"lag{i}" for i in lags]

        for building_num, df in data.groupby("building_num"):

            df_weekly_baseline = df.groupby(["day_of_week", "time_of_day"]).non_shiftable_load.mean().values
            df_weekly_baseline = (5 * weekly_baseline - df_weekly_baseline) / 4

            for i in lags:
                x = np.append(df.non_shiftable_load, np.tile(df_weekly_baseline, 5))
                data.loc[df.index, f"lag{i}"] = np.roll(x, i)[:len(df)]

        # exponential smoothing
        es_cols = [f"es{i}" for i in range(6 * 5 + 6 * N_TARGETS)]

        for building_num, df in data.groupby("building_num"):
            df_daily_baseline = df.groupby(["time_of_day"]).non_shiftable_load.mean().values
            df_daily_baseline = (5 * daily_baseline - df_daily_baseline) / 4

            df_weekly_baseline = df.groupby(["day_of_week", "time_of_day"]).non_shiftable_load.mean().values
            df_weekly_baseline = (5 * weekly_baseline - df_weekly_baseline) / 4

            es = ExponentialSmoother(df_daily_baseline, df_weekly_baseline)
            features = [es.fit_transform(x) for x in df.non_shiftable_load.values]
            data.loc[df.index, es_cols] = np.vstack(features)

        # targets
        leads = 1 + np.arange(N_TARGETS)
        target_cols = [f"target{i}" for i in leads]

        for building_num, df in data.groupby("building_num"):
            for i in leads:
                x = np.roll(df.non_shiftable_load, -i)
                x = np.append(x[:-N_TARGETS], weekly_baseline[:N_TARGETS])
                data.loc[df.index, f"target{i}"] = x

        # try to defragment dataframe
        feature_cols = lag_cols + es_cols
        data = data.dropna().reset_index(drop=True).copy()
        assert len(data) == 5 * 8760

    # train model and create out-of-fold predictions
    for seed in range(5):
        for drop_building_num in [1, 2, 3, 4, 5, None]:

            eval_start_time = time.time()
            seed_everything(seed)

            # create loaders
            if drop_building_num is None:

                trn_loader = DataLoader(
                    DemandDataset(
                        time_of_day=data.time_of_day.values,
                        day_of_week=data.day_of_week.values,
                        features=data[feature_cols].values,
                        targets=data[target_cols].values,
                        device=DEVICE
                    ),
                    batch_size=DEMAND_MODEl_PARAMS["training"]["batch_size"],
                    shuffle=True
                )

                val_loader = DataLoader(
                    DemandDataset(
                        time_of_day=data.time_of_day.values,
                        day_of_week=data.day_of_week.values,
                        features=data[feature_cols].values,
                        targets=data[target_cols].values,
                        device=DEVICE),
                    batch_size=DEMAND_MODEl_PARAMS["training"]["batch_size"],
                    shuffle=False
                )

            else:

                trn_loader = DataLoader(
                    DemandDataset(
                        time_of_day=data.loc[data.building_num != drop_building_num, "time_of_day"].values,
                        day_of_week=data.loc[data.building_num != drop_building_num, "day_of_week"].values,
                        features=data.loc[data.building_num != drop_building_num, feature_cols].values,
                        targets=data.loc[data.building_num != drop_building_num, target_cols].values,
                        device=DEVICE
                    ),
                    batch_size=DEMAND_MODEl_PARAMS["training"]["batch_size"],
                    shuffle=True
                )

                val_loader = DataLoader(
                    DemandDataset(
                        time_of_day=data.loc[data.building_num == drop_building_num, "time_of_day"].values,
                        day_of_week=data.loc[data.building_num == drop_building_num, "day_of_week"].values,
                        features=data.loc[data.building_num == drop_building_num, feature_cols].values,
                        targets=data.loc[data.building_num == drop_building_num, target_cols].values,
                        device=DEVICE),
                    batch_size=DEMAND_MODEl_PARAMS["training"]["batch_size"],
                    shuffle=False
                )

            # initialize model, optimizer, scheduler
            model = DemandModel(**DEMAND_MODEl_PARAMS["model"])
            model.to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), **DEMAND_MODEl_PARAMS["optimizer"])

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                epochs=DEMAND_MODEl_PARAMS["training"]["n_epochs"],
                **DEMAND_MODEl_PARAMS["scheduler"],
                steps_per_epoch=len(trn_loader),
            )

            criterion = nn.MSELoss()

            metrics = []
            for epoch in range(DEMAND_MODEl_PARAMS["training"]["n_epochs"]):

                # training
                model.train()
                start_time = time.time()
                trn_targets, trn_preds = [], []
                for time_of_day, day_of_week, features, targets in trn_loader:

                    optimizer.zero_grad()
                    preds = model(time_of_day, day_of_week, features)

                    trn_targets.append(targets.detach().cpu().numpy())
                    trn_preds.append(preds.detach().cpu().numpy())

                    loss = criterion(preds, targets.to(DEVICE))
                    loss.backward(loss)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), DEMAND_MODEl_PARAMS["training"]["grad_clip"])

                    optimizer.step()
                    scheduler.step()

                # Validation
                model.eval()
                val_targets, val_preds = [], []
                with torch.no_grad():
                    for time_of_day, day_of_week, features, targets in val_loader:
                        preds = model(time_of_day, day_of_week, features)
                        val_targets.append(targets.detach().cpu().numpy())
                        val_preds.append(preds.detach().cpu().numpy())

                # combine results
                trn_targets = np.concatenate(trn_targets)
                val_targets = np.concatenate(val_targets)

                trn_preds = np.concatenate(trn_preds)
                val_preds = np.concatenate(val_preds)

                # calculate metrics
                trn_rmse = np.sqrt(mse_func(trn_targets, trn_preds))
                val_rmse = np.sqrt(mse_func(val_targets, val_preds))

                trn_mae = mae_func(trn_targets, trn_preds)
                val_mae = mae_func(val_targets, val_preds)

                metrics.append(dict(trn_rmse=trn_rmse, trn_mae=trn_mae,
                                    val_rmse=val_rmse, val_mae=val_mae))

                print(f"drop {drop_building_num} - "
                      f"{epoch:02.0f} ({time.time() - start_time:0.1f}s): "
                      f"trn_rmse {trn_rmse:0.5f}  trn_mae {trn_mae:0.5f}  "
                      f"val_rmse {val_rmse:0.5f}  val_mae {val_mae:0.5f}")

            # save
            if drop_building_num is None:
                torch.save(model.state_dict(), f"{MODEL_PATH}/demand_model.{seed}.pt")
            else:
                np.save(f"{FORECAST_PATH}/pred_{drop_building_num}.{seed}", val_preds)
            print()

    # average predictions
    for building_num in range(5):
        preds = 0
        for seed in range(5):
            preds += np.load(f"{FORECAST_PATH}/pred_{building_num+1}.{seed}.npy")/5
        np.save(f"{FORECAST_PATH}/pred_{building_num+1}", preds)

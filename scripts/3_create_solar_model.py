import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error as mse_func, mean_absolute_error as mae_func
from citylearnutils.utils import seed_everything

SEED = 0
N_LAGS = 9*24
N_TARGETS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "./data"
FORECAST_PATH = f"{DATA_PATH}/external/forecasts/solar"
MODEL_PATH = f"{DATA_PATH}/citylearn_challenge_2022_phase_1/models"

SOLAR_MODEL_PARAMS = {
    "training": {
        "batch_size": 256,
        "n_epochs": 25,
        "grad_clip": 0.0014,
        "noise": 0.295,
        "seed": 0
    },
    "scheduler": {
        "max_lr": 0.00055,
        "pct_start": 0.2,
        "div_factor": 1,
        "final_div_factor": 10000
    },
    "optimizer": {
        "weight_decay": 1.28e-05
    },
    "model": {
        "n_features": 442,
        "n_horizon": 10,
        "n_hidden": 2048,
        "dropout": 0.0
    }
}


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


class SolarDataset(Dataset):

    def __init__(self, features, targets, device):
        super(SolarDataset, self).__init__()
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        features = self.features[index]
        targets = self.targets[index]
        return features, targets


class SolarModel(nn.Module):

    def __init__(self, n_features, n_horizon, n_hidden, dropout):
        super(SolarModel, self).__init__()
        self.mlp1 = MLP(n_features, n_hidden, n_hidden, dropout)
        self.mlp2 = SkipMLP(n_hidden, n_hidden, dropout)
        self.mlp3 = MLP(n_hidden, n_hidden, n_horizon, dropout)

    def forward(self, features):
        return self.mlp3(self.mlp2(self.mlp1(features))).clamp_min(0)


if __name__ == "__main__":

    # preprocess data
    with warnings.catch_warnings():

        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        # load data
        data = pd.read_csv(f"{DATA_PATH}/external/observations.csv")

        # average solar value
        group_solar = np.tile(data.groupby(np.tile(np.arange(8760), 5)).solar_generation.sum().values, 5)
        data["group_solar"] = group_solar

        # create baseline forecast
        data["time_of_day"] = np.tile(np.arange(24), 365 * 5)
        daily_baseline = data.groupby(["time_of_day"]).solar_generation.mean().values
        baseline_padding = np.tile(daily_baseline, N_LAGS // 24)

        # lags
        lags = np.arange(N_LAGS)
        lag_cols = [f"lag{i}" for i in lags]

        for building_num, df in data.groupby("building_num"):
            for i in lags:
                x = np.append(df.solar_generation, baseline_padding)
                data.loc[df.index, f"lag{i}"] = np.roll(x, i)[:len(df)]

        # group lags
        group_lag_cols = [f"group_lag{i}" for i in lags]

        for building_num, df in data.groupby("building_num"):
            for i in lags:
                x = np.append((df.group_solar - df.solar_generation) / 4, baseline_padding)
                data.loc[df.index, f"group_lag{i}"] = np.roll(x, i)[:len(df)]

        # group leads
        leads = 1 + np.arange(N_TARGETS)
        group_lead_cols = [f"group_lead{i}" for i in leads]

        for building_num, df in data.groupby("building_num"):
            for i in leads:
                x = np.roll((df.group_solar - df.solar_generation) / 4, -i)
                x = np.append(x[:-N_TARGETS], daily_baseline[:N_TARGETS])
                data.loc[df.index, f"group_lead{i}"] = x

        # targets
        target_cols = [f"target{i}" for i in leads]

        for building_num, df in data.groupby("building_num"):
            for i in leads:
                x = np.roll(df.solar_generation, -i)
                x = np.append(x[:-N_TARGETS], daily_baseline[:N_TARGETS])
                data.loc[df.index, f"target{i}"] = x

        # try to defragment dataframe
        feature_cols = lag_cols + group_lag_cols + group_lead_cols
        data = data.dropna().reset_index(drop=True).copy()

        assert len(data) == 5 * 8760

    # train model and create out-of-fold predictions
    for drop_building_num in [1, 2, 3, 4, 5, None]:

        eval_start_time = time.time()
        seed_everything(SEED)

        # create loaders
        if drop_building_num is None:
            trn_loader = DataLoader(
                SolarDataset(
                    features=data[feature_cols].values,
                    targets=data[target_cols].values,
                    device=DEVICE
                ),
                batch_size=SOLAR_MODEL_PARAMS["training"]["batch_size"],
                shuffle=True
            )

            val_loader = DataLoader(
                SolarDataset(
                    features=data[feature_cols].values,
                    targets=data[target_cols].values,
                    device=DEVICE),
                batch_size=SOLAR_MODEL_PARAMS["training"]["batch_size"],
                shuffle=False
            )

        else:
            trn_loader = DataLoader(
                SolarDataset(
                    features=data.loc[data.building_num != drop_building_num, feature_cols].values,
                    targets=data.loc[data.building_num != drop_building_num, target_cols].values,
                    device=DEVICE
                ),
                batch_size=SOLAR_MODEL_PARAMS["training"]["batch_size"],
                shuffle=True
            )

            val_loader = DataLoader(
                SolarDataset(
                    features=data.loc[data.building_num == drop_building_num, feature_cols].values,
                    targets=data.loc[data.building_num == drop_building_num, target_cols].values,
                    device=DEVICE),
                batch_size=SOLAR_MODEL_PARAMS["training"]["batch_size"],
                shuffle=False
            )

        # initialize model, optimizer, scheduler
        model = SolarModel(**SOLAR_MODEL_PARAMS["model"])
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), **SOLAR_MODEL_PARAMS["optimizer"])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=SOLAR_MODEL_PARAMS["training"]["n_epochs"],
            **SOLAR_MODEL_PARAMS["scheduler"],
            steps_per_epoch=len(trn_loader),
        )

        criterion = nn.MSELoss()

        for epoch in range(SOLAR_MODEL_PARAMS["training"]["n_epochs"]):

            # training
            model.train()
            start_time = time.time()
            trn_targets, trn_preds = [], []
            for features, targets in trn_loader:
                optimizer.zero_grad()
                preds = model(features)

                trn_targets.append(targets.detach().cpu().numpy())
                trn_preds.append(preds.detach().cpu().numpy())

                loss = criterion(preds, targets.to(DEVICE))
                loss.backward(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), SOLAR_MODEL_PARAMS["training"]["grad_clip"])

                optimizer.step()
                scheduler.step()

            # Validation
            model.eval()
            val_targets, val_preds = [], []
            with torch.no_grad():
                for features, targets in val_loader:
                    preds = model(features)
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

            print(f"drop {drop_building_num} - "
                  f"{epoch:02.0f} ({time.time() - start_time:0.1f}s): "
                  f"trn_rmse {trn_rmse:0.5f}  trn_mae {trn_mae:0.5f}  "
                  f"val_rmse {val_rmse:0.5f}  val_mae {val_mae:0.5f}")

        # save
        if drop_building_num is None:
            torch.save(model.state_dict(), f"{MODEL_PATH}/solar_model.pt")
        else:
            np.save(f"{FORECAST_PATH}/pred_{drop_building_num}", val_preds)
        print()

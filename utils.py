import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def prep_time_series_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # TODO: figure out what to do with soc in prediction
    """
    train: soc, nominal_power, location_id, sub_id
    label: power
    """
    df = df.groupby("id").agg(list)
    sessions = []
    labels = []
    for i, row in df.drop("timestamp", axis=1).iterrows():
        soc, power, nominal_power, location_id, sub_id = row
        session = np.empty((40, 4))
        label = np.empty((40))
        session_length = len(soc)
        for i in range(session_length):
            session[i] = np.array([soc[i], nominal_power[i], location_id[i], sub_id[i]])
            label[i] = power[i]

        sessions.append(session)
        labels.append(label)

    sessions = np.array(sessions)
    labels = np.array(labels)

    # np.save("train_data.npy", sessions)
    return sessions, labels


def interpolate_soc(df):
    soc_interpolated = (
        df.groupby("id")["soc"]
        .apply(
            lambda x: x.interpolate(method="spline", order=1, limit_direction="both")
        )
        .reset_index()["soc"]
    )
    df["soc"] = soc_interpolated
    return df


def prep_data(path, train_size=0.8):
    df = pd.read_parquet(path)
    df = df.drop("timestamp", axis=1)

    # Interpolate soc
    df = interpolate_soc(df)

    df = df.dropna()
    x, y = df[["id", "soc", "nominal_power", "location_id", "sub_id"]], df[["power"]]
    split = int(len(x) * train_size)

    x, y = np.array(x), np.array(y)

    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    train = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    return train, val

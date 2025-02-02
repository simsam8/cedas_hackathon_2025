import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def prep_time_series_data(path, train_size=0.8):
    # TODO: figure out what to do with soc in prediction
    """
    train: soc, nominal_power, location_id, sub_id
    label: power
    """
    df = pd.read_parquet(path)
    df = df.drop("timestamp", axis=1)
    # Interpolate soc
    df = interpolate_soc(df)

    df = df.dropna()
    df = df.groupby("id").agg(list)
    sessions = []
    labels = []
    for i, row in df.iterrows():
        soc, power, nominal_power, location_id, sub_id = row

        # Pad to equal sequence length of 40
        session = np.zeros((40, 4))
        label = np.zeros((40))
        session_length = len(soc)
        for i in range(session_length):
            session[i] = np.array([soc[i], nominal_power[i], location_id[i], sub_id[i]])
            label[i] = power[i]

        sessions.append(session)
        labels.append(label)

    x = np.array(sessions)
    y = np.array(labels)

    split = int(len(x) * train_size)
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

    # np.save("train_data.npy", sessions)
    return train, val


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


def prep_data(
    path, do_split=False, train_size=0.8, as_torch_data=False, eval_set=False
):
    df = pd.read_parquet(path)
    df = df.drop("timestamp", axis=1)

    # Interpolate soc
    df = interpolate_soc(df)

    if not eval_set:
        df = df.dropna()
    # df["power"] = df["power"].replace(np.nan, -1)
    x, y = df[["id", "soc", "nominal_power", "location_id", "sub_id"]], df[["power"]]
    split = int(len(x) * train_size)

    x, y = np.array(x), np.array(y)

    if not do_split:
        return x, y

    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    if as_torch_data:
        train = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
    else:
        train = (x_train, y_train)
        val = (x_val, y_val)

    return train, val

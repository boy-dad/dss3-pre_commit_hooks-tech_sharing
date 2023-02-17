"""
This module handles data processing specific to grid demand forecasting.
It involves loading the raw information, adding relevant features,
and generating a dataset apt for training.
"""

from datetime import datetime

import pandas as pd
import torch


def onload(config: dict):
    """_summary_

    Args:
        config (dict): _description_

    Returns:
        _type_: _description_
    """
    paths = config["environment"]["sources"]
    df_raw = pd.read_parquet(paths["rtd_regional_summaries"])
    df_raw.set_index("RUN_TIME", inplace=True)

    df = pd.DataFrame()
    df["RUN_TIME"] = pd.date_range(start=df_raw.index[0], end=df_raw.index[-1], freq="5min")
    df = df.merge(df_raw, on="RUN_TIME")

    if df.isnull().any().sum() > 0:
        df = df.interpolate(method="linear")
    df.set_index("RUN_TIME", inplace=True)
    return df


def generate_time_index(df: pd.DataFrame):
    """_summary_

    Args:
        config (dict): _description_

    Returns:
        _type_: _description_
    """
    base_time_id = datetime.timestamp(datetime.strptime("2021-05-30 00:00:00", "%Y-%m-%d %H:%M:%S"))
    df["time_index"] = pd.DatetimeIndex(df.index).asi8 / 10**9 - base_time_id
    # df["time_index"] -= df["time_index"].min()
    df["time_index"] = df["time_index"] / 300
    df["time_index"] = df.time_index.astype(int)
    """
    Generate time index
    min_tosec = df.index.to_series().dt.minute.values * 60
    hour_tosec = df.index.to_series().dt.hour.values * 60 * 60
    day_tosec = df.index.to_series().dt.day.values * 60 * 60 * 24
    month_tosec = df.index.to_series().dt.month.values * 60 * 60 * 24 * 30
    year_tosec = df.index.to_series().dt.year.values * 60 * 60 * 24 * 30 * 12
    df["time_index"] = year_tosec + month_tosec + day_tosec + hour_tosec + min_tosec
    df["time_index"] -= df["time_index"].min()
    df["time_index"] /= 300
    """
    return df


def handle_holidays(df: pd.DataFrame, config: dict):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        config (dict): _description_

    Returns:
        _type_: _description_
    """
    paths = config["environment"]["sources"]
    df_holidays = pd.read_parquet(paths["holidays"])
    df_holidays.columns = ["date", "holiday"]
    df_holidays["holiday"] = "1"
    df["date"] = pd.to_datetime(df.index.date)
    df = df.reset_index().merge(df_holidays, on="date", how="left")
    df["holiday"] = df["holiday"].fillna("0")
    df["holiday"] = df["holiday"].astype("category")
    df.drop("date", axis=1, inplace=True)
    df.set_index("RUN_TIME", inplace=True)
    return df


def add_peaks_id(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    peaks_id = ["P" if t.hour > 8 and t.hour < 21 else "OP" for t in df.index]
    df["peaks_id"] = peaks_id
    df["peaks_id"] = df["peaks_id"].astype("category")
    return df


def add_temporal_features(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    weekdays = df.index.to_series().dt.weekday
    df["week_day"] = weekdays.astype(str).astype("category")
    hours = df.index.to_series().dt.hour
    df["hour"] = hours.astype(str).astype("category")
    return df


def generate_feature_table(df: pd.DataFrame, config: dict):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        config (dict): _description_

    Returns:
        _type_: _description_
    """
    if df.isnull().any().sum() > 0:
        df = df.interpolate(method="linear")
    df = generate_time_index(df)
    df = add_temporal_features(df)
    df = add_peaks_id(df)
    df = handle_holidays(df, config)
    id_vars = ["RUN_TIME", "time_index", "week_day", "hour", "holiday", "peaks_id"]
    df = df.reset_index().melt(
        value_vars=["CLUZ", "CVIS", "CMIN"],
        var_name="region",
        id_vars=id_vars,
        value_name="demand",
    )
    for index, every_region in enumerate(["CLUZ", "CVIS", "CMIN"]):
        df.loc[df["region"] == every_region, "region"] = f"{index}-{every_region}"
    df = df.sort_values(["RUN_TIME", "region"], ignore_index=True)
    for index, every_region in enumerate(["CLUZ", "CVIS", "CMIN"]):
        df.loc[df["region"] == f"{index}-{every_region}", "region"] = f"{every_region}"
    # df['region'] = df['region'].astype('category')
    for every_col in ["week_day", "holiday", "hour"]:
        df[every_col] = df[every_col].astype(str)
    for every_col in ["peaks_id", "region"]:
        df[every_col] = df[every_col].astype(str)
    df = df.rename(columns={"RUN_TIME": "date"})
    return df


def prepare_encoder_decoder(current_date: str, df: pd.DataFrame, max_encoder_length: int, horizon_length: int):
    """_summary_

    Args:
        current_date (str): _description_
        df (pd.DataFrame): _description_
        max_encoder_length (int): _description_
        horizon_length (int): _description_

    Returns:
        _type_: _description_
    """
    encoder_data = df[df.index < current_date].iloc[-max_encoder_length:, :]
    decoder_data = pd.DataFrame(index=pd.date_range(start=current_date, periods=horizon_length, freq="5min"))
    regions = ["CLUZ", "CVIS", "CMIN"]
    decoder_data[regions] = encoder_data[regions].tail(1).values.tolist() * len(decoder_data)
    df_encoder_decoder = pd.concat([encoder_data, decoder_data], ignore_index=False)
    df_encoder_decoder.index.name = "RUN_TIME"
    return encoder_data, decoder_data, df_encoder_decoder


def process_prediction(prediction: torch.Tensor, forecast_date_string: str, horizon: int):
    """_summary_

    Args:
        prediction (torch.Tensor): _description_
        forecast_date_string (str): _description_
        horizon (int): _description_

    Returns:
        _type_: _description_
    """
    df_predictions = pd.DataFrame()
    df_predictions["date"] = pd.date_range(start=forecast_date_string, periods=horizon, freq="5min")
    columns = []
    for i, every_region in enumerate(["LUZ", "VIS", "MIN"]):
        df_predictions = pd.concat([df_predictions, pd.DataFrame(prediction[i].numpy())], axis=1)
        columns += [f"{every_region} q{x+1}" for x in range(7)]
    df_predictions.columns = ["date"] + columns
    return df_predictions

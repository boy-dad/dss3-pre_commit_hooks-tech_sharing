"""
This module contains the function to use when training for a grid demand model.
Note that a specific format is needed as input for it run properly.
Please see data_processing file for more details.
"""

import os

import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import torch
from mlflow.models.signature import infer_signature
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .data_processing import generate_feature_table, onload, prepare_encoder_decoder, process_prediction

torch.set_float32_matmul_precision("medium")


def run(df: pd.DataFrame, config: dict):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        config (dict): _description_
    """
    dataset = config["grid_demand"]["dataset"]
    time_idx = dataset["time_idx"]
    horizon = eval(dataset["horizon"])
    max_encoder_length = eval(dataset["max_encoder_length"])
    time_idx = dataset["time_idx"]
    target = dataset["target"]
    group_ids = list(dataset["group_ids"])
    static_categoricals = list(dataset["static_categoricals"])
    time_varying_known_categoricals = list(dataset["time_varying_known_categoricals"])
    time_varying_known_reals = list(dataset["time_varying_known_reals"])
    time_varying_unknown_categoricals = list(dataset["time_varying_unknown_categoricals"])
    time_varying_unknown_reals = list(dataset["time_varying_unknown_reals"])
    target_normalizer_groups = list(dataset["target_normalizer_groups"])
    add_relative_time_index = dataset["add_relative_time_index"]
    add_target_scales = dataset["add_target_scales"]
    add_encoder_length = dataset["add_encoder_length"]
    allow_missing_timesteps = dataset["allow_missing_timesteps"]

    training = TimeSeriesDataSet(
        df[lambda x: x[time_idx] <= (df[time_idx].max() - horizon)],
        time_idx=f"{time_idx}",
        target=f"{target}",
        group_ids=group_ids,
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=horizon,
        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=target_normalizer_groups, transformation="softplus"),
        add_relative_time_idx=add_relative_time_index,
        add_target_scales=add_target_scales,
        add_encoder_length=add_encoder_length,
        allow_missing_timesteps=allow_missing_timesteps,
    )

    paths = config["environment"]
    batch_size = dataset["batch_size"]
    num_workers = paths["dataset"]["num_workers"]
    pin_memory = dataset["pin_memory"]

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=num_workers, pin_memory=pin_memory
    )

    trainer_meta = config["grid_demand"]["trainer"]
    epochs = trainer_meta["epochs"]
    gradient_clip = trainer_meta["gradient_clip"]
    enable_progress_bar = trainer_meta["enable_progress_bar"]
    enable_checkpointing = trainer_meta["enable_checkpointing"]
    model_root = os.path.join(paths["model"]["model_root"], trainer_meta["model_root_folder_name"])

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_root,
        filename="grid_demand_1d-{epoch:02d}-{val_loss:.2f}",
    )

    mlflow.set_tracking_uri(paths["mlflow"]["tracking_uri"])

    mlflow.pytorch.autolog()
    mlflow_meta = config["grid_demand"]["mlflow"]
    experiment_id = mlflow_meta["experiment_id"]

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_params(pl.utilities.logger._flatten_dict(config))
        mlf_logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(run.info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
        )

        pl.seed_everything(42)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            gradient_clip_val=gradient_clip,
            default_root_dir=model_root,
            max_epochs=epochs,
            enable_progress_bar=enable_progress_bar,
            enable_checkpointing=enable_checkpointing,
            callbacks=[model_checkpoint],
            logger=mlf_logger,
        )

        hyperparam = config["grid_demand"]["model"]
        learning_rate = hyperparam["learning_rate"]
        hidden_size = hyperparam["hidden_size"]
        hidden_continuous_size = int(hidden_size / 2)
        attn_head_size = hyperparam["attn_head_size"]
        dropout = hyperparam["dropout"]
        lstm_layers = hyperparam["lstm_layers"]
        output_size = hyperparam["output_size"]
        reduce_on_plateau_patience = hyperparam["reduce_on_plateau_patience"]

        tft = TemporalFusionTransformer.from_dataset(
            training,
            # not meaningful for finding the learning rate but otherwise very important
            lstm_layers=lstm_layers,
            learning_rate=learning_rate,
            hidden_size=hidden_size,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=attn_head_size,
            dropout=dropout,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=hidden_continuous_size,  # set to <= hidden_size
            output_size=output_size,  # 7 quantiles by default
            loss=QuantileLoss(),
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=reduce_on_plateau_patience,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_checkpoint.best_model_path)
    df_input = onload(config)
    forecast_date_string = "2023-02-10"

    _, _, df_input = prepare_encoder_decoder(
        forecast_date_string, df_input, max_encoder_length=max_encoder_length, horizon_length=horizon
    )
    df_input = generate_feature_table(df_input, config)
    predictions, raw = best_tft.predict(df_input, mode="raw", return_x=True)
    df_predictions = process_prediction(predictions["prediction"], forecast_date_string, horizon)
    signature = infer_signature(df_input, df_predictions)
    mlflow.pytorch.log_model(best_tft, input_example=df_input, signature=signature, artifact_path="best_model")

    # mlf_logger.experiment.log_artifact(
    # run_id=mlf_logger.run_id,
    # local_path=model_checkpoint.best_model_path)

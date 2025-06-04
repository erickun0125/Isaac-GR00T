#!/usr/bin/env python
"""Simple example for fine-tuning GR00T on a dataset."""

from dataclasses import dataclass

from transformers import TrainingArguments
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1


@dataclass
class Config:
    dataset_path: str
    output_dir: str = "./gr00t_out"
    data_config: str = "gr1_arms_only"
    base_model_path: str = "nvidia/GR00T-N1-2B"
    batch_size: int = 8
    max_steps: int = 1000
    embodiment_tag: str = "new_embodiment"


def main(cfg: Config) -> None:
    emb_tag = EmbodimentTag(cfg.embodiment_tag)
    data_cfg = DATA_CONFIG_MAP[cfg.data_config]
    train_dataset = LeRobotSingleDataset(
        dataset_path=cfg.dataset_path,
        modality_configs=data_cfg.modality_config(),
        transforms=data_cfg.transform(),
        embodiment_tag=emb_tag,
    )

    model = GR00T_N1.from_pretrained(cfg.base_model_path)
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        max_steps=cfg.max_steps,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=False,
    )
    trainer.train()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)


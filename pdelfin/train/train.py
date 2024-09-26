import os
import json
import base64
import logging
import time
from io import BytesIO
from PIL import Image
from functools import partial
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from tqdm import tqdm

import torch
import torch.distributed
from accelerate import Accelerator
from datasets.utils import disable_progress_bars
from datasets.utils.logging import set_verbosity
from peft import LoraConfig, get_peft_model  # pyright: ignore
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import DataLoader

import wandb

from pdelfin.train.core.cli import make_cli, save_config, to_native_types
from pdelfin.train.core.config import TrainConfig
from pdelfin.train.core.loggers import get_logger
from pdelfin.train.core.paths import copy_dir, join_path
from pdelfin.train.core.state import BeakerState

from .utils import (
    RunName,
    get_local_dir,
    log_trainable_parameters,
    packing_collator,
    setup_environment,
)


from pdelfin.train.dataloader import make_dataset
from pdelfin.train.dataprep import batch_prepare_data_for_qwen2_training


class CheckpointUploadCallback(TrainerCallback):
    def __init__(self, save_path: str, logger: Optional[Logger] = None):
        self.save_path = save_path
        self.logger = logger or get_logger(self.__class__.__name__)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero:
            latest_checkpoint = get_last_checkpoint(args.output_dir)
            if not latest_checkpoint:
                return

            dir_name = Path(latest_checkpoint).name
            copy_dir(str(latest_checkpoint), f"{self.save_path}/{dir_name}")
            self.logger.info("Saved checkpoint to %s", f"{self.save_path}/{dir_name}")


def update_wandb_config(config: TrainConfig, trainer: Trainer, model: torch.nn.Module):
    # finding wandb callback
    callbacks = [c for c in trainer.callback_handler.callbacks if isinstance(c, WandbCallback)]  # pyright: ignore
    if not callbacks:
        raise ValueError("WandbCallback not found in trainer callbacks")

    wandb_callback = callbacks[0]
    peft_config = to_native_types(getattr(model, "peft_config", {}))
    script_config = to_native_types(config)
    beaker_envs = {k: v for k, v in os.environ.items() if k.lower().startswith("beaker")}

    on_setup_fn = wandb_callback.setup

    def setup_and_update(args, state, model, **kwargs):
        on_setup_fn(args=args, state=state, model=model, **kwargs)
        wandb.config.update({"peft": peft_config}, allow_val_change=True)
        wandb.config.update({"script": script_config}, allow_val_change=True)
        wandb.config.update({"beaker": beaker_envs}, allow_val_change=True)
        if (run := wandb.run) and (beaker_url := BeakerState().url):
            run.notes = beaker_url

    wandb_callback.setup = setup_and_update


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def run_train(config: TrainConfig):
    if get_rank() == 0:
        logger_level = logging.INFO
    else:
        logger_level = logging.WARN
        disable_progress_bars()

    logger = get_logger(__name__, level=logger_level)
    set_verbosity(logger_level)

    run_name = RunName.get(config)

    setup_environment(aws_config=config.aws, wandb_config=config.wandb, WANDB_RUN_GROUP=run_name.group)

    dataset = make_dataset(
        train_data_config=config.train_data,
        valid_data_config=config.valid_data,
        num_proc=config.num_proc,
        logger=logger,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model.name_or_path, torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if config.model.use_flash_attn else None
    )
    processor = AutoProcessor.from_pretrained(config.model.name_or_path)

    if config.lora is not None:
        peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,  # pyright: ignore
            task_type=config.lora.task_type,
            target_modules=list(config.lora.target_modules),
        )
        model = get_peft_model(model=model, peft_config=peft_config)
        log_trainable_parameters(model=model, logger=logger)

    formatted_dataset = dataset.with_transform(partial(batch_prepare_data_for_qwen2_training, processor=processor))
    print(formatted_dataset)
    print("---------------")
    
    save_path = join_path("", config.save.path, run_name.run)

    save_config(config, join_path("", save_path, "config.yaml"))  # pyright: ignore
 
    train_dataloader = DataLoader(formatted_dataset["train"], batch_size=1, num_workers=4, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    accelerator = Accelerator(mixed_precision="bf16")
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    steps = 0

    for entry in tqdm(train_dataloader):
        print("Sequence len", entry["input_ids"].shape)
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            outputs = model(**entry)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

            steps += 1
            if accelerator.is_local_main_process:
                logger.info(f"step {steps}, training loss : {loss.item()}")


def main():
    train_config = make_cli(TrainConfig)  # pyright: ignore
    run_train(train_config)


if __name__ == "__main__":
    main()
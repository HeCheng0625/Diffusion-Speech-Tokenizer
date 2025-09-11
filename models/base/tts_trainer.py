# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import numpy as np
import shutil
import torch
import time
from pathlib import Path
import torch
from tqdm import tqdm
import re
import logging
import json5
import accelerate
from accelerate.logging import get_logger
from models.base.base_dataset import batch_by_size
from accelerate.utils import ProjectConfiguration
from models.base.base_sampler import VariableSampler
from torch.utils.data import DataLoader
from models.base.new_trainer import BaseTrainer
from transformers import get_inverse_sqrt_schedule, get_constant_schedule


class TTSTrainer(BaseTrainer):

    def __init__(self, args=None, cfg=None):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        # init with accelerate
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        with self.accelerator.main_process_first():
            self.logger = get_logger(args.exp_name, log_level="INFO")

        # Log some info
        self.logger.info("=" * 56)
        self.logger.info("||\t\t" + "New training process started." + "\t\t||")
        self.logger.info("=" * 56)
        self.logger.info("\n")
        self.logger.debug(f"Using {args.log_level.upper()} logging level.")
        self.logger.info(f"Experiment name: {args.exp_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        self.checkpoint_backup_dir = os.path.join(self.exp_dir, "checkpoint_backup")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_backup_dir, exist_ok=True)
            self.logger.debug(
                f"Checkpoint backup directory: {self.checkpoint_backup_dir}"
            )

        self.checkpoint_backup_dir = os.path.join(self.exp_dir, "checkpoint_backup")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_backup_dir, exist_ok=True)
        self.logger.debug(f"Checkpoint backup directory: {self.checkpoint_backup_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        self.logger.info(
            "Max epoch: {}".format(
                self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
            )
        )

        # Check values
        if self.accelerator.is_main_process:
            self.__check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            self.logger.debug(
                f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
            )
            self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            self.logger.info(f"Building dataset done in {(end - start) / 1e6:.2f}ms")

        # save phone table to exp dir. Should be done before building model due to loading phone table in model
        if cfg.preprocess.use_phone and cfg.preprocess.phone_extractor != "lexicon":
            self._save_phone_symbols_file_to_exp_path()

        # setup model
        with self.accelerator.main_process_first():
            self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            self.logger.debug(self.model)
            self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
            self.logger.info(
                f"Model parameters: {self.__count_parameters(self.model)/1e6:.2f}M"
            )

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            self.logger.info(
                f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            self.logger.info(f"Building criterion done in {(end - start) / 1e6:.2f}ms")

        # Resume or Finetune
        with self.accelerator.main_process_first():
            self._check_resume()

        # accelerate prepare
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self._accelerator_prepare()
        end = time.monotonic_ns()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")
        self.device = self.accelerator.device

        if cfg.preprocess.use_spkid and cfg.train.multi_speaker_training:
            self.speakers = self._build_speaker_lut()
            self.utt2spk_dict = self._build_utt2spk_dict()

        # Only for TTS tasks
        self.task_type = "TTS"
        self.logger.info("Task type: {}".format(self.task_type))

    def _check_resume(self):
        # if args.resume:
        if self.args.resume or (
            self.cfg.model_type == "VALLE" and self.args.train_stage == 2
        ):
            checkpoint_dir = self.checkpoint_dir
            if self.cfg.model_type == "VALLE" and self.args.train_stage == 2:
                ls = [str(i) for i in Path(checkpoint_dir).glob("*")]
                if (
                    self.args.checkpoint_path is None or len(ls) == 0
                ):  # Train stage 2 from scratch using the checkpoint of stage 1
                    assert (
                        self.args.ar_model_ckpt_dir is not None
                    ), "Error: ar_model_ckpt_dir should be set to train nar model."
                    self.args.resume_type = "finetune"
                    checkpoint_dir = self.args.ar_model_ckpt_dir
                    self.logger.info(
                        f"Training NAR model at stage 2 using the checkpoint of AR model at stage 1."
                    )

            self.logger.info(f"Resuming from checkpoint: {checkpoint_dir}")
            start = time.monotonic_ns()
            self.ckpt_path = self._load_model(
                checkpoint_dir, self.args.checkpoint_path, self.args.resume_type
            )
            self.logger.info(f"Checkpoint path: {self.ckpt_path}")
            end = time.monotonic_ns()
            self.logger.info(
                f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
            )
            self.checkpoints_path = json.load(
                open(os.path.join(self.ckpt_path, "ckpts.json"), "r")
            )

    def _count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _accelerator_prepare(self):
        (
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.valid_dataloader,
        )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

    ### Following are methods only for TTS tasks ###
    def _build_dataset(self):
        pass

    def _build_criterion(self):
        pass

    def _build_model(self):
        pass

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")
            Dataset, Collator = self._build_dataset()
            if (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            ):
                train_dataset = Dataset(cfg=self.cfg)
            else:
                train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            np.random.seed(self.args.dataloader_seed)
            np.random.shuffle(batch_sampler)
            print(batch_sampler[:1])
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]

            train_loader = DataLoader(
                train_dataset,
                collate_fn=train_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
                prefetch_factor=32,
            )
            self.accelerator.wait_for_everyone()

            valid_loader = None

        else:
            print("Use Normal Batchsize......")
            Dataset, Collator = self._build_dataset()
            if (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            ):
                train_dataset = Dataset(cfg=self.cfg)
            else:
                train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            valid_loader = None
            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.train.adam,
        )
        return optimizer

    def _build_scheduler(self):
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            # num_warmup_steps=self.cfg.train.lr_warmup_steps,  # TODO: need to check wheather need to multiply by num_processes
            num_warmup_steps=self.cfg.train.lr_warmup_steps
            * self.accelerator.num_processes,
        )
        return lr_scheduler

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def _load_model(
        self,
        checkpoint_dir: str = None,
        checkpoint_path: str = None,
        resume_type: str = "",
    ):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            all_ckpts = os.listdir(checkpoint_dir)
            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            ls = list(all_ckpts)
            ls = [os.path.join(checkpoint_dir, i) for i in ls]
            ls.sort(key=lambda x: int(x.split("_")[-2].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]
            print("Resume from {}".format(checkpoint_path))

        if resume_type in ["resume", ""]:
            # Load all the things, including model weights, optimizer, scheduler, and random states.
            self.accelerator.load_state(input_dir=checkpoint_path)

            # set epoch and step
            self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1]) + 1
            self.step = int(checkpoint_path.split("_")[-2].split("-")[-1]) + 1

        elif resume_type == "finetune":
            # Load only the model weights
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            self.logger.info("Load model weights for finetune...")

        else:
            raise ValueError("Resume_type must be `resume` or `finetune`.")

        return checkpoint_path

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_step(self):
        pass

    def _valid_step(self, batch):
        pass

    def _inference(self):
        pass

    def _is_valid_pattern(self, directory_name):
        directory_name = str(directory_name)
        pattern = r"^epoch-\d{4}_step-\d{7}_loss-\d{1}\.\d{6}"
        return re.match(pattern, directory_name) is not None

    def _check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            self.logger.fatal("Invalid gradient_accumulation_step value!")
            self.logger.error(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )

    def __dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )

    def __check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            self.logger.fatal("Invalid gradient_accumulation_step value!")
            self.logger.error(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
        # TODO: check other values

    @staticmethod
    def __count_parameters(model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param

    def _build_speaker_lut(self):
        # combine speakers
        if not os.path.exists(os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)):
            speakers = {}
        else:
            with open(
                os.path.join(self.exp_dir, self.cfg.preprocess.spk2id), "r"
            ) as speaker_file:
                speakers = json.load(speaker_file)
        for dataset in self.cfg.dataset:
            speaker_lut_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.spk2id
            )
            with open(speaker_lut_path, "r") as speaker_lut_path:
                singer_lut = json.load(speaker_lut_path)
            for singer in singer_lut.keys():
                if singer not in speakers:
                    speakers[singer] = len(speakers)
        with open(
            os.path.join(self.exp_dir, self.cfg.preprocess.spk2id), "w"
        ) as speaker_file:
            json.dump(speakers, speaker_file, indent=4, ensure_ascii=False)
        print(
            "speakers have been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            )
        )
        return speakers

    def _build_utt2spk_dict(self):
        # combine speakers
        utt2spk = {}
        if not os.path.exists(os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk)):
            utt2spk = {}
        else:
            with open(
                os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk), "r"
            ) as utt2spk_file:
                for line in utt2spk_file.readlines():
                    utt, spk = line.strip().split("\t")
                    utt2spk[utt] = spk
        for dataset in self.cfg.dataset:
            utt2spk_dict_path = os.path.join(
                self.cfg.preprocess.processed_dir, dataset, self.cfg.preprocess.utt2spk
            )
            with open(utt2spk_dict_path, "r") as utt2spk_dict:
                for line in utt2spk_dict.readlines():
                    utt, spk = line.strip().split("\t")
                    if utt not in utt2spk.keys():
                        utt2spk[utt] = spk
        with open(
            os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk), "w"
        ) as utt2spk_file:
            for utt, spk in utt2spk.items():
                utt2spk_file.write(utt + "\t" + spk + "\n")
        print(
            "utterance and speaker mapper have been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.utt2spk)
            )
        )
        return utt2spk

    def _save_phone_symbols_file_to_exp_path(self):
        phone_symbols_file = os.path.join(
            self.cfg.preprocess.processed_dir,
            self.cfg.dataset[0],
            self.cfg.preprocess.symbols_dict,
        )
        phone_symbols_file_to_exp_path = os.path.join(
            self.exp_dir, self.cfg.preprocess.symbols_dict
        )
        shutil.copy(phone_symbols_file, phone_symbols_file_to_exp_path)
        os.chmod(phone_symbols_file_to_exp_path, 0o666)
        print(
            "phone symbols been dumped to {}".format(
                os.path.join(self.exp_dir, self.cfg.preprocess.symbols_dict)
            )
        )

    @torch.inference_mode()
    def _valid_step(self, batch):
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]
        x_mask = batch["mask"]

        phone_id = batch["phone_id"]
        phone_mask = batch["phone_mask"]

        semantic_code, _ = self._extract_semantic_code(
            input_features, attention_mask
        )  # if len(semantic_code) == 2: [B, T]; else 3: [N, B, T]

        seq_len = semantic_code.shape[-1]
        x_mask = x_mask[:, :seq_len]

        out = self.model(
            phone_ids=phone_id,
            phone_mask=phone_mask,
            target_ids=semantic_code,
            target_mask=x_mask,
        )

        total_loss += out.loss
        valid_losses["ce_loss"] = total_loss
        for item in valid_losses:
            valid_losses[item] = valid_losses[item].item()

        return (total_loss.item(), valid_losses, valid_stats)

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = dict()

        for batch in self.valid_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss = total_loss
            for key, value in valid_losses.items():
                epoch_losses[key] = value

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        ema_loss = None

        steps_to_skip = 0
        # if (
        #     hasattr(self.cfg.train, "resume_skip_steps")
        #     and self.cfg.train.resume_skip_steps
        #     and hasattr(self, "step")
        #     and self.step > 0
        # ):
        if hasattr(self, "step") and self.step >= 0:
            steps_to_skip = self.step
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Resume skip steps enabled, skipping first {steps_to_skip} steps..."
                )

            # 如果使用了动态batch size，我们需要修改batch_sampler
            if self.cfg.train.use_dynamic_batchsize:
                if hasattr(self.train_dataloader, "batch_sampler"):
                    self.train_dataloader.batch_sampler.skip_steps(steps_to_skip)
            # 如果使用了普通batch size，我们需要修改sampler
            else:
                if hasattr(self.train_dataloader, "sampler"):
                    # 计算需要跳过的样本数
                    samples_to_skip = (
                        steps_to_skip
                        * self.cfg.train.batch_size
                        * self.accelerator.num_processes
                    )
                    if isinstance(
                        self.train_dataloader.sampler,
                        torch.utils.data.DistributedSampler,
                    ):
                        self.train_dataloader.sampler.set_start_index(samples_to_skip)
                    elif hasattr(self.train_dataloader.sampler, "skip_samples"):
                        self.train_dataloader.sampler.skip_samples(samples_to_skip)

        # 跟踪当前处理的batch数
        current_batch = steps_to_skip

        for batch in self.train_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1
            ema_loss = (
                0.98 * ema_loss + 0.02 * self.current_loss
                if ema_loss is not None
                else self.current_loss
            )
            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (10 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

                if self.step % self.cfg.train.save_checkpoints_steps == 0:
                    self.save_checkpoint()

                if self.accelerator.is_main_process:
                    if self.step % 100 == 0:
                        print(f"EMA Loss: {ema_loss:.6f}")

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            # 读取self.checkpoint_dir所有的folder
            all_ckpts = os.listdir(self.checkpoint_dir)

            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))
            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))
            self.accelerator.save_state(path)
            self.logger.info("Finished saving state.")

            # if (
            #     hasattr(self.cfg.train, "save_checkpoints_backup_steps")
            #     and self.step % self.cfg.train.save_checkpoints_backup_steps == 0
            # ):
            if self.step % 100000 == 0:
                try:
                    backup_path = os.path.join(
                        self.checkpoint_backup_dir, checkpoint_filename
                    )
                    shutil.copytree(path, backup_path)
                    self.logger.info("Saving backup state to {}...".format(backup_path))
                except Exception as e:
                    self.logger.error("Failed to save backup state: {}".format(e))

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        # if self.accelerator.is_main_process:
        #     self._dump_cfg(self.config_save_path)

        # self.optimizer.zero_grad()

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 32)
                self.logger.info("Epoch {}: ".format(self.epoch))

            # Do training & validating epoch
            train_total_loss, train_losses = self._train_epoch()
            if isinstance(train_losses, dict):
                for key, loss in train_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                    self.accelerator.log(
                        {"Epoch/Train {} Loss".format(key): loss},
                        step=self.epoch,
                    )

            valid_total_loss, valid_losses = 0.0, 0.0
            # if isinstance(valid_losses, dict):
            #     for key, loss in valid_losses.items():
            #         if self.accelerator.is_main_process:
            #             self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
            #         self.accelerator.log(
            #             {"Epoch/Train {} Loss".format(key): loss},
            #             step=self.epoch,
            #         )

            if self.accelerator.is_main_process:
                self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss))
                self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss))
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss,
                    "Epoch/Valid Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_total_loss
                    ),
                )
            )
        self.accelerator.end_training()

    def echo_log(self, losses, mode="Training"):
        message = [
            "{} - Epoch {} Step {}: [{:.3f} s/step]".format(
                mode, self.epoch + 1, self.step, self.time_window.average
            )
        ]

        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(
                    str(key).split("/")[-1] + "=" + str(round(float(losses[key]), 5))
                )
        self.logger.info(", ".join(message))

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

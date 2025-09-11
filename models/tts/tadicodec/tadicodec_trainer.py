# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import torch
import numpy as np
import math
from utils.util import Logger, ValueWindow
from models.base.tts_trainer import TTSTrainer
import torch.nn.functional as F
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from models.tts.tadicodec.tadicodec_dataset import TadiCodecDataset, TadiCodecCollator
from models.codec.melvqgan.melspec import MelSpectrogram

from models.tts.tadicodec.modeling_tadicodec import TaDiCodec

from accelerate import InitProcessGroupKwargs
from datetime import timedelta


class TadiCodecTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                os.makedirs(os.path.join(self.exp_dir, "checkpoint"), exist_ok=True)
                self.log_file = os.path.join(
                    os.path.join(self.exp_dir, "checkpoint"), "train.log"
                )
                self.logger = Logger(self.log_file, level=self.args.log_level).logger

        self.time_window = ValueWindow(100)

        if self.accelerator.is_main_process:
            # Log some info
            self.logger.info("=" * 56)
            self.logger.info("||\t\t" + "New training process started." + "\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        self.checkpoint_backup_dir = os.path.join(self.exp_dir, "checkpoint_backup")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_backup_dir, exist_ok=True)
            self.logger.debug(
                f"Checkpoint backup directory: {self.checkpoint_backup_dir}"
            )

        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
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
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # setup model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

        self._build_mel_model()

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            self.train_dataloader = self.accelerator.prepare(
                self.train_dataloader,
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

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # Resume or Finetune
        try:
            with self.accelerator.main_process_first():
                if args.resume:
                    ## Automatically resume according to the current exprimental name
                    print(
                        "Automatically resuming from latest checkpoint in {}...".format(
                            self.checkpoint_dir
                        )
                    )
                    start = time.monotonic_ns()
                    ckpt_path = self._load_model(
                        checkpoint_dir=self.checkpoint_dir, resume_type=args.resume_type
                    )
                    end = time.monotonic_ns()
                    print(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
        except:
            print("Resume failed")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(180))
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
            kwargs_handlers=[kwargs],
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _build_model(self):
        model = TaDiCodec(cfg=self.cfg.model.tadicodec)
        return model

    def _build_mel_model(self):
        self.mel_model = MelSpectrogram(
            sampling_rate=self.cfg.preprocess.sample_rate,
            n_fft=self.cfg.preprocess.n_fft,
            num_mels=self.cfg.preprocess.num_mels,
            hop_size=self.cfg.preprocess.hop_size,
            win_size=self.cfg.preprocess.win_size,
            fmin=self.cfg.preprocess.fmin,
            fmax=self.cfg.preprocess.fmax,
        )
        self.mel_model.eval()
        self.mel_model.to(self.accelerator.device)

    @torch.no_grad()
    def _extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.cfg.preprocess.mel_mean) / math.sqrt(
            self.cfg.preprocess.mel_var
        )
        return mel_feature

    def _build_dataset(self):
        return TadiCodecDataset, TadiCodecCollator

    def _train_step(self, batch):
        train_losses = {}
        total_loss = 0
        train_stats = {}

        speech = batch["speech"]
        x_mask = batch["mask"]

        if (
            hasattr(self.cfg.model.tadicodec, "use_text_cond")
            and self.cfg.model.tadicodec.use_text_cond
        ):
            text_ids = batch["text_ids"]
            text_mask = batch["text_mask"]
        else:
            text_ids = None
            text_mask = None

        # ############################################################################
        # # Select samples to avoid OOM
        # ############################################################################
        # mel_len = x_mask.shape[1]
        # text_len = text_mask.shape[1] if text_mask is not None else 0
        # bsz = x_mask.shape[0]
        # total_len = mel_len + text_len
        # max_tokens = self.cfg.train.max_tokens

        # if bsz * total_len > max_tokens:
        #     select_num_samples = max_tokens // total_len
        #     start_index = np.random.randint(0, bsz - select_num_samples)
        #     speech = speech[start_index : start_index + select_num_samples]
        #     x_mask = x_mask[start_index : start_index + select_num_samples]
        #     if text_ids is not None:
        #         text_ids = text_ids[start_index : start_index + select_num_samples]
        #         text_mask = text_mask[start_index : start_index + select_num_samples]
        #     if self.cfg.preprocess.use_semantic_feat:
        #         input_features = input_features[
        #             start_index : start_index + select_num_samples
        #         ]
        #         attention_mask = attention_mask[
        #             start_index : start_index + select_num_samples
        #         ]
        # ############################################################################

        mel_feat = self._extract_mel_feature(speech)  # [B, T, d]

        seq_len = x_mask.shape[1]
        if mel_feat is not None:
            seq_len = min(seq_len, mel_feat.shape[1])
        x_mask = x_mask[:, :seq_len]
        if mel_feat is not None:
            mel_feat = mel_feat[:, :seq_len, :]

        x_in = None

        out = self.model(
            x=mel_feat,
            x_mask=x_mask,
            text_ids=text_ids,
            text_mask=text_mask,
            x_in=x_in,
        )

        (
            x,
            noise,
            flow_pred,
            final_mask,
            vq_loss,
            commit_loss,
            ssl_feat_pred,
        ) = (
            out["x"],
            out["noise"],
            out["flow_pred"],
            out["final_mask"],
            out["vq_loss"],
            out["commit_loss"],
            out["ssl_feat_pred"],
        )

        final_mask = final_mask.squeeze(-1)

        flow_gt = x - (1 - self.cfg.model.tadicodec.sigma) * noise

        # use l1 loss
        diff_loss = F.l1_loss(
            flow_pred, flow_gt, reduction="none"
        ).float() * final_mask.unsqueeze(-1)
        diff_loss = torch.mean(diff_loss, dim=2).sum() / final_mask.sum()

        total_loss += diff_loss
        train_losses["diff_loss"] = diff_loss

        # vq loss
        if (
            hasattr(self.cfg.model.tadicodec, "use_vq")
            and not self.cfg.model.tadicodec.use_vq
        ):
            pass
        else:
            total_loss += vq_loss
            train_losses["vq_loss"] = vq_loss

            train_losses["commit_loss"] = commit_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 0.5
            )
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        return (total_loss.item(), train_losses, train_stats)

    def _debug_gradients(self):
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if param.grad is None:
                print(f"Parameter {idx}: {name} has no gradient")

    def _check_need_grad(self):
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            if not param.requires_grad:
                print(f"Parameter {idx}: {name} has no gradient")
        return False

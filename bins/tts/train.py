import argparse
import torch
from models.tts.tadicodec.tadicodec_trainer import TadicodecTrainer


from utils.util import load_config


def build_trainer(args, cfg):
    supported_trainer = {
        "Tadicodec": TadicodecTrainer,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume", action="store_true", help="The model name to restore"
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="resume",
        help="Resume training or finetuning.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--dataloader_seed",
        type=int,
        default=1,
        help="Seed for dataloader",
    )

    # VALLETrainer.add_arguments(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Data Augmentation
    # if (
    #     type(cfg.preprocess.data_augment) == list
    #     and len(cfg.preprocess.data_augment) > 0
    # ):
    #     new_datasets_list = []
    #     for dataset in cfg.preprocess.data_augment:
    #         new_datasets = [
    #             f"{dataset}_pitch_shift" if cfg.preprocess.use_pitch_shift else None,
    #             (
    #                 f"{dataset}_formant_shift"
    #                 if cfg.preprocess.use_formant_shift
    #                 else None
    #             ),
    #             f"{dataset}_equalizer" if cfg.preprocess.use_equalizer else None,
    #             f"{dataset}_time_stretch" if cfg.preprocess.use_time_stretch else None,
    #         ]
    #         new_datasets_list.extend(filter(None, new_datasets))
    #     cfg.dataset.extend(new_datasets_list)

    # # CUDA settings
    cuda_relevant()

    # Build trainer
    trainer = build_trainer(args, cfg)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    trainer.train_loop()


if __name__ == "__main__":
    main()

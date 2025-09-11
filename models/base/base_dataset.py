# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import librosa
import torch
import json
import numpy as np
import logging
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class WarningFilter(logging.Filter):
    def filter(self, record):
        # 只过滤 phonemizer 中的 WARNING 级别日志
        if record.name == "phonemizer" and record.levelno == logging.WARNING:
            return False
        if record.name == "qcloud_cos.cos_client" and record.levelno == logging.INFO:
            return False
        if record.name == "jieba" and record.levelno == logging.DEBUG:
            return False
        return True


filter = WarningFilter()
logging.getLogger("phonemizer").addFilter(filter)
logging.getLogger("jieba").addFilter(filter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import regex


def has_valid_text(text, valid_langs=["zh", "en", "ja", "ko", "de", "fr"]):
    """
    检查字符串是否包含指定语言的文字

    Args:
        text: 要检查的字符串
        valid_langs: 有效的语言代码列表

    Returns:
        bool: 是否包含有效文字
    """
    patterns = {
        "zh": r"\p{Han}",  # 中文汉字
        "en": r"[a-zA-Z]",  # 英文字母
        "ja": r"[\p{Hiragana}\p{Katakana}]",  # 日文平假名和片假名
        "ko": r"[\p{Hangul}]",  # 韩文谚文
        "de": r"[a-zA-ZäöüßÄÖÜ]",  # 德文字母（包括变音符号）
        "fr": r"[a-zA-ZéèêëàâîïôûùüÿçœæÉÈÊËÀÂÎÏÔÛÙÜŸÇŒÆ]",  # 法文字母（包括重音符号）
    }

    # 组合所有需要检查的语言的正则表达式
    combined_pattern = "|".join(patterns[lang] for lang in valid_langs)

    # 检查是否匹配任何一种语言的文字
    return bool(regex.search(combined_pattern, text))


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_type="path",
        cfg=None,
    ):

        self.cfg = cfg
        self.cache_type = cache_type

        self.frame_rate = (
            self.cfg.preprocess.frame_rate
            if hasattr(self.cfg.preprocess, "frame_rate")
            else 50
        )

        self.json_paths = []
        self.wav_paths = []
        self.wav_path_index2duration = []
        self.wav_path_index2phonelen = []
        self.index2num_frames = []

        self.json_path2meta = {}
        self.json2filtered_idx = {}

        self.mnt_path = self.cfg.preprocess.mnt_path
        self.cache_folder = self.cfg.preprocess.cache_folder

        logger.info(f"mnt_path: {self.mnt_path}, cache_folder: {self.cache_folder}")

        if hasattr(self.cfg.preprocess, "use_json_path_cache"):
            self.use_json_path_cache = self.cfg.preprocess.use_json_path_cache
        else:
            self.use_json_path_cache = False

        self.wav_paths_cache = os.path.join(self.cache_folder, "wav_paths_cache.pkl")
        self.duration_cache = os.path.join(self.cache_folder, "duration_cache.pkl")
        if os.path.exists(os.path.join(self.cache_folder, "bpe_token_count_cache.pkl")):
            self.phone_count_cache = os.path.join(
                self.cache_folder, "bpe_token_count_cache.pkl"
            )  # 改用bpe token count
        else:
            self.phone_count_cache = os.path.join(
                self.cache_folder, "phone_count_cache.pkl"
            )
        self.json_paths_cache = os.path.join(self.cache_folder, "json_paths_cache.pkl")

        self.duration_setting = {"min": 1, "max": 40}
        if hasattr(self.cfg.preprocess, "min_dur"):
            self.duration_setting["min"] = self.cfg.preprocess.min_dur
        if hasattr(self.cfg.preprocess, "max_dur"):
            self.duration_setting["max"] = self.cfg.preprocess.max_dur

        if cache_type == "path":
            self.load_cached_paths()
        else:
            logger.info("Incorrect cache loading way")
            exit()

        self.lang2id = {"en": 1, "zh": 2, "ja": 3, "fr": 4, "ko": 5, "de": 6}
        self.id2lang = {v: k for k, v in self.lang2id.items()}

        if (
            hasattr(self.cfg.preprocess, "use_lang_offset")
            and self.cfg.preprocess.use_lang_offset
        ):
            self.use_lang_offset = True
            self.lang_offset = self.cfg.preprocess.lang_offset
            logger.info("Using lang offset")
        else:
            self.use_lang_offset = False

        if hasattr(self.cfg.preprocess, "tokenizer_path"):
            tokenizer_path = self.cfg.preprocess.tokenizer_path
        else:
            tokenizer_path = "./ckpt/TaDiCodec/text_tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.num_frame_indices = np.array(
            sorted(
                range(len(self.index2num_frames)),
                key=lambda k: self.index2num_frames[k],
            )
        )

    def load_cached_paths(self):
        logger.info("Loaded paths from cache files")
        with open(self.wav_paths_cache, "rb") as f:
            self.wav_paths = pickle.load(f)
        if self.cache_type == "path":
            with open(self.duration_cache, "rb") as f:
                self.wav_path_index2duration = pickle.load(f)
            with open(self.phone_count_cache, "rb") as f:
                self.wav_path_index2phonelen = pickle.load(f)
            for duration, phone_count in zip(
                self.wav_path_index2duration, self.wav_path_index2phonelen
            ):
                if self.cfg.preprocess.use_text:
                    self.index2num_frames.append(
                        duration * self.frame_rate + phone_count
                    )
                else:
                    self.index2num_frames.append(duration * self.frame_rate)
            if self.use_json_path_cache:
                with open(self.json_paths_cache, "rb") as f:
                    self.json_paths = pickle.load(f)

        logger.info("All paths got successfully")
        logger.info("Number of wavs: %d" % (len(self.wav_paths)))

        self.emilia_size = len(self.wav_paths)

    def get_meta_from_wav_path(self, wav_path):
        index = int(wav_path.split("_")[-1].split(".")[0])
        audio_name = "_".join(wav_path.split("/")[-1].split("_")[:-1])
        dir_name = "/".join(wav_path.split("/")[:-1])
        json_name = audio_name + ".json"
        json_path = dir_name + "/" + json_name
        meta = None
        if self.cache_type == "meta":
            meta = self.json_path2meta[json_path][str(index)]
            return meta
        elif self.cache_type == "path":
            try:
                if "mls_english_opus" in wav_path:
                    meta = json.load(open(json_path))[os.path.basename(wav_path)]
                else:
                    meta = json.load(open(json_path))[index]
            except Exception as e:
                logger.info("Error json: {} error: {}".format(json_path, e))
        del index, audio_name, dir_name, json_name, json_path
        return meta

    def get_meta_from_idx(self, idx):
        meta = None
        if self.cache_type == "path":
            try:
                meta = self.json_paths[idx]
            except Exception as e:
                logger.info("Error {}".format(e))
        return meta

    def __len__(self):
        return self.wav_paths.__len__()

    def get_num_frames(self, index):
        if self.cfg.preprocess.use_text:
            return (
                self.wav_path_index2duration[index] * self.frame_rate
                + self.wav_path_index2phonelen[index]
            )
        else:
            return self.wav_path_index2duration[index]

    def _get_another(self, idx):
        position = np.where(self.num_frame_indices == idx)[0][0]
        if position == 0:
            position = min(len(self.num_frame_indices), 10000)
        random_index = np.random.choice(self.num_frame_indices[:position])
        del position
        return self.__getitem__(random_index)

    def _get_single_feature(self, speech, text, language):

        single_feature = dict()

        speech_16k, speech_frames, mask = self._reprocess_speech(speech)

        lang_id = self.lang2id[language]

        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        text_mask = [1] * len(text_ids)

        single_feature.update(
            {
                "text_ids": text_ids,
                "text_mask": text_mask,
                "lang_id": lang_id,
                "speech": speech,
                "mask": mask,
            }
        )

        return single_feature

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        name = os.path.splitext(os.path.basename(wav_path))[0]
        wav_path = os.path.join(self.mnt_path, wav_path.replace("_new", ""))
        if self.use_json_path_cache:
            meta = self.get_meta_from_idx(idx)
        else:
            meta = self.get_meta_from_wav_path(wav_path)

        try:
            if os.path.exists(wav_path) and meta is not None:
                try:
                    speech, sr = librosa.load(
                        wav_path, sr=self.cfg.preprocess.sample_rate
                    )
                    if (
                        speech.shape[-1]
                        > self.duration_setting["max"] * self.cfg.preprocess.sample_rate
                    ):
                        return self._get_another(idx)
                except Exception as e:
                    logger.error(f"Failed to load file. Get another. {e}, {wav_path}")
                    return self._get_another(idx)

                text = meta["text"]
                language = meta["language"]

                single_feature = self._get_single_feature(speech, text, language)

                if len(single_feature["text_ids"]) > 512:
                    logger.info("Input ids too long. Get another.")
                    return self._get_another(idx)

                return single_feature

            else:
                logger.error(f"Failed to get file after retries. {wav_path}")
                return self._get_another(idx)
        except Exception as e:
            logger.error(f"error: {e}, {wav_path}")
            return self._get_another(idx)

    def _reprocess_speech(self, speech):
        # resample the speech to 16k for feature extraction
        if self.cfg.preprocess.sample_rate != 16000:
            speech_16k = librosa.resample(
                speech, orig_sr=self.cfg.preprocess.sample_rate, target_sr=16000
            )
        else:
            speech_16k = speech

        speech_frames = len(speech) // self.cfg.preprocess.hop_size
        mask = np.ones(speech_frames)
        return speech_16k, speech_frames, mask


class BaseCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "speech":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).float() for utt in batch], batch_first=True
                )
            if key == "mask":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).long() for utt in batch], batch_first=True
                )
            if key == "text_ids":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).long() for utt in batch],
                    batch_first=True,
                    padding_value=32000,
                )
            if key == "text_mask":
                packed_batch_features[key] = pad_sequence(
                    [torch.tensor(utt[key]).long() for utt in batch], batch_first=True
                )
            else:
                pass

        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches

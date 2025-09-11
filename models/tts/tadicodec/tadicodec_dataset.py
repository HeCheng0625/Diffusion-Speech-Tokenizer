# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from models.base.base_dataset import BaseDataset, BaseCollator


class WarningFilter(logging.Filter):
    def filter(self, record):
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


class TadiCodecDataset(BaseDataset):
    def __init__(self, cache_type="path", cfg=None):
        super().__init__(cache_type, cfg)


class TadiCodecCollator(BaseCollator):
    def __init__(self, cfg):
        super().__init__(cfg)

import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callback import *

from pytorch_pretrained_bert import BertTokenizer


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(set, key, val)

config = Config(
    testing=False,
    bert_model_name='bert-base-chinese',
    max_lr=3e-5,
    epochs=100,
    use_fp16=False,
    bs=8,
    max_seq_len=128
)

bert_tok = BertTokenizer.from_pretrained(
    config.bert_model_name
)

class FastAiBertTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str) -> List[str]:
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len-2] + ["[SEP]"]

fastai_tokenizer = Tokenizer(
    tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len),
    pre_rules=[],
    post_rules=[]
)
fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))

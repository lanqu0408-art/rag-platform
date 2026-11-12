# encoding=utf-8
import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_HANLP_HOME = REPO_ROOT / "models" / "hanlp_home"
DEFAULT_HANLP_URL = "https://file.hankcs.com/hanlp/"


def ensure_hanlp_env() -> str:
    """
    Force HanLP to use the repo-local model directory first, so startup doesn't
    repeatedly attempt remote downloads.
    """
    os.environ["HANLP_HOME"] = str(LOCAL_HANLP_HOME)
    os.environ["HANLP_URL"] = DEFAULT_HANLP_URL
    return os.environ["HANLP_HOME"]


@lru_cache(maxsize=1)
def _hanlp():
    ensure_hanlp_env()
    import hanlp  # local import to avoid import-time side effects

    return hanlp


@lru_cache(maxsize=1)
def get_chinese_tokenizers() -> Tuple[object, object]:
    hanlp = _hanlp()
    coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
    return coarse, fine


@lru_cache(maxsize=1)
def get_sentence_splitter():
    hanlp = _hanlp()
    return hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)

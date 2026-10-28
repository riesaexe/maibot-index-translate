"""漫画日文识别：使用固定版本模型，保留竖排阅读顺序。"""
from pathlib import Path
from functools import cache
from threading import Lock
from typing import Final, TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from manga_ocr import MangaOcr

REVISION: Final = 'aa6573bd10b0d446cbf622e29c3e084914df9741'
_lock = Lock()


def japanese_text(image: Image.Image) -> str:
    """在线程池中调用；模型只加载一次，同步保护推理。"""
    with _lock:
        return reader()(image)


@cache
def reader() -> 'MangaOcr':
    from huggingface_hub import snapshot_download
    from manga_ocr import MangaOcr
    import torch

    folder = Path(__file__).parent / '.models' / 'manga-ocr'
    if not (folder / 'pytorch_model.bin').exists():
        snapshot_download('kha-white/manga-ocr-base', revision=REVISION, local_dir=str(folder),
                          allow_patterns=['*.json', 'vocab.txt', 'pytorch_model.bin'])
    torch.set_num_threads(2)
    return MangaOcr(str(folder), force_cpu=True)

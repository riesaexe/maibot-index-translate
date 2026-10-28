"""漫画日文识别：使用固定版本模型，保留竖排阅读顺序。"""
from functools import cache
from threading import Lock
from typing import Final, TYPE_CHECKING

from PIL import Image
from .model_cache import model_path

if TYPE_CHECKING:
    from manga_ocr import MangaOcr

REVISION: Final = 'aa6573bd10b0d446cbf622e29c3e084914df9741'
_lock = Lock()


def japanese_text(image: Image.Image) -> str:
    """在线程池中调用；模型只加载一次，同步保护推理。"""
    with _lock:
        return reader()(image)


def prewarm() -> None:
    """执行一次本地空白图推理，复用真实识别的加载锁和模型缓存。"""
    japanese_text(Image.new('RGB', (256, 64), 'white'))


@cache
def reader() -> 'MangaOcr':
    from huggingface_hub import snapshot_download
    from manga_ocr import MangaOcr
    import torch

    folder = model_path('manga-ocr')
    if not (folder / 'pytorch_model.bin').exists():
        snapshot_download('kha-white/manga-ocr-base', revision=REVISION, local_dir=str(folder),
                          allow_patterns=['*.json', 'vocab.txt', 'pytorch_model.bin'])
    torch.set_num_threads(2)
    return MangaOcr(str(folder), force_cpu=True)

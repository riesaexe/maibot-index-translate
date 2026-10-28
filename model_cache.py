"""模型存入宿主运行期目录，兼容旧版源码目录中的下载。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

_root = Path('data/plugins/riesa.translate/models')
_legacy = Path(__file__).parent / '.models'


def configure(data_dir: Path) -> None:
    """由 on_load 注入宿主分配的数据目录。"""
    global _root
    _root = data_dir / 'models'


def model_path(name: str) -> Path:
    """在线程池使用；仅在目标不存在时复用旧文件，不删除旧缓存。"""
    _root.mkdir(parents=True, exist_ok=True)
    target = _root / name
    old = _legacy / name
    if not target.exists() and old.exists():
        with TemporaryDirectory(dir=_root) as temporary:
            staged = Path(temporary) / name
            if old.is_dir():
                shutil.copytree(old, staged)
            else:
                shutil.copy2(old, staged)
            staged.replace(target)
    return target

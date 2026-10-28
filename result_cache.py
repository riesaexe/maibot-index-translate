"""仅内存缓存成功译图；引用消息由每次请求重新构造。"""
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class CachedImage:
    image: str
    unreadable: int
    created: float


class ResultCache:
    """按访问顺序淘汰，最多 4 张、24 MiB、10 分钟。"""

    def __init__(self) -> None:
        self.items: OrderedDict[str, CachedImage] = OrderedDict()

    def clear(self) -> None:
        self.items.clear()

    def get(self, key: str) -> CachedImage | None:
        item = self.items.get(key)
        if item is None:
            return None
        if monotonic() - item.created > 600:
            del self.items[key]
            return None
        self.items.move_to_end(key)
        return item

    def put(self, key: str, image: str, unreadable: int) -> None:
        if len(image) > 24 * 1024 * 1024:
            return
        self.items[key] = CachedImage(image, unreadable, monotonic())
        self.items.move_to_end(key)
        while len(self.items) > 4 or sum(len(item.image) for item in self.items.values()) > 24 * 1024 * 1024:
            self.items.popitem(last=False)

"""同步模型的后台预热；不占用宿主事件循环。"""
from collections.abc import Callable
from logging import Logger
from threading import Event, Thread
import time


class ModelWarmup:
    """生命周期持有的单次后台任务，停止后不再启动后续模型。"""

    def __init__(self, logger: Logger, is_idle: Callable[[], bool],
                 stages: tuple[tuple[str, Callable[[], None]], ...],
                 delay: float = 30, poll: float = 5) -> None:
        self.logger = logger
        self.is_idle = is_idle
        self.stages = stages
        self.delay = delay
        self.poll = poll
        self._stop = Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        Thread(target=self._run, name='translate-model-warmup', daemon=True).start()

    def stop(self) -> None:
        # 原生模型加载不能安全强杀；当前步骤结束后停止，不再加载下一模型。
        self._stop.set()

    def _run(self) -> None:
        if self._stop.wait(self.delay):
            return
        for name, load in self.stages:
            while not self.is_idle():
                if self._stop.wait(self.poll):
                    return
            if self._stop.is_set():
                return
            started = time.monotonic()
            self.logger.info('开始预热%s模型；缓存缺失时将下载', name)
            try:
                load()
            except Exception as exc:  # noqa: BLE001 — 后台任务边界，记录错误而不影响正常按需加载。
                self.logger.warning('%s模型预热失败，正常翻译仍可按需重试：%s', name, exc)
            else:
                self.logger.info('%s模型预热完成，耗时 %.2fs', name, time.monotonic() - started)

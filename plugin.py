"""聊天文字和图片翻译为中文。"""

from typing import Dict, Optional

import re
import hashlib
import time

import anyio
import httpx

from maibot_sdk import Command, HookHandler, MaiBotPlugin
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder
from pydantic import JsonValue, ValidationError

from .image_request import single_payload, comic_payload
from .image_geometry import rectify
from .image_detection import chinese_only
from .result_cache import ResultCache
from .commands import DEFAULT_COMMAND, command_pattern
from .config import PluginOptions as PluginOptions, TranslationConfig
from .image_translation import UNREADABLE, Page, TranslationMappingError, batches, detect, parse_translations, payload, render
from .translation import COMMAND_PATTERN, Incoming, Input, TranslationError, build_payload, extract, parse_response


class TranslatePlugin(MaiBotPlugin):
    config_model = TranslationConfig

    def __init__(self) -> None:
        super().__init__()
        self.busy: set[str] = set()
        self.images: Dict[tuple[str, str], tuple[float, Incoming]] = {}
        self.results = ResultCache()
        self.region_limiter = anyio.CapacityLimiter(3)
        self.registered_command: str | None = None

    def trigger(self) -> str:
        return self.registered_command or self.configured_trigger()

    def configured_trigger(self) -> str:
        try:
            return TranslationConfig.model_validate(self.config.model_dump()).plugin.command
        except RuntimeError:  # 宿主在配置注入前允许枚举组件。
            return DEFAULT_COMMAND

    def get_components(self) -> list[dict[str, JsonValue]]:
        components = super().get_components()
        self.registered_command = self.configured_trigger()
        for component in components:
            if component['name'] == 'translate' and component['type'] == 'COMMAND':
                component['metadata'].update(command_pattern=command_pattern(self.registered_command),
                                             aliases=[self.registered_command],
                                             description=f'{self.registered_command}：将文字或图片翻译为中文，支持引用消息')
        return components

    async def on_load(self) -> None:
        self.ctx.logger.info('中文翻译已加载，命令：%s', self.trigger())

    async def on_unload(self) -> None:
        self.images.clear()
        self.results.clear()

    async def on_config_update(self, scope: str, config_data: Dict[str, JsonValue], version: str) -> None:
        # SDK 在调用前已更新 self.config；后续命令直接读取新配置。
        del scope, config_data, version
        self.images.clear()
        self.results.clear()

    @HookHandler("chat.receive.after_process", name="capture_translation_image",
                 description="规范引用命令匹配文本并短暂保留直接附图原图", mode=HookMode.BLOCKING,
                 order=HookOrder.EARLY, error_policy=ErrorPolicy.SKIP)
    async def capture(self, message: Optional[Dict[str, JsonValue]] = None, **kwargs: JsonValue) -> Optional[Dict[str, JsonValue]]:
        now = time.monotonic()
        self.images = {key: value for key, value in self.images.items() if now - value[0] < 60}
        if message is None:
            return
        incoming = Incoming.model_validate(message)
        text = "".join(s.data for s in incoming.raw_message if s.type == "text" and isinstance(s.data, str))
        if not re.match(command_pattern(self.trigger()), text):
            return None
        stream = message.get("session_id", "")
        if incoming.has_images and isinstance(stream, str):
            if len(self.images) >= 16:
                self.images.pop(next(iter(self.images)))
            self.images[(stream, incoming.message_id)] = (now, incoming)
        # 宿主匹配 processed_plain_text，其中可能先出现 @、引用正文或 [image]。
        # 只在原始文本段明确以命令开头时规范匹配文本，引用 ID、原图与路由原样保留。
        normalized = dict(message)
        normalized['processed_plain_text'] = text
        return {'modified_kwargs': {**kwargs, 'message': normalized}}

    async def load_message(self, message_id: str, stream_id: str) -> Incoming:
        data = await self.ctx.message.get_by_id(message_id, stream_id=stream_id, include_binary_data=True)
        if data is None:
            raise TranslationError("未找到消息，可能尚未保存或已过期，请直接发送原文或图片。")
        return Incoming.model_validate(data)

    async def source(self, incoming: Incoming, stream_id: str) -> Input:
        # 当前命令尚未落库，原图由预处理 Hook 提供；引用消息走会话限定查询。
        if incoming.has_images:
            cached = self.images.pop((stream_id, incoming.message_id), None)
            if cached is None or time.monotonic() - cached[0] >= 60:
                raise TranslationError(f"图片已过期或未读取到，请重新发送图片并附上 {self.trigger()}。")
            incoming = cached[1]
        direct = await anyio.to_thread.run_sync(lambda: extract(incoming, command=True, trigger=self.trigger()))
        if incoming.reply_id:
            quoted = await self.load_message(incoming.reply_id, stream_id)
            reference = await anyio.to_thread.run_sync(lambda: extract(quoted))
            combined = Input(text="\n".join(t for t in (direct.text, reference.text) if t),
                             images=direct.images + reference.images)
        else:
            combined = direct
        if len(combined.text) > 12000 or len(combined.images) > 4:
            raise TranslationError("每次最多翻译 12000 个字符和 4 张图片，请分批发送。")
        if not combined.text and not combined.images:
            raise TranslationError(f"用法：{self.trigger()} 原文；也可附带图片，或引用文字/图片消息发送 {self.trigger()}。")
        return combined

    async def request(self, client: httpx.AsyncClient, body: Dict[str, JsonValue]) -> str:
        response = await client.post(
            "https://index-translate.bilibili.com/v1/chat/completions",
            json=body, headers={"User-Agent": "Index-Translate-Client/1.0"},
        )
        if response.status_code in {502, 503, 504}:
            await anyio.sleep(1)
            response = await client.post(
                "https://index-translate.bilibili.com/v1/chat/completions",
                json=body, headers={"User-Agent": "Index-Translate-Client/1.0"},
            )
        response.raise_for_status()
        return parse_response(response.json())

    async def reply(self, incoming: Incoming, segment: Dict[str, JsonValue], stream_id: str) -> bool:
        segments = [incoming.reply_segment()] if incoming.reply_id or incoming.message_id else []
        if segment.get("type") == "image":
            # 保留图文结构以兼容 QQ PC 引用显示，零宽空格不增加可见文案。
            segments.append({"type": "text", "content": "\u200b"})
        segments.append(segment)
        return bool(await self.ctx.send.hybrid(segments, stream_id))

    async def translate_regions(self, client: httpx.AsyncClient, page: Page, model: str) -> tuple[str, ...]:
        if len(page.regions) > 1 and any(region.quad or region.comic for region in page.regions):
            values = [''] * len(page.regions)
            errors: list[TranslationError | httpx.HTTPError | ValueError] = []
            async def translate_one(index: int) -> None:
                try:
                    async with self.region_limiter:
                        values[index] = (await self.translate_regions(client, Page(page.image, (page.regions[index],)), model))[0]
                except (TranslationError, httpx.HTTPError, ValueError) as exc:
                    errors.append(exc)
                    group.cancel_scope.cancel()
            async with anyio.create_task_group() as group:
                for index in range(len(page.regions)):
                    group.start_soon(translate_one, index)
            if errors:
                raise errors[0]
            return tuple(values)
        if len(page.regions) == 1:
            region = page.regions[0]
            if region.comic:
                if region.text:
                    if not re.search(r'[\w\u3040-\u30ff]', region.text):
                        return (region.text,)
                    value = await self.request(client, build_payload(Input(text=region.text, images=()), model))
                    return (value,)
                body = await anyio.to_thread.run_sync(comic_payload, page.image, region.box, model, region.japanese)
                value = await self.request(client, body)
                if re.search(r'[A-Za-z]{2}', value) and not re.search(r'[\u3400-\u9fff]', value):
                    value = await self.request(client, build_payload(Input(text=value.strip(), images=()), model))
                return (UNREADABLE if UNREADABLE in value else value,)
            if chinese_only(region.text):
                return (region.text,)
            image = await anyio.to_thread.run_sync(rectify, page.image, region.quad) if region.quad else page.image
            if region.quad and abs(region.quad[1][1] - region.quad[0][1]) > abs(region.quad[1][0] - region.quad[0][0]):
                # 竖向单词检测框常带入上下相邻行的残片，只读居中的完整单词。
                image = await anyio.to_thread.run_sync(image.crop, (0, round(image.height * .18), image.width, round(image.height * .82)))
            box = (0, 0, image.width, image.height) if region.quad else region.box
            body = await anyio.to_thread.run_sync(single_payload, image, box, model)
            value = await self.request(client, body)
            if region.quad and abs(region.quad[1][1] - region.quad[0][1]) > abs(region.quad[1][0] - region.quad[0][0]) and re.fullmatch(r"[A-Za-z][A-Za-z' -]{1,60}", value.strip()):
                # 视觉服务有时只抄出英文单词；使用它读出的文本再执行明确的中文翻译。
                value = await self.request(client, build_payload(Input(text=value.strip(), images=()), model))
            return (UNREADABLE if UNREADABLE in value else value,)
        body = await anyio.to_thread.run_sync(payload, page, model)
        result = await self.request(client, body)
        try:
            return parse_translations(result, len(page.regions))
        except TranslationMappingError:
            middle = len(page.regions) // 2
            left = await self.translate_regions(client, Page(page.image, page.regions[:middle]), model)
            right = await self.translate_regions(client, Page(page.image, page.regions[middle:]), model)
            return left + right

    @Command("translate", description="/rs 翻译：将文字或图片翻译为中文，支持引用消息",
             pattern=COMMAND_PATTERN, timeout_ms=360_000)
    async def translate(self, stream_id: str = "", message: Optional[Dict[str, JsonValue]] = None,
                        **kwargs: JsonValue) -> tuple[bool, str, int]:
        del kwargs
        cfg = TranslationConfig.model_validate(self.config.model_dump()).plugin
        if not cfg.enabled:
            return False, "翻译插件已停用", 1
        if not stream_id:
            return False, "缺少聊天上下文", 1
        try:
            incoming = Incoming.model_validate(message or {})
        except ValidationError:
            await self.ctx.send.text("原始消息格式异常，请重新发送。", stream_id)
            return False, "原始消息格式异常", 1
        if stream_id in self.busy:
            await self.reply(incoming, {"type": "text", "content": "本聊天已有翻译正在进行，请稍候。"}, stream_id)
            return False, "翻译忙碌", 1
        self.busy.add(stream_id)
        try:
            with anyio.fail_after(330):
                if message is None:
                    raise TranslationError("未收到原始消息，请重试。")
                source = await self.source(incoming, stream_id)
                # 沿用宿主 pyproject.toml 中已选择的 httpx，不更改主程序依赖。
                async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10), follow_redirects=False) as client:
                    translated_images = []
                    unreadable = 0
                    for url in source.images:
                        key = hashlib.sha256((cfg.model + cfg.comic_detector + cfg.font_path + '\0' + url).encode()).hexdigest()
                        cached = self.results.get(key)
                        if cached is not None:
                            unreadable += cached.unreadable
                            translated_images.append(cached.image)
                            self.ctx.logger.info('译图命中内存缓存')
                            continue
                        started = time.monotonic()
                        page = await anyio.to_thread.run_sync(detect, url, cfg.comic_detector)
                        detected = time.monotonic()
                        translations = []
                        for part in batches(page):
                            translations.extend(await self.translate_regions(client, part, cfg.model))
                        unreadable += translations.count(UNREADABLE)
                        translated = time.monotonic()
                        image = await anyio.to_thread.run_sync(render, page, tuple(translations), cfg.font_path)
                        self.ctx.logger.info('译图耗时：定位识别 %.2fs，翻译 %.2fs，排版 %.2fs',
                                             detected - started, translated - detected, time.monotonic() - translated)
                        self.results.put(key, image, translations.count(UNREADABLE))
                        translated_images.append(image)
                    result = await self.request(client, build_payload(Input(text=source.text), cfg.model)) if source.text else ""
                for image in translated_images:
                    if not await self.reply(incoming, {"type": "image", "content": image}, stream_id):
                        return False, "译图发送失败", 1
                for start in range(0, len(result), 1800):
                    if not await self.reply(incoming, {"type": "text", "content": result[start:start + 1800]}, stream_id):
                        return False, "译文发送失败", 1
                if unreadable:
                    await self.reply(incoming, {"type": "text", "content": f"有 {unreadable} 处文字未能可靠辨认，已保留原图中的原文。"}, stream_id)
            return True, "翻译完成", 1
        except TranslationError as exc:
            reason = str(exc)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            reason = {
                429: "翻译服务请求过于频繁，请稍后重试。",
                504: "翻译服务网关超时（HTTP 504），重试后仍未恢复，请稍后再试。",
                502: "翻译服务网关异常（HTTP 502），重试后仍未恢复，请稍后再试。",
                503: "翻译服务繁忙（HTTP 503），重试后仍未恢复，请稍后再试。",
            }.get(status, f"翻译请求失败（HTTP {status}），请稍后重试。")
            self.ctx.logger.warning("翻译服务 HTTP 状态：%s", status)
        except (TimeoutError, httpx.TimeoutException):
            reason = "翻译请求超时，请稍后重试。"
        except httpx.RequestError:
            reason = "无法连接翻译服务，请稍后重试。"
        except (ValidationError, ValueError):
            reason = "消息或翻译响应格式异常，请重新发送后重试。"
        finally:
            self.busy.discard(stream_id)
        await self.reply(incoming, {"type": "text", "content": reason}, stream_id)
        return False, reason, 1


def create_plugin() -> TranslatePlugin:
    return TranslatePlugin()

"""Index-Translate 请求与消息解析。"""

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Union

import base64
import binascii
import re

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError


COMMAND_PATTERN = r"^\s*/rs\s+翻译(?=\s|\[|$)[\s\S]*$"
PREFIX = re.compile(r"^\s*/rs\s+翻译(?=\s|$)\s*")


class TranslationError(Exception):
    """可直接向用户展示的翻译错误。"""


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str
    data: Union[str, Dict[str, JsonValue]] = ""
    binary_data_base64: str = ""


class Incoming(BaseModel):
    model_config = ConfigDict(frozen=True)
    message_id: str = ""
    raw_message: List[Segment] = Field(default_factory=list)
    processed_plain_text: str = ""
    reply_to: Optional[str] = None
    message_info: Dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def reply_id(self) -> str:
        for segment in self.raw_message:
            if segment.type == "reply":
                if isinstance(segment.data, str):
                    return segment.data
                value = segment.data.get("target_message_id", "")
                if isinstance(value, str):
                    return value
        return self.reply_to or ""

    @property
    def has_images(self) -> bool:
        return any(s.type in {"image", "emoji"} for s in self.raw_message)

    def reply_segment(self) -> Dict[str, JsonValue]:
        target = self.reply_id or self.message_id
        data: Dict[str, JsonValue] = {"target_message_id": target}
        if self.reply_id:
            for segment in self.raw_message:
                if segment.type == "reply" and isinstance(segment.data, dict):
                    data.update(segment.data)
                    break
        else:
            data["target_message_content"] = self.processed_plain_text[:300] or (
                "".join(s.data for s in self.raw_message if s.type == "text" and isinstance(s.data, str))[:300]
            )
            user = self.message_info.get("user_info")
            if isinstance(user, dict):
                for key in ("user_id", "user_nickname", "user_cardname"):
                    value = user.get(key)
                    if isinstance(value, str):
                        data[{"user_id": "target_message_sender_id", "user_nickname": "target_message_sender_nickname",
                              "user_cardname": "target_message_sender_cardname"}[key]] = value
        return {"type": "reply", "data": data}


@dataclass(frozen=True, slots=True)
class Input:
    text: str = ""
    images: tuple[str, ...] = ()


def image_url(encoded: str) -> str:
    if len(encoded) > 14_000_000:
        raise TranslationError("单张图片不能超过 10 MB。")
    try:
        binary = base64.b64decode(encoded, validate=True)
        if len(binary) > 10_000_000:
            raise TranslationError("单张图片不能超过 10 MB。")
        with Image.open(BytesIO(binary)) as image:
            mime = Image.MIME.get(image.format or "")
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"}:
                raise TranslationError("不支持此图片格式，请发送 PNG、JPEG、WebP、GIF 或 BMP。")
            image.verify()
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise TranslationError("图片数据无效，请重新发送图片。") from exc
    return f"data:{mime};base64,{encoded}"


def extract(message: Incoming, *, command: bool = False, trigger: str = '/rs 翻译') -> Input:
    text = "".join(s.data for s in message.raw_message if s.type == "text" and isinstance(s.data, str))
    if not message.raw_message:
        text = message.processed_plain_text
    if command:
        # 此模块也用于独立解析探针，保持按文件导入兼容。
        prefix = r'^\s*' + r'\s+'.join(re.escape(part) for part in trigger.split()) + r'(?=\s|\[|$)\s*'
        text = re.sub(prefix, '', text, count=1)
    images = [s for s in message.raw_message if s.type in {"image", "emoji"}]
    if len(images) > 4:
        raise TranslationError("每次最多翻译 4 张图片。")
    if any(not s.binary_data_base64 for s in images):
        raise TranslationError(f"无法读取图片原文件，请重新发送图片并附上 {trigger}。")
    if len(text) > 12000:
        raise TranslationError("每次最多翻译 12000 个字符，请分段发送。")
    return Input(text=text.strip(), images=tuple(image_url(s.binary_data_base64) for s in images))


def build_payload(source: Input, model: str) -> Dict[str, JsonValue]:
    content: JsonValue
    if source.images:
        prompt = "请将图片中的文本翻译为中文，直接输出翻译结果，不要进行任何解释。"
        if source.text:
            prompt += "\n同时翻译以下文本：\n" + source.text
        parts: List[JsonValue] = [{"type": "text", "text": prompt}]
        parts.extend({"type": "image_url", "image_url": {"url": url}} for url in source.images)
        content = parts
    else:
        content = "请将以下文本翻译为中文，直接输出翻译结果，不要进行任何解释。\n\n" + source.text
    return {
        "model": model, "messages": [{"role": "user", "content": content}],
        "temperature": 0.0, "max_tokens": 8192, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class ReplyMessage(BaseModel):
    content: str


class Choice(BaseModel):
    message: ReplyMessage
    finish_reason: Optional[str] = None


class Completion(BaseModel):
    choices: List[Choice] = Field(min_length=1)


def parse_response(data: JsonValue) -> str:
    try:
        reply = Completion.model_validate(data).choices[0]
    except ValidationError as exc:
        raise TranslationError("翻译服务返回了无效结果，请稍后重试。") from exc
    if reply.finish_reason == "length":
        raise TranslationError("译文超出服务输出上限，已被截断，请缩短文字或减少图片后重试。")
    text = reply.message.content.split("</think>")[-1].strip()
    if not text or text.startswith("<think>"):
        raise TranslationError("翻译服务未返回译文，请稍后重试。")
    return text

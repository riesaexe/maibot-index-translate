"""把带编号的原图文字区域交给视觉翻译，不把 OCR 猜测当成原文。"""

from io import BytesIO
from typing import Dict, List, Tuple

import base64
import json

from PIL import Image, ImageDraw
from pydantic import JsonValue


def crop_image(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    crop = image.crop(box)
    scale = min(4.0, max(1.0, 128 / crop.height), 1600 / crop.width, 1600 / crop.height)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)


def visual_payload(image: Image.Image, boxes: Tuple[Tuple[int, int, int, int], ...], model: str, prepared: Tuple[Image.Image, ...] = ()) -> Dict[str, JsonValue]:
    crops: List[Image.Image] = []
    for index, box in enumerate(boxes):
        crop = prepared[index] if prepared else image.crop(box)
        crops.append(crop_image(crop, (0, 0, crop.width, crop.height)))
    width = max(crop.width for crop in crops) + 70
    sheet = Image.new('RGB', (width, sum(c.height + 45 for c in crops) + 10), 'white')
    draw = ImageDraw.Draw(sheet)
    y = 10
    for index, crop in enumerate(crops):
        draw.text((8, y + 2), str(index), fill='black', font_size=24)
        sheet.paste(crop, (55, y))
        y += crop.height + 45
    keys = json.dumps({str(i): '<该区域的译文>' for i in range(len(crops))}, ensure_ascii=False)
    prompt = (
        '请直接看图，把每个编号区域的文字翻译为简体中文。手写字、草书、空心字、艺术字'
        '需要根据图像辨认，不要猜测或留下错误的英文残片。每个区域是一段完整内容，须结合上下文连贯翻译，不要逐词逐行硬译。'
        '每个编号仅对应紧随其后的图片，编号本身不翻译。'
        '保留数字、代码与专名，已经是中文的内容保持原样；无需翻译的区域必须返回原文，不能返回空字符串。'
        '仓库名、账号、路径应直接抄写，不要因为不认识专名而标记为无法辨认。'
        '只返回 JSON 对象，key 为区域编号，value 为对应译文。不要 Markdown 或解释。'
        '无法辨认的区域写为“[无法辨认]”，不要编造。JSON 结构：\n\n' + keys
    )
    return request_body(sheet, prompt, model)


def single_payload(image: Image.Image, box: Tuple[int, int, int, int], model: str) -> Dict[str, JsonValue]:
    prompt = (
        '请直接看图，把图片中的文字翻译为简体中文。仔细辨认手写字、草书和艺术字。'
        '以图片正中的完整文字为准，忽略边缘被裁断的相邻文字。普通英文单词和句子必须翻译为中文，不要当作专名保留。'
        '数字、代码、专名和已经是中文的内容保持原样。仓库名、账号、路径直接抄写。'
        '请结合整段上下文连贯翻译，修复断行，不要逐词硬译。只输出译文，不要解释、编号或额外格式。'
        '无法辨认时只输出[无法辨认]，不要编造。'
    )
    return request_body(crop_image(image, box), prompt, model)


def request_body(image: Image.Image, prompt: str, model: str) -> Dict[str, JsonValue]:
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    return {
        'model': model,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': url}},
        ]}],
        'temperature': 0.0, 'max_tokens': 2048, 'stream': False,
        'chat_template_kwargs': {'enable_thinking': False},
    }



def comic_payload(image: Image.Image, box: Tuple[int, int, int, int], model: str, japanese: bool) -> Dict[str, JsonValue]:
    direction = ('这是日文漫画。竖排按从上到下、列从右到左阅读。汉字旁边的小假名是注音，不要重复翻译。'
                 if japanese else '这是英文漫画。将整个区域的台词连贯翻译成口语自然的中文，保留段落分隔。')
    prompt = (direction + '只翻译图片中的文字，不要描述人物、表情或画面。'
              '不要添加原文没有的年份、问候语、情节和解释。所有台词都翻译为简体中文，专名保持一致。'
              '已经是中文的内容原样返回。省略号、感叹号和表情符号保留。'
              '只输出译文，无法可靠辨认时只输出[无法辨认]。')
    return request_body(crop_image(image, box), prompt, model)

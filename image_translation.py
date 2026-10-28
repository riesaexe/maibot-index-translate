"""本地定位图片文字，将 Index-Translate 译文排回原位置。"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

import base64
import os
import re

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import TypeAdapter, ValidationError
from rapidocr_onnxruntime import RapidOCR

from .translation import TranslationError
from .image_layout import clean_region, ink_height, clean_paper, foreground_protection, restore_foreground
from .image_geometry import Quad, composite, paragraphs, rectify
from .image_detection import locate, textured_background
from .image_request import visual_payload
from .comic_detector import ComicDetection, detect_comic
from .comic_render import draw_comic
from .comic_ocr import japanese_text
from .image_style import chalk_alpha, inspect_style, restore_chalk, split_mixed_line, style_font


@dataclass(frozen=True, slots=True)
class Region:
    box: Tuple[int, int, int, int]
    text: str
    quad: Quad = ()
    line_height: int = 0
    paper: bool = False
    comic: bool = False
    japanese: bool = False


@dataclass(frozen=True, slots=True)
class Page:
    image: Image.Image
    regions: Tuple[Region, ...]
    comic: ComicDetection | None = None


_lock = Lock()
_engine: RapidOCR | None = None
_lines = TypeAdapter(List[Tuple[List[Tuple[float, float]], str, float]])
_translations = TypeAdapter(Dict[str, str])
UNREADABLE = '[无法辨认]'


class TranslationMappingError(TranslationError):
    """视觉批次响应无法按区域一一对应。"""


def detect(url: str, comic_mode: str = "auto") -> Page:
    binary = base64.b64decode(url.split(',', 1)[1], validate=True)
    with Image.open(BytesIO(binary)) as source:
        if source.width * source.height > 20_000_000:
            raise TranslationError("图片像素过大，请缩小到 2000 万像素以内。")
        image = ImageOps.exif_transpose(source).convert('RGB')
    try:
        comic = detect_comic(image, comic_mode)
    except (ValueError, OSError) as exc:
        raise TranslationError("漫画检测模型不可用，请检查模型文件或关闭 comic_detector。") from exc
    if comic is not None:
        regions = []
        for block in comic.blocks:
            a, b, c, d = block.box
            crop = image.crop((max(0, a - 8), max(0, b - 8), min(image.width, c + 8), min(image.height, d + 8)))
            text = japanese_text(crop) if block.japanese else ''
            regions.append(Region(block.box, text, line_height=block.font_size, comic=True, japanese=block.japanese))
        return Page(image, tuple(regions), comic)
    textured = textured_background(image)
    # 小字号截图先放大，提高文字定位与最终中文的可读性。
    if image.width < 1600 and image.width * image.height < 5_000_000:
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    global _engine
    with _lock:
        if _engine is None:
            _engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        try:
            lines = locate(_engine, image)
        except ValidationError as exc:
            raise TranslationError("图片文字定位结果异常，请换一张图片重试。") from exc
    if not lines:
        raise TranslationError("图片中未识别到可翻译的文字，请发送更清晰的原图。")
    regions = []
    located = []
    for points, text in lines:
        located.append((tuple(points), text))
        left = max(0, int(min(p[0] for p in points)) - 2)
        top = max(0, int(min(p[1] for p in points)) - 2)
        right = min(image.width, int(max(p[0] for p in points)) + 3)
        bottom = min(image.height, int(max(p[1] for p in points)) + 3)
        if right > left and bottom > top:
            regions.extend(Region(box, value) for box, value in split_mixed_line(image, (left, top, right, bottom), text))
    if not regions:
        raise TranslationError("图片中未识别到可靠文字，请发送更清晰的原图。")
    if len(regions) > 180 or sum(len(r.text) for r in regions) > 10000:
        raise TranslationError("图片文字过多，请裁剪后分批翻译。")
    # 大字号实拍手写页按段处理；小字号界面与已稳定的粉笔分支保留原逻辑。
    import numpy as np
    heights = [float(np.linalg.norm(np.array(points[3]) - points[0])) for points, _ in located]
    paragraph_size = bool(heights) and (float(np.median(heights)) >= 42 or
                     (textured and float(np.median(heights)) >= max(20, image.width * .025)))
    if paragraph_size and not any(inspect_style(image, r.box, r.text).chalk for r in regions):
        regions = []
        for quad, text, line_height in paragraphs(located, image.size):
            box = (max(0, int(min(p[0] for p in quad))), max(0, int(min(p[1] for p in quad))),
                   min(image.width, int(max(p[0] for p in quad)) + 1), min(image.height, int(max(p[1] for p in quad)) + 1))
            levels = np.array(rectify(image, quad)).astype(float).mean(axis=2)
            low, middle, high = np.quantile(levels, [.05, .5, .95])
            paper = middle - low > high - middle
            regions.append(Region(box, text, quad, line_height, paper))
    return Page(image, tuple(regions))


def payload(page: Page, model: str):
    return visual_payload(page.image, tuple(region.box for region in page.regions), model,
                          tuple(rectify(page.image, region.quad) if region.quad else page.image.crop(region.box) for region in page.regions))


def batches(page: Page) -> Tuple[Page, ...]:
    parts = []
    current: List[Region] = []
    length = 0
    for region in page.regions:
        if current and (len(current) >= 4 or length + len(region.text) > 1000):
            parts.append(Page(page.image, tuple(current)))
            current = []
            length = 0
        current.append(region)
        length += len(region.text)
    if current:
        parts.append(Page(page.image, tuple(current)))
    return tuple(parts)


def parse_translations(text: str, count: int) -> Tuple[str, ...]:
    cleaned = text.strip()
    if cleaned.startswith('```') and cleaned.endswith('```'):
        cleaned = cleaned.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        values = _translations.validate_json(cleaned, strict=True)
    except ValidationError as exc:
        raise TranslationMappingError("翻译服务未返回可排版的译文，请稍后重试。") from exc
    if set(values) != {str(i) for i in range(count)} or any(not v.strip() for v in values.values()):
        raise TranslationMappingError("译文与图片文字区域无法对应，请稍后重试。")
    if any(UNREADABLE in value for value in values.values()):
        raise TranslationMappingError("图片中有无法可靠辨认的文字，需要逐区核对。")
    return tuple(values[str(i)].strip() for i in range(count))


def find_font(configured: str) -> str:
    if configured:
        if Path(configured).is_file():
            return configured
        raise TranslationError("配置的中文字体文件不存在，请检查 font_path。")
    candidates = [
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/msyh.ttc',
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise TranslationError("未找到中文字体，请在插件配置 font_path 中指定中文字体文件。")


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> List[str]:
    lines = []
    for paragraph in text.splitlines():
        line = ''
        tokens = re.findall(r"[A-Za-z0-9]+(?:[.'’_-][A-Za-z0-9]+)*|\s+|[^A-Za-z0-9\s]", paragraph)
        for token in tokens:
            if token.isspace():
                if line:
                    line += ' '
                continue
            units = [token] if font.getlength(token) <= width else list(token)
            for unit in units:
                if line and font.getlength(line + unit) > width:
                    if (unit in '，。！？、；：）》」』】”’' or line.rstrip()[-1] in '（《「『【“‘') and len(line.rstrip()) > 1:
                        trimmed = line.rstrip()
                        cut = len(trimmed) - 1
                        while cut > 0 and trimmed[cut] in '，。！？、；：）》」』】”’':
                            cut -= 1
                        if cut:
                            lines.append(trimmed[:cut].rstrip())
                            line = trimmed[cut:]
                    else:
                        lines.append(line.rstrip())
                        line = ''
                line += unit
        lines.append(line.rstrip())
    return lines


def unchanged(source: str, translated: str) -> bool:
    if source == translated or translated == UNREADABLE:
        return True
    # 无中文译文时保留原字形，不用识别错误的 OCR 文本决定是否重绘拉丁原文。
    return not bool(re.search(r'[㐀-鿿]', translated))


def render(page: Page, translations: Tuple[str, ...], font_path: str) -> str:
    if len(translations) != len(page.regions):
        raise TranslationError("译文与图片区域数量不一致。")
    if page.comic is not None:
        return draw_comic(page.image, page.comic, translations, find_font(font_path))
    if any(region.quad for region in page.regions):
        canvas = page.image.copy()
        for region, text in zip(page.regions, translations, strict=True):
            if unchanged(region.text, text):
                continue
            if region.quad:
                patch = rectify(page.image, region.quad)
                local = Region((0, 0, patch.width, patch.height), region.text,
                               line_height=region.line_height, paper=region.paper)
                painted = Image.open(BytesIO(base64.b64decode(render(Page(patch, (local,)), (text,), font_path))))
                composite(canvas, painted, region.quad)
            else:
                painted = Image.open(BytesIO(base64.b64decode(render(Page(page.image, (region,)), (text,), font_path))))
                canvas.paste(painted.crop(region.box), region.box[:2])
        restore_foreground(page.image, canvas)
        buffer = BytesIO()
        canvas.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('ascii')
    image = page.image.copy()
    font_file = find_font(font_path)
    colors = []
    styles = [inspect_style(page.image, region.box, region.text) for region in page.regions]
    # 每个框都从未修改的原图取样，全部擦除后再排字，避免采到刚绘制的译文。
    for region, text, style in zip(page.regions, translations, styles, strict=True):
        if unchanged(region.text, text):
            colors.append((32, 32, 32))
            continue
        left, top, right, bottom = region.box
        cleanup_box = (max(0, left - 3), max(0, top - 2),
                       min(image.width, right + 3), min(image.height, bottom + 2))
        if region.paper:
            cleaned, color = clean_paper(page.image, cleanup_box)
        else:
            cleaned, color = restore_chalk(page.image, cleanup_box) if style.chalk else clean_region(page.image, cleanup_box)
        image.paste(cleaned.crop(cleanup_box), cleanup_box[:2])
        colors.append(color)
    for region, text in zip(page.regions, translations, strict=True):
        if unchanged(region.text, text):
            image.paste(page.image.crop(region.box), region.box[:2])
    for region, text, color, style in zip(page.regions, translations, colors, styles, strict=True):
        if unchanged(region.text, text):
            continue
        if region.paper:
            text = '\n\n'.join(' '.join(block.split()) for block in re.split(r'\n\s*\n', text.strip()))
        left, top, right, bottom = region.box
        width, height = right - left, bottom - top
        if region.paper:
            import numpy as np
            occupied = np.flatnonzero(foreground_protection(page.image.crop(region.box)).mean(axis=0) > .05)
            if len(occupied) and occupied[0] > width * .55:
                width = max(8, int(occupied[0]) - 4)
        selected_font = style_font(style, font_file, font_path)
        if region.paper and not font_path:
            name = 'simhei.ttf' if region.text.isupper() else 'simkai.ttf'
            handwriting = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / name
            if handwriting.is_file():
                selected_font = str(handwriting)
        font = ImageFont.truetype(selected_font, 8)
        lines = wrap(text, font, width)
        maximum_size = min(region.line_height or ink_height(page.image, region.box) + 2, int(height * 0.82), 256 if style.chalk or region.paper else 64)
        if region.paper and region.text.isupper() and len(text) <= 12:
            maximum_size = min(maximum_size, max(8, width // 4))
        for size in range(maximum_size, 7, -1):
            candidate = ImageFont.truetype(selected_font, size)
            wrapped = wrap(text, candidate, width)
            if len(wrapped) * (size + 1) <= height and all(candidate.getlength(line) <= width for line in wrapped):
                font, lines = candidate, wrapped
                break
        if len(lines) * (font.size + 1) > height or any(font.getlength(line) > width for line in lines):
            raise TranslationError("图片文字过密，中文无法完整排入原位置，请裁剪或放大后重试。")
        # 二倍分辨率绘制后缩小，统一细小文字的边缘；按实际字形高度垂直居中。
        overlay = Image.new('RGBA', (width * 2, height * 2))
        draw = ImageDraw.Draw(overlay)
        smooth_font = ImageFont.truetype(selected_font, font.size * 2)
        line_height = font.size + 1
        glyph_height = max(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines)
        total_height = glyph_height + (len(lines) - 1) * line_height
        y = max(0, min(height - total_height, region.line_height // 6)) if region.paper else max(0, (height - total_height) // 2)
        for line in lines:
            x = max(0, (width * 2 - smooth_font.getlength(line)) / 2) if style.chalk else 0
            draw.text((x, y * 2), line, font=smooth_font, fill=(*color, 255), anchor='lt')
            y += line_height
        if style.chalk:
            overlay = chalk_alpha(overlay, style)
        overlay = overlay.resize((width, height), Image.Resampling.LANCZOS)
        image.paste(overlay, (left, top), overlay)
    buffer = BytesIO()
    restore_foreground(page.image, image)
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')

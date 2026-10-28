"""漫画专用擦字与排版：只改文字掩膜，竖排字形保持直立、列从右到左。"""
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
import base64
import math
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .comic_detector import ComicDetection
from .comic_layout import balloons
from .image_layout import clean_region


def draw_comic(image: Image.Image, detected: ComicDetection, translations: Tuple[str, ...], font_path: str) -> str:
    canvas = image.copy()
    pixels = np.array(image)
    source_mask = np.array(detected.mask)
    bubbles = balloons(image, detected.blocks)
    details = []
    for index, (block, text) in enumerate(zip(detected.blocks, translations, strict=True)):
        # 原文/省略号被原样返回时保留笔迹；不能把重复假名当成独立译文。
        if '[无法辨认]' in text or not re.search(r'[\u3400-\u9fff]', text):
            details.append(None)
            continue
        left, top, right, bottom = block.box
        padding = max(3, round(block.font_size * .18))
        box = (max(0, left - padding), max(0, top - padding), min(image.width, right + padding), min(image.height, bottom + padding))
        a, b, c, d = box
        patch_mask = (source_mask[b:d, a:c] > 55).astype(np.uint8) * 255
        kernel = max(3, min(25, round(block.font_size * .16))) | 1
        patch_mask = cv2.dilate(patch_mask, np.ones((kernel, kernel), np.uint8))
        if index in bubbles:
            patch_mask &= bubbles[index][1][b:d, a:c]
        background, _ = clean_region(image, box)
        original = pixels[b:d, a:c]
        replacement = np.array(background.crop(box))
        selected = patch_mask > 0
        restored = original.copy()
        restored[selected] = replacement[selected]
        canvas.paste(Image.fromarray(restored), (a, b))
        bg = np.median(replacement.reshape(-1, 3), axis=0)
        ink = original[selected].astype(float)
        red = ink[(ink[:, 0] > ink[:, 1] * 1.6) & (ink[:, 0] > ink[:, 2] * 1.6) & (ink[:, 0] > 45)]
        color = tuple(int(v) for v in np.median(red, axis=0)) if len(red) > 10 else ((20, 20, 20) if bg.mean() > 150 else (245, 245, 245))
        outline = bool(bg.mean() < 140 and len(red) > 10 and np.any(ink.mean(axis=1) > 215))
        details.append((bubbles[index][0] if index in bubbles else block.box, color, outline))
    for block, text, detail in zip(detected.blocks, translations, details, strict=True):
        if detail is None:
            continue
        box, color, outline = detail
        left, top, right, bottom = box
        width, height = right - left, bottom - top
        vertical = block.japanese and (block.box[3] - block.box[1]) > (block.box[2] - block.box[0]) * .9
        overlay = lettering(text, (width, height), font_path, block.font_size, vertical, color, outline)
        canvas.paste(overlay, (left, top), overlay)
    buffer = BytesIO()
    canvas.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def horizontal_lines(text: str, font: ImageFont.FreeTypeFont, width: int) -> List[str]:
    output = []
    for paragraph in text.splitlines():
        line = ''
        for token in re.findall(r'[A-Za-z0-9]+|[^A-Za-z0-9]', paragraph):
            if line and font.getlength(line + token) > width and token not in '，。！？：；、）】」』':
                output.append(line.strip())
                line = ''
            line += token
        if line.strip():
            output.append(line.strip())
    return output or ['']


def lettering(text: str, size: Tuple[int, int], font_path: str, maximum: int, vertical: bool,
              color: Tuple[int, int, int], outline: bool) -> Image.Image:
    width, height = size
    text = text.strip()
    # 日文注音不进入译文；中文竖排保留直立字符，标点使用竖排字形。
    content = re.sub(r'\s+', '', text) if vertical else text
    for font_size in range(max(8, min(maximum, height)), 7, -1):
        font = ImageFont.truetype(font_path, font_size * 2)
        stroke = max(1, round(font_size * .055)) if outline else 0
        step = font_size * 1.15
        if vertical:
            rows = max(1, int((height - stroke * 2) / step))
            columns = math.ceil(len(content) / rows)
            fits = columns * step + stroke * 2 <= width
        else:
            lines = horizontal_lines(content, font, (width - stroke * 2) * 2)
            fits = (len(lines) * step + stroke * 2 <= height and
                    max(font.getlength(line) for line in lines) + stroke * 4 <= width * 2)
        if fits:
            break
    else:
        raise ValueError('漫画译文过长，无法安全排入文字区域')
    overlay = Image.new('RGBA', (width * 2, height * 2))
    draw = ImageDraw.Draw(overlay)
    if vertical:
        columns = math.ceil(len(content) / rows)
        x0 = (width + columns * step) / 2 - step
        symbol_path = Path('C:/Windows/Fonts/seguisym.ttf')
        symbol_font = ImageFont.truetype(str(symbol_path), font_size * 2) if symbol_path.exists() else font
        for column in range(columns):
            part = content[column * rows:(column + 1) * rows]
            y0 = (height - len(part) * step) / 2
            for row, character in enumerate(part):
                glyph_font = symbol_font if character in '♡♥' else font
                x = (x0 - column * step) * 2 + (font_size * 2 - glyph_font.getlength(character)) / 2
                draw.text((x, (y0 + row * step) * 2), character, font=glyph_font, fill=(*color, 255), anchor='lt')
    else:
        y = (height - len(lines) * step) / 2
        for line in lines:
            x = (width * 2 - font.getlength(line)) / 2
            draw.text((x, y * 2), line, font=font, fill=(*color, 255), anchor='lt',
                      stroke_width=stroke * 2, stroke_fill='white' if outline else None)
            y += step
    return overlay.resize((width, height), Image.Resampling.LANCZOS)

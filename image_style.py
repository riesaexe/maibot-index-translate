"""对大字、粉笔及手写字做可解释的风格近似；普通界面截图保持常规排字。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import os

import cv2
import numpy as np

from PIL import Image, ImageFilter


@dataclass(frozen=True, slots=True)
class Style:
    chalk: bool = False
    outline: bool = False
    block: bool = False


def inspect_style(image: Image.Image, box: Tuple[int, int, int, int], text: str) -> Style:
    patch = np.array(image.crop(box).convert('RGB'))
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    background = float(np.median(gray))
    foreground = gray[gray > background + 65]
    # 大幅深色、有明显粉化明度差异的白色笔画；界面小字不走此分支。
    chalk = (box[3] - box[1] >= 70 and background < 105 and len(foreground) > 100
             and float(foreground.mean()) > 155 and float(foreground.std()) > 18)
    if not chalk:
        return Style()
    mask = (gray > max(110, background + 65)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    large_holes = 0
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            parent = hierarchy[0, index, 3]
            if parent >= 0 and cv2.contourArea(contours[parent]) > 500:
                if cv2.contourArea(contour) / cv2.contourArea(contours[parent]) > 0.45:
                    large_holes += 1
    return Style(chalk=True, outline=large_holes >= 2, block=text.isupper())


def style_font(style: Style, default: str, configured: str) -> str:
    if configured or not style.chalk:
        return default
    name = 'simhei.ttf' if style.outline or style.block else 'simkai.ttf'
    candidate = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / name
    return str(candidate) if candidate.is_file() else default


def split_mixed_line(image: Image.Image, box: Tuple[int, int, int, int], text: str):
    words = text.split()
    if len(words) != 2 or not words[0].isupper() or words[1].isupper() or not inspect_style(image, box, text).chalk:
        return ((box, text),)
    patch = np.array(image.crop(box).convert('RGB'))
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    ink = gray > max(120, float(np.median(gray)) + 65)
    occupied = ink.sum(axis=0) > max(3, patch.shape[0] * 0.015)
    gaps = []
    start = 0
    for index, value in enumerate(occupied):
        if value:
            if index - start >= max(15, patch.shape[0] * 0.1) and 0.25 < (start + index) / (2 * len(occupied)) < 0.8:
                gaps.append((start + index) // 2)
            start = index + 1
    if not gaps:
        return ((box, text),)
    target = len(occupied) * len(words[0]) / sum(len(word) for word in words)
    cut = min(gaps, key=lambda position: abs(position - target))
    left, top, right, bottom = box
    return (((left, top, left + cut, bottom), words[0]), ((left + cut, top, right, bottom), words[1]))


def chalk_alpha(overlay: Image.Image, style: Style) -> Image.Image:
    alpha = overlay.getchannel('A')
    if style.outline:
        size = max(3, min(11, overlay.height // 45)) | 1
        inner = np.array(alpha.filter(ImageFilter.MinFilter(size))).astype(np.int16)
        alpha = Image.fromarray(np.clip(np.array(alpha).astype(np.int16) - inner, 0, 255).astype(np.uint8))
    # 固定随机种子保证同一排版可复现；轻微掉粉而非整片半透明。
    values = np.array(alpha).astype(float)
    noise = np.random.default_rng(17).uniform(0.68, 1.0, values.shape)
    holes = np.random.default_rng(23).random(values.shape) < 0.035
    values *= noise
    values[holes] *= 0.25
    overlay.putalpha(Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)))
    return overlay


def restore_chalk(image: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int, int]]:
    left, top, right, bottom = box
    margin = 18
    outer = (max(0, left - margin), max(0, top - margin),
             min(image.width, right + margin), min(image.height, bottom + margin))
    patch = np.array(image.crop(outer).convert('RGB'))
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    level = float(np.median(gray))
    # 白色粉笔的通道差较小，排除肤色、红指甲等非文字前景。
    spread = patch.max(axis=2).astype(int) - patch.min(axis=2).astype(int)
    mask = ((gray > max(100, level + 45)) & (spread < 65)).astype(np.uint8) * 255
    allowed = np.zeros_like(mask)
    x, y = left - outer[0], top - outer[1]
    allowed[y:y + bottom - top, x:x + right - left] = 255
    mask &= allowed
    mask = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8)) & allowed
    red = patch[:, :, 0].astype(float)
    skin = (red > patch[:, :, 1] * 1.12) & (red > patch[:, :, 2] * 1.2) & (red > 60)
    mask[skin] = 0
    pixels = patch[(gray > max(170, level + 100)) & (allowed > 0)]
    if len(pixels):
        brightness = pixels.mean(axis=1)
        pixels = pixels[brightness >= np.quantile(brightness, 0.6)]
        color = tuple(int(v) for v in np.median(pixels, axis=0))
    else:
        color = (225, 228, 230)
    restored = cv2.inpaint(patch, mask, 5, cv2.INPAINT_NS)
    result = image.copy()
    result.paste(Image.fromarray(restored).crop((x, y, x + right - left, y + bottom - top)), (left, top))
    return result, color

"""定位与识别分离：保留低识别置信度的文字框，补充混合方向检测。"""
from typing import List, Tuple
import re

import numpy as np
from PIL import Image
from pydantic import TypeAdapter
from rapidocr_onnxruntime import RapidOCR

from .image_geometry import Quad

_lines = TypeAdapter(List[Tuple[List[Tuple[float, float]], str, float]])


def chinese_only(text: str) -> bool:
    return len(re.findall(r'[\u3400-\u9fff]', text)) >= 2 and not re.search(r'[\u3040-\u30ff]|[A-Za-z]{2}', text)


def locate(engine: RapidOCR, image: Image.Image) -> List[Tuple[Quad, str]]:
    result, _ = engine(np.array(image), text_score=0, box_thresh=.35)
    if not result:
        return []
    rows = _lines.validate_python(result)
    candidates = [(tuple(points), reading_text(points, text, score)) for points, text, score in rows
                  if len(points) == 4 and np.linalg.norm(np.array(points[1]) - points[0]) >= 8]
    tall = [points for points, _ in candidates
            if np.linalg.norm(np.array(points[3]) - points[0]) > np.linalg.norm(np.array(points[1]) - points[0]) * 1.6]
    if len(tall) < 3:
        return candidates
    # 对竖向排布的横写文字补一次 90 度检测，检测框按阅读方向映回原图。
    rotated = image.transpose(Image.Transpose.ROTATE_90)
    result, _ = engine(np.array(rotated), text_score=0, box_thresh=.25)
    if not result:
        return candidates
    additions = []
    bounds = np.array(tall).reshape(-1, 2)
    margin = float(np.median([np.linalg.norm(np.array(points[1]) - points[0]) for points in tall])) * 3
    minimum, maximum = bounds.min(axis=0) - [margin, margin / 3], bounds.max(axis=0) + [margin, margin / 3]
    for points, text, score in _lines.validate_python(result):
        if len(points) != 4 or np.linalg.norm(np.array(points[1]) - points[0]) < np.linalg.norm(np.array(points[3]) - points[0]) * 1.6:
            continue
        quad = tuple((float(image.width - 1 - y), float(x)) for x, y in points)
        # 新框必须与原有竖向检测区相邻，不能把另一方向的正文重复加入。
        center = np.array(quad).mean(axis=0)
        if np.all(center >= minimum) and np.all(center <= maximum):
            additions.append((quad, reading_text(points, text, score)))
    if not additions:
        return candidates
    candidates = [(points, text) for points, text in candidates
                  if not any(overlap(points, added) > .3 for added, _ in additions)]
    return candidates + additions


def overlap(first: Quad, second: Quad) -> float:
    a, b = np.array(first), np.array(second)
    left, top = np.maximum(a.min(axis=0), b.min(axis=0))
    right, bottom = np.minimum(a.max(axis=0), b.max(axis=0))
    intersection = max(0, right - left) * max(0, bottom - top)
    area = min(np.prod(a.max(axis=0) - a.min(axis=0)), np.prod(b.max(axis=0) - b.min(axis=0)))
    return float(intersection / max(1, area))


def textured_background(image: Image.Image) -> bool:
    """只为有实拍纹理的小手写字启用段落模式，界面小字保持原排版。"""
    import cv2
    gray = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    background = gray >= np.quantile(gray, .55)
    noise = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    return float(np.median(noise[background])) > 1.1


def reading_text(points: List[Tuple[float, float]], text: str, score: float) -> str:
    width = np.linalg.norm(np.array(points[1]) - points[0])
    height = np.linalg.norm(np.array(points[3]) - points[0])
    # 孤立的低置信度涂画不交给模型猜词；长行仍保留，供段落视觉识别补全。
    if score < .2 and width < height * 3:
        return '[无法辨认]'
    return text.strip() or '[文字区域]'

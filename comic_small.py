"""用高置信度 OCR 补充漫画的小字号旁白，排除已有文字块与切边残词。"""
import re
from functools import cache
from threading import Lock

import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .comic_detector import Box

_lock = Lock()


@cache
def ocr_engine() -> RapidOCR:
    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1, max_side_len=5000)


def search_windows(pixels: np.ndarray, existing: list[Box]) -> list[Box]:
    """一次整页扫描找到小字种子，再对种子附近补读；避免全页重复切块。"""
    height, width = pixels.shape[:2]
    rows, _ = ocr_engine()(pixels, text_score=.8, box_thresh=.35)
    windows: list[Box] = []
    for points, text, confidence in rows or []:
        p = np.array(points)
        a, b = p.min(axis=0).astype(int)
        c, d = p.max(axis=0).astype(int) + 1
        area = max(1, (c - a) * (d - b))
        if confidence < .8 or len(re.sub('[^A-Za-z]', '', text)) < 3:
            continue
        if any(max(0, min(c, old[2]) - max(a, old[0])) * max(0, min(d, old[3]) - max(b, old[1])) > area * .3 for old in existing):
            continue
        if min(c - a, d - b) > max(width, height) * .035:
            continue
        padding = max(c - a, d - b) * 3
        windows.append((max(0, int(a - padding)), max(0, int(b - padding)), min(width, int(c + padding)), min(height, int(d + padding))))
    return merge_neighbors(windows)


def small_captions(image: Image.Image, existing: list[Box], mask: np.ndarray) -> list[Box]:
    with _lock:
        return scan_captions(image, existing, mask)


def scan_captions(image: Image.Image, existing: list[Box], mask: np.ndarray) -> list[Box]:
    engine = ocr_engine()
    found: list[Box] = []
    pixels = np.array(image)
    for x, y, right, bottom in search_windows(pixels, existing):
            tile = pixels[y:bottom, x:right]
            rows, _ = engine(tile, text_score=.8, box_thresh=.35)
            for points, text, confidence in rows or []:
                p = np.array(points)
                if confidence < .8 or len(re.sub('[^A-Za-z]', '', text)) < 3:
                    continue
                if p.min() < 3 or p[:, 0].max() > tile.shape[1] - 4 or p[:, 1].max() > tile.shape[0] - 4:
                    continue
                p += [x, y]
                a, b = p.min(axis=0).astype(int)
                c, d = p.max(axis=0).astype(int) + 1
                box = (int(a), int(b), int(c), int(d))
                area = max(1, (c - a) * (d - b))
                if any(max(0, min(c, old[2]) - max(a, old[0])) * max(0, min(d, old[3]) - max(b, old[1])) > area * .3 for old in existing):
                    continue
                if min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0])) > max(image.size) * .035:
                    continue
                # 补框必须有可分离的浅色背景，且仅将 OCR 四边形内的深色笔画加入掩膜。
                patch = pixels[b:d, a:c]
                if np.median(patch) < 210:
                    continue
                polygon = np.zeros((d - b, c - a), np.uint8)
                cv2.fillConvexPoly(polygon, (p - [a, b]).astype(np.int32), 1)
                ink = (patch.mean(axis=2) < 170) & (polygon > 0)
                mask[b:d, a:c] = np.maximum(mask[b:d, a:c], ink.astype(float))
                found.append(box)
    return merge_neighbors(found)


def merge_neighbors(boxes: list[Box]) -> list[Box]:
    merged: list[Box] = []
    for box in sorted(boxes):
        a, b, c, d = box
        for index, (e, f, g, h) in enumerate(merged):
            gap = max(a - g, e - c, 0)
            if gap < max(d - b, h - f) * .8 and min(d, h) > max(b, f):
                merged[index] = (min(a, e), min(b, f), max(c, g), max(d, h))
                break
        else:
            merged.append(box)
    return merged

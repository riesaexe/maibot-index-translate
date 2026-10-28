"""Comic Text Detector ONNX 接口：文字块、语言类别与像素掩膜分开使用。"""
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Tuple
import hashlib
import os

import cv2
import httpx
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapidocr_onnxruntime.ch_ppocr_det.utils import DBPostProcess

MODEL_URL = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx'
MODEL_SHA256 = '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'
MODEL_LICENSE = 'GPL-3.0; https://github.com/dmMaze/comic-text-detector'
_lock = Lock()
_session: ort.InferenceSession | None = None
Box = Tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ComicBlock:
    box: Box
    japanese: bool
    font_size: int


@dataclass(frozen=True, slots=True)
class ComicDetection:
    blocks: Tuple[ComicBlock, ...]
    mask: Image.Image


def model_session() -> ort.InferenceSession:
    global _session
    if _session is not None:
        return _session
    folder = Path(__file__).parent / '.models'
    folder.mkdir(exist_ok=True)
    path = folder / 'comictextdetector.pt.onnx'
    if not path.exists():
        temporary = path.with_suffix('.download')
        digest = hashlib.sha256()
        try:
            with httpx.stream('GET', MODEL_URL, follow_redirects=True, timeout=90) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for part in response.iter_bytes(1024 * 1024):
                        output.write(part)
                        digest.update(part)
            if digest.hexdigest() != MODEL_SHA256:
                raise ValueError('漫画检测模型校验失败')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('漫画检测模型文件损坏')
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    settings.inter_op_num_threads = 1
    settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    _session = ort.InferenceSession(str(path), sess_options=settings, providers=['CPUExecutionProvider'])
    return _session


def infer(image: Image.Image) -> Tuple[List[Tuple[Box, bool, float]], np.ndarray, List[np.ndarray]]:
    width, height = image.size
    scale = 1024 / max(width, height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    padded = np.full((1024, 1024, 3), 114, np.uint8)
    padded[:rh, :rw] = cv2.resize(np.array(image), (rw, rh))
    session = model_session()
    blocks, segmentation, detection = session.run(None, {'images': padded.astype(np.float32).transpose(2, 0, 1)[None] / 255})
    values = blocks[0]
    scores = values[:, 4] * values[:, 5:].max(axis=1)
    selected = values[scores > .4]
    scores = scores[scores > .4]
    boxes = np.column_stack((selected[:, 0] - selected[:, 2] / 2, selected[:, 1] - selected[:, 3] / 2,
                             selected[:, 2], selected[:, 3]))
    kept = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), .4, .35)
    result = []
    for index in np.asarray(kept).reshape(-1):
        x, y, w, h = boxes[index] / scale
        box = (max(0, int(x)), max(0, int(y)), min(width, int(x + w)), min(height, int(y + h)))
        if box[2] > box[0] and box[3] > box[1]:
            result.append((box, bool(selected[index, 5:].argmax() == 1), float(scores[index])))
    mask = cv2.resize(segmentation[0, 0, :rh, :rw], (width, height))
    line_boxes, _ = DBPostProcess(thresh=.3, box_thresh=.6, unclip_ratio=1.5)(detection[:, :1], (1024, 1024))
    lines = [points.astype(float) / scale for points in line_boxes]
    return result, mask, lines


def automatic(blocks: List[Tuple[Box, bool, float]], image: Image.Image) -> bool:
    vertical = sum(japanese and box[3] - box[1] > (box[2] - box[0]) * .9 for box, japanese, _ in blocks)
    if vertical >= 2:
        return True
    # 彩色、深底、浅描边漫画字幕；普通纸张、界面与粉笔样例不切换。
    for box, japanese, _ in blocks:
        patch = np.array(image.crop(box)).astype(float)
        background = np.median(patch.reshape(-1, 3), axis=0)
        gray = patch.mean(axis=2)
        if not japanese and len(blocks) >= 2 and background[0] > background[1] * 1.12 and background[0] > background[2] * 1.12:
            if np.quantile(gray, .8) - np.quantile(gray, .2) > 80:
                return True
    return False


def detect_comic(image: Image.Image, mode: str) -> ComicDetection | None:
    pixels = np.array(image)
    if mode == 'off' or (mode == 'auto' and (min(image.size) < 250 or float(pixels.std()) < 45)):
        return None
    with _lock:
        blocks, mask, lines = infer(image)
    if not blocks or (mode == 'auto' and not automatic(blocks, image)):
        return None
    with _lock:
        supplement_large(image, blocks, mask, lines)
    if max(image.size) > 2200 and not any(japanese for _, japanese, _ in blocks):
        from .comic_small import small_captions
        blocks.extend((box, False, .8) for box in small_captions(image, [box for box, _, _ in blocks], mask))
    # 模型行检测补全小字；只有高置信度文字行参与，不能拿脸部白色区域补框。
    for points in lines:
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        box = (max(0, int(left)), max(0, int(top)), min(image.width, int(right) + 1), min(image.height, int(bottom) + 1))
        area = max(1, (right - left) * (bottom - top))
        covered = any(max(0, min(right, b[2]) - max(left, b[0])) * max(0, min(bottom, b[3]) - max(top, b[1])) > area * .4 for b, _, _ in blocks)
        if not covered and min(right - left, bottom - top) >= 8:
            blocks.append((box, sum(jp for _, jp, _ in blocks) > len(blocks) / 2, .6))
    output = []
    for box, japanese, _ in blocks:
        left, top, right, bottom = box
        sizes = [min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0])) for p in lines
                 if left <= p[:, 0].mean() <= right and top <= p[:, 1].mean() <= bottom]
        font_size = round(float(np.quantile(sizes, .7))) if sizes else min(right - left, bottom - top)
        font_size = max(12, min(160, font_size))
        parts = [box] if japanese else split_captions(box, mask, font_size)
        output.extend(ComicBlock(part, japanese, font_size) for part in parts)
    return ComicDetection(tuple(output), Image.fromarray((mask.clip(0, 1) * 255).astype(np.uint8)))


def supplement_large(image: Image.Image, blocks: List[Tuple[Box, bool, float]], mask: np.ndarray,
                     lines: List[np.ndarray]) -> None:
    if max(image.size) <= 2200:
        return
    scale = 3072 / max(image.size)
    working = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    for y in sorted(set(range(0, max(1, working.height - 1024), 896)) | {max(0, working.height - 1024)}):
        for x in sorted(set(range(0, max(1, working.width - 1024), 896)) | {max(0, working.width - 1024)}):
            tile = working.crop((x, y, min(working.width, x + 1024), min(working.height, y + 1024)))
            found, tile_mask, tile_lines = infer(tile)
            for box, japanese, score in found:
                a, b, c, d = box
                mapped = (int((a + x) / scale), int((b + y) / scale), min(image.width, int((c + x) / scale)), min(image.height, int((d + y) / scale)))
                area = max(1, (mapped[2] - mapped[0]) * (mapped[3] - mapped[1]))
                covered = any(max(0, min(mapped[2], old[2]) - max(mapped[0], old[0])) * max(0, min(mapped[3], old[3]) - max(mapped[1], old[1])) > area * .4 for old, _, _ in blocks)
                if not covered and score >= .6:
                    blocks.append((mapped, japanese, score))
            lines.extend((points + [x, y]) / scale for points in tile_lines)
            left, top = round(x / scale), round(y / scale)
            right, bottom = min(image.width, round((x + tile.width) / scale)), min(image.height, round((y + tile.height) / scale))
            restored = cv2.resize(tile_mask, (right - left, bottom - top))
            mask[top:bottom, left:right] = np.maximum(mask[top:bottom, left:right], restored)


def split_captions(box: Box, mask: np.ndarray, font_size: int) -> List[Box]:
    left, top, right, bottom = box
    occupied = (mask[top:bottom, left:right] > .6).sum(axis=1) > max(2, (right - left) * .01)
    positions = np.flatnonzero(occupied)
    if not len(positions):
        return [box]
    gaps = np.flatnonzero(np.diff(positions) > max(8, font_size * .25))
    if not len(gaps):
        return [box]
    spans = np.split(positions, gaps + 1)
    return [(left, max(top, top + int(span[0]) - 3), right, min(bottom, top + int(span[-1]) + 4)) for span in spans if len(span) > font_size * .2]

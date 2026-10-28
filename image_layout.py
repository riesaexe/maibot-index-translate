"""译图的局部文字清除与颜色采样，不改动文字框以外的像素。"""

from typing import Tuple

import numpy as np

from PIL import Image


def clean_region(image: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int, int]]:
    left, top, right, bottom = box
    patch = np.array(image.crop(box).convert('RGB'))
    border = np.concatenate((patch[0], patch[-1], patch[:, 0], patch[:, -1]))
    background = np.median(border, axis=0)
    distance = np.max(np.abs(patch.astype(np.float32) - background), axis=2)
    foreground = patch[distance > 50]
    if len(foreground):
        color = tuple(int(value) for value in np.median(foreground, axis=0))
    else:
        color = (32, 32, 32) if float(background.mean()) > 128 else (240, 240, 240)
    # 用主底色像素拟合平滑背景，拒绝相邻行文字；不能逐列复制边缘。
    margin = 5
    outer = (max(0, left - margin), max(0, top - margin),
             min(image.width, right + margin), min(image.height, bottom + margin))
    padded = np.array(image.crop(outer).convert('RGB'))
    x, y = left - outer[0], top - outer[1]
    height, width = patch.shape[:2]
    yy, xx = np.indices(padded.shape[:2])
    samples = padded.reshape(-1, 3).astype(float)
    buckets = (samples // 24).astype(int)
    keys = buckets[:, 0] * 121 + buckets[:, 1] * 11 + buckets[:, 2]
    dominant = np.bincount(keys).argmax()
    center = np.median(samples[keys == dominant], axis=0)
    selected = np.max(np.abs(samples - center), axis=1) < 28
    coordinates = np.column_stack((np.ones(xx.size), xx.ravel(), yy.ravel()))
    coefficients = np.zeros((3, 3))
    coefficients[0] = center
    for _ in range(3):
        if selected.sum() < 6:
            break
        coefficients = np.linalg.lstsq(coordinates[selected], samples[selected], rcond=None)[0]
        selected &= np.max(np.abs(coordinates @ coefficients - samples), axis=1) < 5
    py, px = np.indices((height, width))
    targets = np.stack((np.ones_like(px), px + x, py + y), axis=-1)
    restored = np.clip(np.rint(targets @ coefficients), 0, 255).astype(np.uint8)
    result = image.copy()
    result.paste(Image.fromarray(restored), (left, top))
    return result, color


def ink_height(image: Image.Image, box: Tuple[int, int, int, int]) -> int:
    patch = np.array(image.crop(box).convert('RGB')).astype(float)
    border = np.concatenate((patch[0], patch[-1], patch[:, 0], patch[:, -1]))
    background = np.median(border, axis=0)
    ink = np.max(np.abs(patch - background), axis=2) > 40
    rows = np.flatnonzero(ink.sum(axis=1) >= 2)
    return int(rows[-1] - rows[0] + 1) if len(rows) else max(8, box[3] - box[1] - 6)


def clean_paper(image: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int, int]]:
    """仅修复深色墨迹，保留纸纹、横线和光照，不填充整个矩形。"""
    import cv2
    patch = np.array(image.crop(box).convert('RGB'))
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    # 局部阈值跟随纸张阴影，不能用全段统一亮度把阴影也判为文字。
    window = max(31, min(151, min(patch.shape[:2]) // 2 * 2 - 1))
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, window, 18)
    protected = foreground_protection(image.crop(box))
    mask[protected] = 0
    selected = gray[mask > 0]
    foreground = (mask > 0) & (gray <= np.quantile(selected, .2)) if len(selected) else mask > 0
    pixels = patch[foreground]
    color = tuple(int(v) for v in np.median(pixels, axis=0)) if len(pixels) else (40, 45, 65)
    # 校正后的长直横线不是笔画，保留其未被文字覆盖的部分。
    rules = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, max(80, patch.shape[1] // 4)), np.uint8))
    mask[rules > 0] = 0
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
    mask[rules > 0] = 0
    mask[protected] = 0
    # 同一掩膜同步排除三个色通道，避免蓝墨水的蓝通道被误当成浅色纸张。
    valid = (mask == 0).astype(np.float32)
    span = max(31, min(301, patch.shape[1] // 3)) | 1
    weights = cv2.GaussianBlur(valid, (span, span), 0)
    background_rgb = np.empty_like(patch, dtype=np.float32)
    for channel in range(3):
        samples = cv2.GaussianBlur(patch[:, :, channel].astype(np.float32) * valid, (span, span), 0)
        background_rgb[:, :, channel] = samples / np.maximum(weights, .0001)
    if np.any(weights < .0001):
        background_rgb[weights < .0001] = np.median(patch[valid > 0], axis=0) if np.any(valid) else np.median(patch.reshape(-1, 3), axis=0)
    alpha = cv2.GaussianBlur(mask, (9, 9), 0).astype(float)[:, :, None] / 255
    alpha[protected] = 0
    restored = np.clip(patch * (1 - alpha) + background_rgb * alpha, 0, 255).round().astype(np.uint8)
    result = image.copy()
    result.paste(Image.fromarray(restored), box[:2])
    return result, color


def foreground_protection(image: Image.Image) -> np.ndarray:
    """保护文字附近的肤色实物，避免擦字和排字侵入手部。"""
    import cv2
    pixels = np.array(image.convert('RGB')).astype(float)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    skin = (red > green * 1.12) & (red > blue * 1.15) & (blue > green * .65) & (red > 90) & (green > 45)
    # 棕色墨迹也满足肤色比例；只有内部有足够厚度的连通实物才保护，不能恢复细笔画。
    binary = skin.astype(np.uint8)
    diameter = max(9, min(31, min(binary.shape) // 25)) | 1
    core = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    count, labels = cv2.connectedComponents(binary)
    support = np.bincount(labels[core > 0], minlength=count)
    selected = support >= 25
    selected[0] = False
    protected = selected[labels].astype(np.uint8)
    return cv2.dilate(protected, np.ones((11, 11), np.uint8)).astype(bool)


def restore_foreground(source: Image.Image, result: Image.Image) -> None:
    protected = foreground_protection(source)
    if np.any(protected):
        pixels = np.array(result)
        pixels[protected] = np.array(source)[protected]
        result.paste(Image.fromarray(pixels))

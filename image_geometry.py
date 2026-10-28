"""保留文字四边形，在阅读坐标中合并同一段落，再映射回原图。"""
from typing import List, Tuple
import math

import cv2
import numpy as np
from PIL import Image

Quad = Tuple[Tuple[float, float], ...]


def rectify(image: Image.Image, points: Quad) -> Image.Image:
    source = np.asarray(points, dtype=np.float32)
    width = max(1, round(max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3]))))
    height = max(1, round(max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1]))))
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(source, target)
    return Image.fromarray(cv2.warpPerspective(np.array(image), matrix, (width, height), borderMode=cv2.BORDER_REPLICATE))


def composite(canvas: Image.Image, patch: Image.Image, points: Quad) -> None:
    source = np.float32([[0, 0], [patch.width - 1, 0], [patch.width - 1, patch.height - 1], [0, patch.height - 1]])
    matrix = cv2.getPerspectiveTransform(source, np.float32(points))
    warped = cv2.warpPerspective(np.array(patch.convert('RGBA')), matrix, canvas.size, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(np.full((patch.height, patch.width), 255, np.uint8), matrix, canvas.size)
    warped[:, :, 3] = mask
    canvas.paste(Image.fromarray(warped), (0, 0), Image.fromarray(warped[:, :, 3]))


def paragraphs(lines: List[Tuple[Quad, str]], size: Tuple[int, int]) -> List[Tuple[Quad, str, int]]:
    angles = [math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]) for p, _ in lines]
    heights = [np.linalg.norm(np.array(p[3]) - p[0]) for p, _ in lines]
    # 各方向分别建阅读坐标系，不能让正文的角度覆盖边栏或旋转文字。
    clusters: List[List[int]] = []
    for i, angle in enumerate(angles):
        for cluster in clusters:
            if abs(math.atan2(math.sin(angle - angles[cluster[0]]), math.cos(angle - angles[cluster[0]]))) < .3:
                cluster.append(i)
                break
        else:
            clusters.append([i])
    if len(clusters) > 1:
        return [item for cluster in clusters for item in paragraphs([lines[i] for i in cluster], size)]
    angle = float(np.median(angles))
    if abs(angle) >= 1.0:
        return [(points, text, round(float(height))) for (points, text), height in zip(lines, heights, strict=True)]
    rotation = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    boxes = []
    for points, _ in lines:
        projected = np.array(points) @ rotation.T
        boxes.append((projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()))
    groups: List[List[int]] = []
    # 先连同行片段，再连相邻行，避免 OCR 把手写诗拆成独立单词。
    for i, box in enumerate(boxes):
        matching = []
        for n, group in enumerate(groups):
            for j in group:
                other = boxes[j]
                h = min(heights[i], heights[j])
                overlap_y = min(box[3], other[3]) - max(box[1], other[1])
                overlap_x = min(box[2], other[2]) - max(box[0], other[0])
                gap_y = max(box[1], other[1]) - min(box[3], other[3])
                gap_x = max(box[0], other[0]) - min(box[2], other[2])
                same_line = overlap_y > h * .35 and gap_x < h * 1.8 and abs(angle) < 1.0
                next_line = overlap_x > min(box[2] - box[0], other[2] - other[0]) * .45 and gap_y < h * .85 and abs(angle) < 1.0
                from .image_detection import chinese_only
                if chinese_only(lines[i][1]) != chinese_only(lines[j][1]):
                    continue
                if abs(angles[i] - angles[j]) < .22 and max(heights[i], heights[j]) < h * 2.5 and (same_line or next_line):
                    matching.append(n)
                    break
        merged = [i]
        for n in reversed(matching):
            merged.extend(groups.pop(n))
        groups.append(merged)
    result = []
    for group in groups:
        values = np.array([boxes[i] for i in group])
        padding = max(4, float(np.median([heights[i] for i in group])) * (.12 if abs(angle) >= 1.0 else .5))
        left, top = values[:, :2].min(axis=0) - [4, padding]
        right, bottom = values[:, 2:].max(axis=0) + [4, padding]
        quad = np.array([[left, top], [right, top], [right, bottom], [left, bottom]]) @ rotation
        quad[:, 0] = quad[:, 0].clip(0, size[0] - 1)
        quad[:, 1] = quad[:, 1].clip(0, size[1] - 1)
        ordered = sorted(group, key=lambda i: (round(boxes[i][1] / max(1, np.median(heights))), boxes[i][0]))
        result.append((tuple(tuple(float(v) for v in p) for p in quad), ' '.join(lines[i][1] for i in ordered), round(float(np.median([heights[i] for i in group])))))
    return sorted(result, key=lambda item: min(p[1] for p in item[0]))

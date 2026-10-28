"""仅在模型确认文字之后使用气泡背景约束排版，不把白色角色区域当文字。"""
from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image

from .comic_detector import Box, ComicBlock


def balloons(image: Image.Image, blocks: Tuple[ComicBlock, ...]) -> Dict[int, Tuple[Box, np.ndarray]]:
    if not any(block.japanese for block in blocks):
        return {}
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    white = cv2.erode((gray > 235).astype(np.uint8), np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    output = {}
    for index, block in enumerate(blocks):
        if not block.japanese:
            continue
        left, top, right, bottom = block.box
        region = labels[top:bottom, left:right]
        choices, counts = np.unique(region[region > 0], return_counts=True)
        if not len(choices):
            continue
        label = int(choices[counts.argmax()])
        x, y, width, height, area = map(int, stats[label])
        if area > gray.size * .08 or area < (right - left) * (bottom - top) * .45:
            continue
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(component)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        filled = cv2.erode(filled, np.ones((5, 5), np.uint8))
        safe = largest_rectangle(filled)
        if safe is None:
            continue
        a, b, c, d = safe
        safe_box = (x + a, y + b, x + c, y + d)
        allowed = np.zeros(gray.shape, np.uint8)
        allowed[y:y + height, x:x + width] = filled
        output[index] = (safe_box, allowed)
    return output


def largest_rectangle(mask: np.ndarray) -> Box | None:
    heights = np.zeros(mask.shape[1], dtype=int)
    best = 0
    rectangle = None
    for y, row in enumerate(mask):
        heights = np.where(row > 0, heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            value = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > value:
                left, height = stack.pop()
                area = (x - left) * height
                if area > best:
                    best = area
                    rectangle = (left, y - height + 1, x, y + 1)
                start = left
            stack.append((start, value))
    return rectangle

"""触发命令按字面量匹配，空白分隔兼容聊天客户端。"""
import re
from typing import Final

DEFAULT_COMMAND: Final = '/rs 翻译'


def normalize_command(value: str) -> str:
    if '\n' in value or '\r' in value:
        raise ValueError('触发命令只能填写一行文字')
    command = ' '.join(value.split())
    if not command or len(command) > 40:
        raise ValueError('触发命令长度必须为 1～40 个字符')
    return command


def command_prefix(value: str) -> str:
    return r'^\s*' + r'\s+'.join(re.escape(part) for part in normalize_command(value).split()) + r'(?=\s|\[|$)'


def command_pattern(value: str) -> str:
    return command_prefix(value) + r'[\s\S]*$'

"""翻译插件设置与触发命令校验。"""
from typing import Literal

from maibot_sdk import Field, PluginConfigBase
from pydantic import field_validator

from .commands import DEFAULT_COMMAND, normalize_command


class PluginOptions(PluginConfigBase):
    config_version: str = '1.4.0'
    enabled: bool = Field(default=True, description='启用中文翻译')
    command: str = Field(default=DEFAULT_COMMAND, description='完整触发命令，例如 /rs 翻译、/tr；修改后重载插件生效')
    model: Literal['Index-Translate-35B-A3B', 'Index-Translate-9B', 'Index-Translate-2B'] = Field(
        default='Index-Translate-35B-A3B', description='翻译模型，文字和图片均可使用')
    comic_detector: Literal['auto', 'on', 'off'] = Field(default='auto', description='漫画专用检测：auto 自动识别，on 强制使用，off 关闭；检测模型约 95 MB，日文识别模型约 444 MB，首次使用自动下载')
    prewarm_models: bool = Field(default=False, description='启动后预热模型：延迟 30 秒并等待本插件翻译空闲，在后台加载漫画检测与日文 OCR；缓存缺失时会下载约 95 MB + 444 MB。关闭漫画模式时不预热')
    font_path: str = Field(default='', description='译图中文字体路径；留空自动使用系统中文字体')

    @field_validator('command')
    @classmethod
    def validate_command(cls, value: str) -> str:
        return normalize_command(value)


class TranslationConfig(PluginConfigBase):
    plugin: PluginOptions = Field(default_factory=PluginOptions)

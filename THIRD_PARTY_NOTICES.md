# 第三方依赖与模型

插件业务代码独立编写，以下依赖与模型不因插件的 MIT 许可而改变其许可证。安装或使用时应同时遵循对应上游条款。

| 项目 | 用途 | 许可证 / 来源 |
| --- | --- | --- |
| [Comic Text Detector](https://github.com/dmMaze/comic-text-detector) | 漫画文字块、像素掩膜 | GPL-3.0；模型来自 [manga-image-translator beta-0.3](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3) |
| [Manga OCR](https://github.com/kha-white/manga-ocr) | 日文漫画 OCR | Apache-2.0 |
| [manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) | 日文识别权重 | Apache-2.0；固定 revision 见 `model_sources.json` |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) | 普通图像文字定位、小字补检 | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 模型推理 | MIT |
| [OpenCV](https://github.com/opencv/opencv) | 几何变换、掩膜处理 | Apache-2.0（当前依赖版本） |

其他运行依赖由宿主根据 `_manifest.json` 安装，保留各依赖包内的许可证。模型权重在首次需要时下载到 `.models/`，不包含在 Git 仓库或源码 Release 中。系统字体仅从用户设备读取，不随插件分发。

翻译服务由哔哩哔哩提供，本项目不代表哔哩哔哩，也不拥有该服务或其模型权重。

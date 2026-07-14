# 《动手学深度学习》(D2L) — marimo 版本

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/)

这是 [《动手学深度学习》](https://zh.d2l.ai/)（Dive into Deep Learning，D2L）中文版的 **marimo notebook** 版本。

- 📖 原书：[zh.d2l.ai](https://zh.d2l.ai/)
- 🔬 框架：**PyTorch**
- 📓 格式：marimo `.py` notebook（142 个）
- 🌐 在线运行：[molab](https://molab.marimo.io/)（免费 NVIDIA RTX Pro 6000 GPU）

## 为什么用 marimo？

- ✅ **Git 友好**：纯 Python 文件，diff 清晰
- ✅ **无隐藏状态**：删一个 cell，它的变量也跟着消失
- ✅ **反应式执行**：改代码自动跑依赖 cell（也可设 lazy 模式手动控制）
- ✅ **免费 GPU**：一键在 molab 上跑，NVIDIA RTX Pro 6000（96GB VRAM）

## 快速开始

### 本地运行

```bash
# 安装 marimo
uv tool install marimo

# 启动任意章节
marimo edit chapter_preliminaries/ndarray.py
```

### 在线运行（免费 GPU）

1. 打开 [molab](https://molab.marimo.io/)
2. 粘贴本仓库任意 `.py` 文件的 GitHub URL
3. 点击 GPU 开关，开始学习

## 目录

| 章节 | 内容 |
|------|------|
| chapter_introduction | 引言 |
| chapter_preliminaries | 预备知识 |
| chapter_linear-networks | 线性神经网络 |
| chapter_multilayer-perceptrons | 多层感知机 |
| chapter_deep-learning-computation | 深度学习计算 |
| chapter_convolutional-neural-networks | 卷积神经网络 |
| chapter_convolutional-modern | 现代卷积神经网络 |
| chapter_recurrent-neural-networks | 循环神经网络 |
| chapter_recurrent-modern | 现代循环神经网络 |
| chapter_attention-mechanisms | 注意力机制 |
| chapter_optimization | 优化算法 |
| chapter_computational-performance | 计算性能 |
| chapter_computer-vision | 计算机视觉 |
| chapter_natural-language-processing-pretraining | 自然语言处理：预训练 |
| chapter_natural-language-processing-applications | 自然语言处理：应用 |

## 许可

本书内容采用 [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可，代码采用 [MIT-0](https://github.com/d2l-ai/d2l-zh/blob/master/LICENSE) 许可。

## 致谢

原始内容来自 [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh) 和 [d2l-ai/d2l-zh-pytorch-colab](https://github.com/d2l-ai/d2l-zh-pytorch-colab)。

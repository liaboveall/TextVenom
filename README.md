# 📚🔍 TextVenom: Deep OCR Adversarial Playground

*让AI读错字的艺术 - The Art of Making AI Misread Text*

---

## 🎭 项目简介

TextVenom 是一个专门针对深度文本识别（OCR）模型的对抗攻击研究项目。灵感来源于 [CLOVA AI 的深度文本识别基准](https://github.com/clovaai/deep-text-recognition-benchmark)，本项目通过实现 BIM（Basic Iterative Method）等对抗攻击算法，探索如何通过微小的像素扰动让强大的 OCR 模型"看花眼"。

## 🎯 为什么叫 TextVenom？

- **Text**: 专注于文本识别领域
- **Venom**: 像毒液一样的微小扰动，却能产生致命的效果
- 寓意：看似无害的图像扰动，却能让 AI 模型彻底"中毒"，产生错误的识别结果

## ✨ 主要特性

- 🎪 **对抗攻击算法**：实现了 BIM（Basic Iterative Method）攻击
- 🎨 **可视化支持**：直观展示原始图像 vs 对抗样本的对比效果
- 📊 **详细评估指标**：计算攻击成功率、L2/L∞范数等关键指标
- 🎯 **多模型支持**：兼容 CTC 和 Attention 机制的文本识别模型
- 🚀 **Windows 优化**：提供专门的 Windows 版本实现

## 🛠️ 支持的模型架构

项目支持以下预训练模型的攻击测试：

- `None-ResNet-None-CTC.pth`
- `None-VGG-BiLSTM-CTC.pth`  
- `TPS-ResNet-BiLSTM-Attn.pth`
- `TPS-ResNet-BiLSTM-CTC.pth`

## 📁 项目结构

```
TextVenom/
├── 📄 attack.py              # 主要攻击脚本（Linux/macOS）
├── 📄 attack_win.py          # Windows 优化版本
├── 📁 src/                   # 核心源码模块
│   ├── 🧠 model.py           # 模型定义
│   ├── 📊 dataset.py         # 数据集处理
│   ├── 🛠️ utils.py           # 工具函数
│   ├── 🎨 visualization.py   # 可视化工具
│   └── 📁 modules/           # 模型组件
├── 📁 saved_models/          # 预训练模型
├── 📁 CUTE80/               # 测试数据集
└── 📖 README.md             # 本文档
```

## 🚀 快速开始

### 环境要求

按照目标项目进行环境配置
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

### 下载预训练模型

在开始攻击测试前，请先下载所需的预训练模型：

🔗 **模型下载链接**: [https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)

请下载以下四个预训练模型文件并放置在 `saved_models/` 目录中：

- ✅ `None-ResNet-None-CTC.pth`
- ✅ `None-VGG-BiLSTM-CTC.pth`
- ✅ `TPS-ResNet-BiLSTM-Attn.pth`
- ✅ `TPS-ResNet-BiLSTM-CTC.pth`

### 基本使用

#### Windows 用户

```powershell
# 运行基本攻击测试
python attack_win.py

# 指定特定模型测试
python attack_win.py --model_path "saved_models/TPS-ResNet-BiLSTM-Attn.pth"
```

#### Linux/macOS 用户

```bash
# 运行基本攻击测试
python attack.py

# 指定特定模型测试  
python attack.py --model_path "saved_models/TPS-ResNet-BiLSTM-Attn.pth"
```

### 🎨 攻击效果可视化

程序会自动生成对比图像，展示：
- 📸 原始图像 vs 对抗样本
- 📝 原始识别结果 vs 攻击后结果
- 📈 扰动强度可视化

## 🔬 攻击原理

### BIM 攻击算法

BIM（Basic Iterative Method）是一种迭代式对抗攻击方法：

1. **初始化**：从原始图像开始
2. **迭代扰动**：在每次迭代中计算梯度并添加微小扰动
3. **约束限制**：确保扰动在指定范围内（ε-球约束）
4. **收敛**：经过多次迭代生成最终对抗样本

### 关键参数

- `epsilon (ε)`: 最大扰动幅度（默认 0.3）
- `alpha (α)`: 每次迭代的步长（默认 0.01）
- `num_iterations`: 迭代次数（默认 20）

## 📊 评估指标

- **攻击成功率**：对抗样本导致错误识别的比例
- **L2 范数**：扰动的欧几里得距离
- **L∞ 范数**：扰动的最大像素变化

## 🎯 实验结果

项目可以生成详细的攻击报告，包括：

```
模型: TPS-ResNet-BiLSTM-Attn
攻击成功率: 85.3%
平均 L2 范数: 0.123
平均 L∞ 范数: 0.301
```

## 🤔 研究意义

TextVenom 项目揭示了现代 OCR 系统的脆弱性：

1. **安全评估**：帮助评估文本识别系统的鲁棒性
2. **防御研究**：为开发更强健的模型提供测试基准
3. **学术价值**：深入理解对抗攻击在计算机视觉中的机制

## 🛡️ 防御建议

- **对抗训练**：使用对抗样本增强训练数据
- **输入预处理**：应用去噪和平滑技术
- **集成方法**：使用多模型投票机制
- **检测机制**：部署对抗样本检测器

## 🚨 免责声明

本项目仅用于学术研究和安全评估目的。请勿将此技术用于恶意攻击或非法活动。使用者需自行承担相关责任。

## 📖 参考文献

- [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- Kurakin, A., Goodfellow, I., & Bengio, S. (2016). Adversarial examples in the physical world.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples.

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！如果你有新的攻击算法想法或发现了 Bug，请随时联系我们。

---

*"The best way to attack is to make the enemy think they're winning while you control the game."* 🎭
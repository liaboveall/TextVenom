# TextVenom: Deep OCR Adversarial Playground

[![English](https://img.shields.io/badge/lang-English-blue)](README.md) [![中文](https://img.shields.io/badge/语言-中文-brightgreen)](README_CN.md)

## 项目简介

TextVenom 是一个针对深度文本识别（OCR）模型的对抗攻击研究项目，灵感来源于 [CLOVA AI 基准](https://github.com/clovaai/deep-text-recognition-benchmark)。通过实现 BIM（Basic Iterative Method）等方法，展示微小像素扰动如何显著影响识别结果。

## 主要特性

- 对抗攻击：当前实现 BIM（可扩展更多算法）
- 可视化：原始图像 vs 对抗样本，对比扰动
- 评估指标：攻击成功率、L2 / L∞ 范数
- 多模型支持：兼容 CTC 与 Attention 结构
- Windows 适配：提供专门脚本 `attack_win.py`

## 支持的模型

放置于 `saved_models/` 目录：

- ✅ `None-ResNet-None-CTC.pth`
- ✅ `None-VGG-BiLSTM-CTC.pth`
- ✅ `TPS-ResNet-BiLSTM-Attn.pth`
- ✅ `TPS-ResNet-BiLSTM-CTC.pth`

## 项目结构

```
TextVenom/
├── attack.py              # 主攻击脚本（Linux/macOS）
├── attack_win.py          # Windows 版本
├── src/
│   ├── model.py           # 模型定义
│   ├── dataset.py         # 数据集处理
│   ├── utils.py           # 工具函数
│   ├── visualization.py   # 可视化
│   └── modules/           # 组件子模块
├── saved_models/          # 预训练模型
├── CUTE80/                # 测试数据集
└── README_CN.md           # 中文文档
```

## 快速开始

### 环境

参考上游项目配置依赖：<https://github.com/clovaai/deep-text-recognition-benchmark>

### 下载模型

链接：<https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW>

将四个模型放入 `saved_models/`。

### 运行示例

Windows:

```powershell
python attack_win.py
python attack_win.py --model_path "saved_models/TPS-ResNet-BiLSTM-Attn.pth"
```

Linux / macOS:

```bash
python attack.py
python attack.py --model_path "saved_models/TPS-ResNet-BiLSTM-Attn.pth"
```

## 可视化

自动输出：
- 原图 vs 对抗样本
- 原/对抗预测结果
- 扰动热力/差分图

## 攻击原理：BIM

迭代更新样本：
1. 初始化为原始输入
2. 计算损失梯度并按步长更新
3. 投影回 ε 范围（L∞ 约束）
4. 若达到迭代轮次或早停条件则输出

参数：
- `epsilon`：最大扰动 (默认 0.3)
- `alpha`：每步扰动 (默认 0.01)
- `num_iterations`：迭代次数 (默认 20)

## 评估指标

- 攻击成功率
- 平均 L2 范数
- 平均 L∞ 范数

示例：
```
模型: TPS-ResNet-BiLSTM-Attn
攻击成功率: 85.3%
平均 L2: 0.123
平均 L∞: 0.301
```

## 研究意义

揭示 OCR 系统在对抗扰动下的脆弱性，为鲁棒性与防御方法研究提供实验平台。

## 防御建议

- 对抗训练
- 输入预处理（去噪/平滑）
- 多模型集成
- 对抗样本检测

## 免责声明

仅限科研与安全评估，不得用于非法用途，风险自负。

## 参考

- [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- Kurakin et al., 2016
- Goodfellow et al., 2014

## 贡献

欢迎 Issue / PR，鼓励添加更多攻击与防御方法。

---

"The best way to attack is to make the enemy think they're winning while you control the game."

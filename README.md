# TextVenom: Deep OCR Adversarial Playground

[![English](https://img.shields.io/badge/lang-English-blue)](README.md) [![中文](https://img.shields.io/badge/lang-中文-brightgreen)](README_CN.md)

An experimental playground for crafting adversarial attacks against deep Optical Character Recognition (OCR) models. Inspired by the original [CLOVA AI benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). The project currently implements the BIM (Basic Iterative Method) attack to demonstrate how small perturbations can flip model predictions.

## Features

- Adversarial attack implementation: BIM (extensible design for more)
- Visualization of original vs adversarial images and perturbation maps
- Evaluation metrics: attack success rate, L2 and L∞ norms
- Supports both CTC-based and Attention-based recognition models
- Cross-platform: dedicated Windows script `attack_win.py`

## Supported Pretrained Models

Place the following weights inside `saved_models/`:

- ✅ `None-ResNet-None-CTC.pth`
- ✅ `None-VGG-BiLSTM-CTC.pth`
- ✅ `TPS-ResNet-BiLSTM-Attn.pth`
- ✅ `TPS-ResNet-BiLSTM-CTC.pth`

## Project Structure

```
TextVenom/
├── attack.py              # Main attack script (Linux/macOS)
├── attack_win.py          # Windows optimized script
├── src/
│   ├── model.py           # Model definitions
│   ├── dataset.py         # Dataset utilities
│   ├── utils.py           # Helper functions
│   ├── visualization.py   # Visualization helpers
│   └── modules/           # Sub modules
├── saved_models/          # Pretrained weights
├── CUTE80/                # Test dataset
└── README.md              # English documentation
```

## Getting Started

### Environment

Follow the upstream repository for dependency setup:
<https://github.com/clovaai/deep-text-recognition-benchmark>

### Download Weights

Link: <https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW>

Place the four model files into `saved_models/`.

### Usage

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

## Visualization

Automatically generated outputs:
- Original vs adversarial image comparison
- Original vs adversarial predictions
- Perturbation heat/difference map

## Attack Method: BIM

Iterative procedure:
1. Initialize with the clean input
2. Compute gradients and apply a step update
3. Project back into the ε L∞ ball
4. Stop after fixed iterations or early criteria

Parameters:
- `epsilon`: max perturbation (default 0.3)
- `alpha`: step size (default 0.01)
- `num_iterations`: iterations (default 20)

## Metrics

- Attack success rate
- Mean L2 norm
- Mean L∞ norm

Example:
```
Model: TPS-ResNet-BiLSTM-Attn
Attack Success Rate: 85.3%
Mean L2: 0.123
Mean L∞: 0.301
```

## Motivation

Highlights the vulnerability of OCR systems under carefully crafted perturbations and provides a baseline for robustness and defense research.

## Defense Suggestions

- Adversarial training
- Input preprocessing (denoise / smoothing)
- Ensemble strategies
- Adversarial sample detection

## Disclaimer

For research and security evaluation only. Do not use for malicious or illegal purposes. You assume all responsibility.

## References

- [Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
- Kurakin et al., 2016
- Goodfellow et al., 2014

## Contributing

Issues and PRs are welcome—especially new attack/defense methods.

---

"The best way to attack is to make the enemy think they're winning while you control the game."
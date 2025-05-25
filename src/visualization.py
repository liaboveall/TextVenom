import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def tensor_to_image(tensor):
    """将tensor转换为可显示的图像"""
    img = (tensor + 1) / 2
    if len(img.shape) == 4:
        img = img[0]
    img = img.cpu().detach().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1)

def clean_text(text):
    """清理文本中的特殊标记"""
    if isinstance(text, str):
        return text.split('[s]')[0]  # 只保留[s]之前的内容
    return text

def visualize_attack_results(original_image, adv_image, original_text, adv_text, save_path):
    """可视化单个攻击结果"""
    perturbation = adv_image - original_image
    
    # 清理识别文本中的特殊标记
    original_text = clean_text(original_text)
    adv_text = clean_text(adv_text)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    axs[0].imshow(tensor_to_image(original_image))
    axs[0].set_title(f'Original\nPrediction: {original_text}')
    axs[0].axis('off')
    
    # 显示对抗样本
    axs[1].imshow(tensor_to_image(adv_image))
    axs[1].set_title(f'Adversarial\nPrediction: {adv_text}')
    axs[1].axis('off')
    
    # 显示放大后的扰动
    perturbation_vis = tensor_to_image(5 * perturbation)
    axs[2].imshow(perturbation_vis)
    axs[2].set_title('Perturbation (×5)')
    axs[2].axis('off')
    
    plt.suptitle('Attack Visualization', y=1.05)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def save_batch_visualizations(original_images, adv_images, original_texts, adv_texts, model_name, output_dir='./attack_results'):
    """保存一批图像的攻击结果"""
    # 创建输出目录
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 可视化每个样本
    for i in range(min(len(original_texts), 10)):  # 只保存前5个样本
        save_path = os.path.join(model_output_dir, f'sample_{i+1}.png')
        visualize_attack_results(
            original_images[i],
            adv_images[i],
            original_texts[i],
            adv_texts[i],
            save_path
        )

def create_comparison_grid(results_dict, output_path='./attack_results/comparison.png'):
    """创建不同模型攻击效果的对比网格"""
    num_models = len(results_dict)
    fig, axs = plt.subplots(num_models, 3, figsize=(15, 5*num_models))
    
    if num_models == 1:
        axs = axs.reshape(1, -1)
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        success_rate, l2_norm, linf_norm = results
        
        # 绘制条形图
        axs[idx, 0].bar(['Success Rate'], [success_rate], color='red')
        axs[idx, 0].set_title(f'{model_name}\nSuccess Rate: {success_rate:.2f}%')
        axs[idx, 0].set_ylim(0, 100)
        
        axs[idx, 1].bar(['L2 Norm'], [l2_norm], color='blue')
        axs[idx, 1].set_title(f'L2 Norm: {l2_norm:.4f}')
        
        axs[idx, 2].bar(['L∞ Norm'], [linf_norm], color='green')
        axs[idx, 2].set_title(f'L∞ Norm: {linf_norm:.4f}')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
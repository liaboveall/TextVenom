import os
import torch
import torch.nn.functional as F
import numpy as np
from src.utils import CTCLabelConverter, AttnLabelConverter
from src.dataset import hierarchical_dataset, AlignCollate
from src.model import Model
from src.visualization import save_batch_visualizations, create_comparison_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BIMAttacker:
    """Basic Iterative Method (BIM) 攻击实现"""
    def __init__(self, model, epsilon=0.3, alpha=0.01, num_iterations=20):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations
        
    def compute_loss(self, preds, text_for_loss, length_for_loss, preds_size, opt):
        """计算攻击损失"""
        if 'CTC' in opt.Prediction:
            preds = preds.log_softmax(2)
            cost = F.ctc_loss(
                preds.permute(1, 0, 2),
                text_for_loss,
                preds_size,
                length_for_loss,
                zero_infinity=True
            )
        else:
            target = text_for_loss[:, 1:]  # 移除 [GO] 符号
            cost = F.cross_entropy(
                preds.view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
                ignore_index=0
            )
        return cost

    def attack_single_batch(self, images, labels, converter, opt):
        """对单个batch进行BIM攻击"""
        # 保存原始训练模式并切换到训练模式
        was_training = self.model.training
        self.model.train()
        
        images = images.to(device)
        batch_size = images.size(0)
        
        # 准备目标
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        text_for_loss, length_for_loss = text_for_loss.to(device), length_for_loss.to(device)
        
        # 初始化对抗样本
        x_adv = images.clone().detach()
        
        # BIM 迭代
        for _ in range(self.num_iterations):
            x_adv.requires_grad = True
            
            # 前向传播
            if 'CTC' in opt.Prediction:
                preds = self.model(x_adv, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
            else:
                preds = self.model(x_adv, text_for_pred[:, :-1], is_train=False)
                preds_size = None
            
            # 计算损失
            loss = self.compute_loss(preds, text_for_loss, length_for_loss, preds_size, opt)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 获取梯度符号并更新
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv.detach() + self.alpha * grad_sign
            
            # 限制扰动范围
            delta = torch.clamp(x_adv - images, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(images + delta, -1, 1).detach()
        
        # 恢复模型状态
        if not was_training:
            self.model.eval()
            
        return x_adv

    def attack_dataset(self, eval_loader, converter, opt, model_name):
        """对整个数据集进行攻击并计算指标"""
        l2_norms = []
        linf_norms = []
        success_count = 0
        total_count = 0
        visualization_batch = None
        
        for i, (images, labels) in enumerate(eval_loader):
            print(f"\rProcessing batch {i+1}/{len(eval_loader)}", end='')
            
            # 将图像移到设备上并生成对抗样本
            images = images.to(device)
            adv_images = self.attack_single_batch(images, labels, converter, opt)
            
            # 计算范数 (确保在同一设备上)
            perturbation = adv_images - images
            l2_norms.append(torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1).mean().item())
            linf_norms.append(torch.norm(perturbation.view(perturbation.shape[0], -1), p=float('inf'), dim=1).mean().item())
            
            # 获取预测结果
            self.model.eval()
            with torch.no_grad():
                # 原始图像预测
                if 'CTC' in opt.Prediction:
                    preds = self.model(images.to(device), torch.LongTensor(images.size(0)).fill_(0).to(device))
                    preds = preds.log_softmax(2).argmax(2)
                    original_texts = converter.decode(preds.data, torch.IntTensor([preds.size(1)] * images.size(0)).data)
                else:
                    preds = self.model(images.to(device), torch.LongTensor(images.size(0)).fill_(0).to(device), is_train=False)
                    _, preds_index = preds.max(2)
                    original_texts = converter.decode(preds_index, torch.IntTensor([opt.batch_max_length] * images.size(0)).data)
                    # 清理Attention模型的输出
                    original_texts = [text.split('[s]')[0] for text in original_texts]
                
                # 对抗样本预测
                if 'CTC' in opt.Prediction:
                    preds = self.model(adv_images, torch.LongTensor(images.size(0)).fill_(0).to(device))
                    preds = preds.log_softmax(2).argmax(2)
                    adv_texts = converter.decode(preds.data, torch.IntTensor([preds.size(1)] * images.size(0)).data)
                else:
                    preds = self.model(adv_images, torch.LongTensor(images.size(0)).fill_(0).to(device), is_train=False)
                    _, preds_index = preds.max(2)
                    adv_texts = converter.decode(preds_index, torch.IntTensor([opt.batch_max_length] * images.size(0)).data)
                    # 清理Attention模型的输出
                    adv_texts = [text.split('[s]')[0] for text in adv_texts]
                
                # 计算成功率
                for pred, label in zip(adv_texts, labels):
                    total_count += 1
                    if pred != label:
                        success_count += 1
                
                # 保存第一个批次用于可视化
                if visualization_batch is None:
                    visualization_batch = {
                        'original_images': images.to(device),  # 确保在正确的设备上
                        'adv_images': adv_images,
                        'original_texts': original_texts,
                        'adv_texts': adv_texts
                    }
        
        # 计算平均指标
        success_rate = success_count / total_count * 100
        avg_l2 = np.mean(l2_norms)
        avg_linf = np.mean(linf_norms)
        
        # 保存可视化结果
        if visualization_batch is not None:
            save_batch_visualizations(
                visualization_batch['original_images'],
                visualization_batch['adv_images'],
                visualization_batch['original_texts'],
                visualization_batch['adv_texts'],
                model_name
            )
        
        return success_rate, avg_l2, avg_linf

def attack_all_models():
    """对所有模型执行攻击"""
    # 基本配置
    opt = type('', (), {})()
    opt.imgH, opt.imgW = 32, 100
    opt.batch_max_length = 25
    opt.character = '0123456789abcdefghijklmnopqrstuvwxyz'
    opt.sensitive = False
    opt.PAD = False
    opt.num_fiducial = 20
    opt.input_channel = 1
    opt.output_channel = 512
    opt.hidden_size = 256
    opt.batch_size = 256
    opt.workers = 0
    opt.rgb = False
    opt.data_filtering_off = True
    
    # 模型配置
    models_config = [
        {
            'name': 'CRNN',
            'path': '.\\saved_models\\None-VGG-BiLSTM-CTC.pth',
            'Transformation': 'None',
            'FeatureExtraction': 'VGG',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'CTC'
        },
        {
            'name': 'Rosetta',
            'path': '.\\saved_models\\None-ResNet-None-CTC.pth',
            'Transformation': 'None',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'None',
            'Prediction': 'CTC'
        },
        {
            'name': 'STAR',
            'path': '.\\saved_models\\TPS-ResNet-BiLSTM-CTC.pth',
            'Transformation': 'TPS',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'CTC'
        },
        {
            'name': 'TRBA',
            'path': '.\\saved_models\\TPS-ResNet-BiLSTM-Attn.pth',
            'Transformation': 'TPS',
            'FeatureExtraction': 'ResNet',
            'SequenceModeling': 'BiLSTM',
            'Prediction': 'Attn'
        }
    ]
    
    # 数据加载配置
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    
    # 收集所有结果
    all_results = {}
    
    # 攻击每个模型
    for model_config in models_config:
        print(f"\n=== Attacking {model_config['name']} model ===")
        
        # 更新模型配置
        for k, v in model_config.items():
            if k != 'name' and k != 'path':
                setattr(opt, k, v)
        
        # 配置转换器
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)
        
        # 加载模型
        model = Model(opt).to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_config['path'], map_location=device))
        model.eval()
        
        # 创建攻击器
        attacker = BIMAttacker(model)
        
        # 加载数据集
        eval_data, _ = hierarchical_dataset(root='.\\CUTE80', opt=opt)
        eval_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=align_collate,
            pin_memory=True
        )
        
        # 执行攻击
        success_rate, avg_l2, avg_linf = attacker.attack_dataset(
            eval_loader, converter, opt, model_config['name'])
        
        # 存储结果
        all_results[model_config['name']] = (success_rate, avg_l2, avg_linf)
        
        # 打印结果
        print(f"\n{model_config['name']} Attack Results:")
        print(f"Attack Success Rate: {success_rate:.2f}%")
        print(f"Average L2 norm: {avg_l2:.4f}")
        print(f"Average L∞ norm: {avg_linf:.4f}")
    
    # 创建比较图
    create_comparison_grid(all_results)

if __name__ == '__main__':
    attack_all_models()
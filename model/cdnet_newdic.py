import os
import cv2
import numpy as np
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

import model.backbone_resnet as backbone_resnet
from collections import OrderedDict
from torch.nn import init
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

try:
    import clip  # for linear_assignment
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip

# 加载 SAM 模型
# sam_checkpoint = r"./sam_vit_b_01ec64.pth"  # SAM 权重文件路径
# model_type = "vit_b"  # 模型类型
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).cuda()
# predictor = SamPredictor(sam)


class PanopticFPN(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(PanopticFPN, self).__init__()
        self.backbone = backbone_resnet.__dict__[backbone](pretrained=True)
        self.decoder = FPNDecoder(backbone)

    def forward(self, x):
        feats = self.backbone(x)
        outs = self.decoder(feats)

        return outs


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, ):
        super(up, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.bilinear = bilinear

    def forward(self, x1, x2):
        if self.bilinear == True:
            x1 = nn.functional.interpolate(x1, scale_factor=2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class FPNDecoder(nn.Module):
    def __init__(self, backbone):
        super(FPNDecoder, self).__init__()

        if backbone == 'resnet18' or backbone == 'resnet34':
            print(backbone)
            mfactor = 1
            out_dim = 64
        else:
            mfactor = 4
            out_dim = 128

        self.layer0 = nn.Conv2d(64, out_dim, kernel_size=1)
        self.layer1 = nn.Conv2d(512 * mfactor // 8, out_dim, kernel_size=1)
        self.layer2 = nn.Conv2d(512 * mfactor // 4, out_dim, kernel_size=1)
        self.layer3 = nn.Conv2d(512 * mfactor // 2, out_dim, kernel_size=1)
        self.layer4 = nn.Conv2d(512 * mfactor, out_dim, kernel_size=1)

        self.up1 = up(out_dim * 2, out_dim)
        self.up2 = up(out_dim * 2, out_dim)
        self.up3 = up(out_dim * 2, out_dim)
        self.up4 = up(out_dim * 2, out_dim)

    def forward(self, x):

        x0 = self.layer0(x['res0'])
        x1 = self.layer1(x['res1'])
        x2 = self.layer2(x['res2'])
        x3 = self.layer3(x['res3'])
        x4 = self.layer4(x['res4'])

        x = {}
        x3 = self.up1(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up3(x2, x1)
        x0 = self.up4(x1, x0)

        return x0  # , x1, x2, x3

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y


class Classifier(nn.Module):
    def __init__(self, channel_num, class_num):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num * 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel_num * 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channel_num * 2, class_num, kernel_size=1)

    # def forward(self, x,prefix):
    def forward(self, x):

        output_dir = './heatmap'

        x = self.conv1(x)
        # save_heatmap(f, f'{prefix}input_f.png', output_dir)

        x = self.bn1(x)
        # save_heatmap(x, f'{prefix}input_xbn.png', output_dir)

        x = self.relu1(x)
        # save_heatmap(x, f'{prefix}input_xrelu.png', output_dir)

        x = self.conv2(x)
        # save_heatmap(x, f'{prefix}input_xfinal.png', output_dir)

        return x


def Load_Weight_FordataParallel(state_dict, need_dataparallel=0):
    if_dataparallel = 1
    for k, v in state_dict.items():
        name = k[:6]
        if name != "module":
            if_dataparallel = 0
    if need_dataparallel == 1:
        if if_dataparallel == 1:
            return state_dict
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k
                new_state_dict[name] = v
            return new_state_dict
    else:
        if if_dataparallel == 0:
            return state_dict
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            return new_state_dict

def visualize_tensor(tensor, name, output_dir='heatmaps', batch_idx=0, cmap='jet'):
        """
        可视化张量的热力图

        参数:
        tensor (torch.Tensor): 要可视化的张量
        name (str): 保存文件名
        output_dir (str): 输出目录
        batch_idx (int): 批次索引，默认为0
        cmap (str): 颜色映射，默认为'jet'
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 确保张量在CPU上
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # 处理不同维度的张量
        if tensor.dim() == 4:  # [B, C, H, W]
            # 可视化批次中第一个样本的每个通道
            tensor = tensor[batch_idx]  # [C, H, W]

            if tensor.size(0) > 1:  # 多通道特征图
                # 创建子图网格
                grid_size = int(np.ceil(np.sqrt(tensor.size(0))))
                fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
                axes = axes.flatten()

                for i in range(tensor.size(0)):
                    if i < len(axes):
                        img = axes[i].imshow(tensor[i].detach().numpy(), cmap=cmap)
                        axes[i].set_title(f'Channel {i}')
                        plt.colorbar(img, ax=axes[i])

                # 隐藏空白子图
                for i in range(tensor.size(0), len(axes)):
                    axes[i].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{name}.png'))
                plt.close()
            else:  # 单通道
                plt.figure(figsize=(10, 10))
                img = plt.imshow(tensor[0].detach().numpy(), cmap=cmap)
                plt.colorbar(img)
                plt.title(name)
                plt.savefig(os.path.join(output_dir, f'{name}.png'))
                plt.close()

        elif tensor.dim() == 3:  # [C, H, W]
            plt.figure(figsize=(10, 10))
            # 可视化第一个通道或平均值
            if tensor.size(0) > 1:
                img_data = tensor.mean(0).detach().numpy()
            else:
                img_data = tensor[0].detach().numpy()

            img = plt.imshow(img_data, cmap=cmap)
            plt.colorbar(img)
            plt.title(name)
            plt.savefig(os.path.join(output_dir, f'{name}.png'))
            plt.close()

        elif tensor.dim() == 2:  # [H, W]
            plt.figure(figsize=(10, 10))
            img = plt.imshow(tensor.detach().numpy(), cmap=cmap)
            plt.colorbar(img)
            plt.title(name)
            plt.savefig(os.path.join(output_dir, f'{name}.png'))
            plt.close()


def save_heatmap(tensor, filename, save_dir='./heatmap'):
    """
    保存热力图为PNG文件（提取前2个通道加权后归一化）

    参数:
        tensor (torch.Tensor): 输入张量，支持4D(batch, channel, h, w)或3D(channel, h, w)
        filename (str): 保存的文件名（不含目录）
        save_dir (str): 保存目录，默认'./heatmap'
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 处理多通道张量
    if tensor.dim() == 4:  # 4D: (batch, channel, h, w)
        tensor = tensor[0]  # 取第一个样本
    if tensor.dim() >= 3:  # 至少有通道维度
        channel_num = tensor.size(0)
        # 提取前2个通道（若存在）
        if channel_num >= 2:
            selected_channels = tensor[:2]  # 提取前2个通道
            print(f"已提取前2个通道（共{channel_num}个通道）")
        else:
            selected_channels = tensor  # 通道数不足2，使用全部通道
            print(f"警告：通道数不足2（{channel_num}个），使用全部通道")
        # 对前2个通道加权求和（默认等权重）
        weights = torch.ones(selected_channels.size(0), device=tensor.device)
        tensor = torch.sum(selected_channels * weights.view(-1, 1, 1), dim=0)

    # 归一化处理
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # 防止全零张量导致除零错误
    if tensor_max == tensor_min:
        normalized = torch.zeros_like(tensor)
    else:
        normalized = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

    # 转换为numpy数组
    np_tensor = normalized.detach().cpu().numpy()

    # 绘制热力图（提高分辨率）
    plt.figure(figsize=(10, 8), dpi=300)
    plt.axis("off")
    plt.imshow(np_tensor, cmap='jet', vmin=0, vmax=1)

    # 处理文件名冲突
    base_name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(save_dir, new_filename)):
        counter += 1
        new_filename = f"{base_name}_{counter}{ext}"

    # 保存文件并关闭画布
    save_path = os.path.join(save_dir, new_filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()





class ChangeDetector(nn.Module):
    def __init__(self, channel_num, class_num, key, word_num, backbone='resnet50',
                 clip_weight_path='./ViT-B-32.pt'):
        super(ChangeDetector, self).__init__()
        self.feature_extractor_deep = PanopticFPN(backbone=backbone)

        # SAM 模型
        # self.sam_predictor = SamPredictor(sam)

        self.classifier = Classifier(word_num * 2, class_num)
        self.conv1 = nn.Conv2d(1,3,3,padding=1)
        self.convsf1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel_num),
        )
        self.convsf2 = nn.Sequential(
            nn.Conv2d(channel_num, word_num, kernel_size=1, bias=False),
            nn.BatchNorm2d(word_num),
        )
        self.image_conv = double_conv(3, word_num)

        word_length = word_num
        self.dictionary = nn.Parameter(
            init.kaiming_uniform_(torch.randn(1, 1, word_num, word_length), a=math.sqrt(5))[0, 0])

        self.fcd1 = nn.Linear(word_num, channel_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.fcd2 = nn.Linear(channel_num, channel_num)

        self.eye = nn.Parameter(torch.eye(word_num, word_num))
        self.eye.requires_grad = False
        self.key = key
        self.bn1 = nn.BatchNorm2d(word_num)
        self.bn2 = nn.BatchNorm2d(word_num)

        self.upsample = nn.UpsamplingBilinear2d(size=(256, 256))
        self.channel_num = channel_num
        self.conv_1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.conv_t = nn.Conv2d(word_num * 2, word_num, kernel_size=1, bias=False)
        # 在 ChangeDetector 类中添加以下代码
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        # 加载 CLIP 模型
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # self.model_clip, self.clip_preprocess = clip.load(str(clip_weight_path), device=self.device)

        # 定义预定义文本描述
        self.lc_list = [
            # bld related.
            'roof', 'rooftop', 'building', 'house', 'apartment', 'residential', 'factory',
            # non-bld related.
            'vegetation', 'tree', 'vehicle', 'playground', 'baseball diamond', 'swimming pool', 'roundabout',
            'basketball court', 'bareland', 'sand'
        ]

    def Feature_meanlize(self, f, seg, num, max_num):

        bs, f_c, f_w, f_h = f.shape[0], f.shape[1], f.shape[2], f.shape[3]
        single_long = f_w * f_h
        single_base = (torch.arange(bs) * single_long).view(bs, 1, 1).cuda()

        seg_ = seg[:, :torch.max(num), :torch.max(max_num)]

        seg_onehot = (seg_ > 0).float().unsqueeze(3)

        leng = torch.sum(seg_ > 0, dim=2)
        seg_ = (seg_ + single_base).reshape(-1)

        f_ = f.permute(0, 2, 3, 1).reshape(bs * f_w * f_h, -1)
        f_x = f_[seg_.long()].reshape(bs, torch.max(num), torch.max(max_num), f_c)
        f_x = f_x * seg_onehot

        f_x = torch.sum(f_x, dim=2) / (leng.unsqueeze(2) + (leng.unsqueeze(2) == 0).float())

        f_x = f_x.repeat(torch.max(max_num), 1, 1, 1).permute(1, 2, 0, 3).reshape(
            bs * torch.max(num) * torch.max(max_num), f_c)

        f_[seg_.long()] = f_x
        f_ = f_.reshape(bs, f_w, f_h, f_c).permute(0, 3, 1, 2)

        return f_

    # def Feature_meanlize(self, x, seg, num, max_num1):
    #     """支持Felzenszwalb分割的特征平均化方法
    #         Args:
    #             x: 输入特征 (B, C, H, W)
    #             seg: 分割图 (B, H, W)，必须为torch.long类型
    #         """
    #     # 类型检查与转换
    #     if seg.dtype != torch.int64:
    #         seg = seg.long()  # 确保是int64类型
    #
    #     B, C, H, W = x.shape
    #     seg_flat = seg.view(B, -1)  # (B, H*W)
    #
    #     # 动态计算最大区域数
    #     max_num = int(seg_flat.max().item()) + 1  # 转换为Python整数
    #
    #     # 创建区域掩码 (B, max_num, H*W)
    #     mask = torch.zeros(B, max_num, H * W, device=x.device)
    #     mask.scatter_(1, seg_flat.unsqueeze(1), 1)  # 关键：需要int64类型的索引
    #
    #     # 计算区域特征均值
    #     x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
    #     sum_features = torch.bmm(mask, x_flat)  # (B, max_num, C)
    #     count = mask.sum(dim=2, keepdim=True) + 1e-8  # 区域像素计数
    #
    #     avg_features = sum_features / count
    #     expanded = avg_features.gather(1, seg_flat.unsqueeze(2).expand(-1, -1, C))
    #     return x + expanded.permute(0, 2, 1).view_as(x)

    # def dic_learning(self, f, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2):
    #     torch.cuda.empty_cache()
    #     # print((m).shape)
    #     S = torch.sigmoid(self.convsf2(self.convsf1(f + m)))
    #     S1 = S[:batch_size]
    #     S2 = S[batch_size:]
    #     S1 = self.Feature_meanlize(S1.detach(), seg1, num1, max_num1) + S1
    #     S2 = self.Feature_meanlize(S2.detach(), seg2, num2, max_num2) + S2
    #     Word_set1 = self.bn1(torch.matmul(S1.permute(0, 2, 3, 1), self.dictionary).permute(0, 3, 1, 2))
    #     Word_set2 = self.bn2(torch.matmul(S2.permute(0, 2, 3, 1), self.dictionary).permute(0, 3, 1, 2))
    #     Word_set1 = self.upsample(Word_set1)
    #     Word_set2 = self.upsample(Word_set2)
    #     c = torch.cat((Word_set1, Word_set2), dim=1)
    #     torch.cuda.empty_cache()
    #     return c


    #
    # def dic_learning(self, f, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2,
    #                   prefix):

    def dic_learning(self, f, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2,
                         ):
        """
        带可视化功能的字典学习函数

        参数:
        f, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2: 与原函数相同
        output_dir (str): 热力图保存目录
        prefix (str): 文件名前缀，用于标识不同的调用
        """
        output_dir = './heatmap'
        torch.cuda.empty_cache()
        # prefix = str(prefix[0])
        # print(type(prefix))
        # 可视化输入
        # save_heatmap(f, f'{prefix}input_f.png', output_dir)
        # save_heatmap(m, f'{prefix}input_m.png', output_dir)

        # 计算S
        S = torch.sigmoid(self.convsf2(self.convsf1(f + m)))
        # save_heatmap(S, f'{prefix}S.png', output_dir)

        # 分割S
        S1 = S[:batch_size]
        S2 = S[batch_size:]
        # save_heatmap(S1, f'{prefix}S1.png', output_dir)
        # save_heatmap(S2, f'{prefix}S2.png', output_dir)

        # 特征归一化
        S1_normalized = self.Feature_meanlize(S1.detach(), seg1, num1, max_num1) + S1
        S2_normalized = self.Feature_meanlize(S2.detach(), seg2, num2, max_num2) + S2
        # save_heatmap(S1_normalized, f'{prefix}S1_normalized.png', output_dir)
        # save_heatmap(S2_normalized, f'{prefix}S2_normalized.png', output_dir)

        # 计算Word sets
        Word_set1 = self.bn1(torch.matmul(S1_normalized.permute(0, 2, 3, 1), self.dictionary).permute(0, 3, 1, 2))
        Word_set2 = self.bn2(torch.matmul(S2_normalized.permute(0, 2, 3, 1), self.dictionary).permute(0, 3, 1, 2))
        # save_heatmap(Word_set1, f'{prefix}Word_set1.png', output_dir)
        # save_heatmap(Word_set2, f'{prefix}Word_set2.png', output_dir)

        # 上采样
        Word_set1 = self.upsample(Word_set1)
        Word_set2 = self.upsample(Word_set2)
        # save_heatmap(Word_set1, f'{prefix}Word_set1_upsampled.png', output_dir)
        # save_heatmap(Word_set2, f'{prefix}Word_set2_upsampled.png', output_dir)

        # 拼接结果
        c = torch.cat((Word_set1, Word_set2), dim=1)
        # save_heatmap(c, f'{prefix}output_c.png', output_dir)

        torch.cuda.empty_cache()
        return c


    # def dic_learning(self, f, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2):
    #     S = torch.sigmoid(self.convsf2(self.convsf1(f + m)))
    #     S1, S2 = S[:batch_size], S[batch_size:]
    #
    #     # 转换分割图类型
    #     seg1 = seg1.long() if seg1.dtype != torch.int64 else seg1
    #     seg2 = seg2.long() if seg2.dtype != torch.int64 else seg2
    #
    #     S1 = self.Feature_meanlize(S1, seg1,num1, max_num1)
    #     S2 = self.Feature_meanlize(S2, seg2,num2, max_num2)
    #
    #     # 后续处理保持不变
    #     Word_set1 = self.bn1(torch.matmul(S1.permute(0, 2, 3, 1), self.dictionary).permute(0, 3, 1, 2))
    #     Word_set2 = self.bn2(torch.matmul(S2.permute(0, 2, 3, 1), self.dictionary).permute(0, 3, 1, 2))
    #     return torch.cat([self.upsample(Word_set1), self.upsample(Word_set2)], dim=1)

    # def apply_sam_mask(self, image_batch):
    #     """
    #     使用 SAM 生成掩码，并合并所有掩码。
    #     支持批处理图像输入。
    #     """
    #     batch_size = image_batch.shape[0]
    #     weighted_images = []
    #     mask_weights_list = []
    #
    #     for i in range(batch_size):
    #         # 获取单张图像
    #         image = image_batch[i]  # 形状为 (channels, height, width)
    #
    #         # 将图像转换为 SAM 输入格式
    #         self.sam_predictor.set_image(image.permute(1, 2, 0).cpu().numpy())  # 转换为 (H, W, C) 格式
    #
    #         # 获取图像掩码（SAM 生成多个掩码）
    #         masks, scores, _ = self.sam_predictor.predict()
    #
    #         # 合并所有掩码
    #         if len(masks) > 0:
    #             # 方法 1：加权求和（根据置信度 scores 分配权重）
    #             weights = torch.softmax(torch.tensor(scores, device=image.device), dim=0)  # 归一化置信度
    #             combined_mask = torch.sum(torch.tensor(masks, device=image.device) * weights.view(-1, 1, 1), dim=0)
    #
    #             # 方法 2：逻辑 OR 操作（合并所有掩码）
    #             # combined_mask = torch.any(torch.tensor(masks, device=image.device), dim=0).float()
    #
    #             # 将合并后的掩码转换为权重矩阵（通过 softmax）
    #             mask_weights = torch.sigmoid(combined_mask.flatten()).reshape(combined_mask.shape)
    #         else:
    #             # 如果没有生成掩码，直接使用原始图像
    #             mask_weights = torch.ones_like(image[0], device=image.device)  # 全 1 权重矩阵
    #
    #         # 将掩码权重与输入图像相乘
    #         weighted_image = image * mask_weights.unsqueeze(0)
    #
    #         weighted_images.append(weighted_image)
    #         mask_weights_list.append(mask_weights)
    #
    #     # 将结果堆叠为批处理张量
    #     weighted_images = torch.stack(weighted_images, dim=0)  # 形状为 (batch_size, channels, height, width)
    #     mask_weights = torch.stack(mask_weights_list, dim=0)  # 形状为 (batch_size, height, width)
    #
    #     return weighted_images, mask_weights

    import torchvision.transforms as transforms



    # def get_clip_similarity(self, f0):
    #     """
    #     使用 CLIP 计算图像特征与文本特征的相似度
    #     """
    #     tokenized_text = torch.cat([clip.tokenize(f"satellite image of {c}").to(self.device) for c in self.lc_list])
    #     text_features = self.model_clip.encode_text(tokenized_text)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #
    #     # 调整输入图像尺寸
    #     f0 = self.clip_transform(f0)
    #
    #     # 假设 f0 可以转换为图像特征
    #     image_features = self.model_clip.encode_image(f0)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #
    #     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #     # 扩展 similarity 到和 c0 形状相同
    #     similarity = similarity.unsqueeze(1).unsqueeze(2).expand(-1, f0.size(1), f0.size(2), f0.size(3))
    #
    #     return similarity

    # def forward(self, i1, i2, seg1, num1, max_num1, seg2, num2, max_num2,mask1,mask2,prefix):
    def forward(self, i1, i2, seg1, num1, max_num1, seg2, num2, max_num2, mask1, mask2):

        batch_size = i1.shape[0]
        mask1=mask1.unsqueeze(1)
        mask2=mask2.unsqueeze(1)
        # i1_weighted, _ = self.apply_sam_mask(i1)
        # i2_weighted, _ = self.apply_sam_mask(i2)
        i1_weighted = i1 * torch.sigmoid(self.conv1(mask1))
        i2_weighted = i2 * torch.sigmoid(self.conv1(mask2))

        i = torch.cat((i1 + i1_weighted, i2 + i2_weighted), dim=0)
        f0 = self.feature_extractor_deep(i)  # , f1, f2, f3
        # print(f0.shape)
        torch.cuda.empty_cache()
        m = self.relu1(self.fcd1(torch.mean(self.dictionary, dim=1).unsqueeze(0)))
        m = self.fcd2(m).reshape(1, self.channel_num, 1, 1)
        f0 = self.conv_1(f0)


        # c0 = self.dic_learning(f0, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2,prefix)
        c0 = self.dic_learning(f0, m, batch_size, seg1, num1, max_num1, seg2, num2, max_num2)


        # # 使用 CLIP 进行特征提取和相似度计算
        # similarity1 = self.get_clip_similarity(i1_weighted)
        # similarity2 = self.get_clip_similarity(i2_weighted)
        # similarity = abs(similarity2 - similarity1)

        # 这里可以根据相似度对 c0 进行调整
        # 示例：简单地将相似度与 c0 相乘
        c0 = c0

        # c = self.classifier(c0,prefix)
        c = self.classifier(c0)


        return c


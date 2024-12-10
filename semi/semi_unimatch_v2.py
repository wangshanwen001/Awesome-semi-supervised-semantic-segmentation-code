import os

import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss, DiceLoss, softmax_mse_loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr

from utils.utils_metrics import f_score
import numpy as np
from torch.nn import functional as F
from PIL import Image
import random
from torch.nn import Softmax, LayerNorm
from torch import nn
import torchvision.transforms as transforms
from utils.dataloader_unlabel import cutmix_images, SA


def get_adaptive_threshold(epoch, args):
    """计算自适应阈值"""
    if epoch < args['warm_up']:
        return args['threshold_warmup']
    else:
        # 线性增加阈值
        current = min(1., args['threshold'] + (epoch - args['warm_up']) / args['ramp_up'] * 0.2)
        return current


class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def apply_uniperb(images, eps):
    """
    应用UniPerb扰动（注意原文只用了0.5的dropout，这里是测试其他扰动对UniPerd的效果，if遵循原文可以改为dropout）
    Args:
        images: 输入图像
        eps: 扰动大小
    Returns:
        perturbed_images: 添加扰动后的图像
    """
    # 生成均匀分布的随机噪声
    noise = torch.empty_like(images).uniform_(-eps, eps)
    # 添加扰动
    perturbed_images = images + noise
    # 裁剪到合法范围
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images


def complementary_dropout_for_two_inputs(input1, input2, p=0.5, training=True):
    if not training:
        return input1, input2

    # 生成一个随机mask
    mask = torch.bernoulli(torch.ones_like(input1) * (1 - p))

    # input1 使用 mask
    outputs_1 = input1 * mask / (1 - p)

    # input2 使用相反的 mask
    outputs_2 = input2 * (1 - mask) / p

    return outputs_1, outputs_2

def fit_one_epoch(model_train, model, model_train_unlabel, ema_model, loss_history, eval_callback, optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_unlabel, gen_val, Epoch, cuda, dice_loss, focal_loss,
                  cls_weights, num_classes, \
                  fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    suloss_item = 0
    loss_u = 0
    val_loss = 0
    val_f_score = 0
    dropout = nn.Dropout(p=0.5)
    args = {
        'threshold': 0.95,  # 置信度阈值
        'eps_uni': 0.02,  # 置信度阈值
        'threshold_warmup': 0.8,  # 预热阶段的阈值
        'warm_up': 5,  # 预热轮数
        'ramp_up': 50,  # 阈值提升的轮数
        'align_alpha': 0.9,  # 类别分布对齐参数
        'lambda_u': 1.0,  # 无监督损失权重
        'lambda_c': 0.01,  # yicixing shunshi quanzhong
        'temperature': 0.5,
        'print_freq': 50,  # 打印频率
    }
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    dice_loss = DiceLoss(num_classes)
    criterion_u = nn.CrossEntropyLoss().cuda(local_rank)
    for iteration, ((imgs_label, pngs, labels), imgs_unlabel) in enumerate(zip(gen, gen_unlabel)):
        if iteration >= epoch_step:
            break
        imgs_unlabel = imgs_unlabel
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs_unlabel_s = SA(imgs_unlabel, imgs_label)
                imgs_label = imgs_label.cuda(local_rank)
                imgs_unlabel = imgs_unlabel.cuda(local_rank)
                imgs_unlabel_s = imgs_unlabel_s.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if fp16:
            # ----------------------#
            #
            # ----------------------#
            model_train.train()
            model_train_unlabel.train()
            num_lb, num_ulb = imgs_label.shape[0], imgs_unlabel_s.shape[0]
            outputs_total, outputs_encoder, low_level_features = ema_model(torch.cat((imgs_label, imgs_unlabel_s)))
            outputs_label, outputs_unlabel = outputs_total.split([num_lb, num_ulb])
            _, encoder_unlabel = outputs_encoder.split([num_lb, num_ulb])
            _, low_level_features = low_level_features.split([num_lb, num_ulb])
            # ----------------------#
            #   监督学习
            # ----------------------#
            if focal_loss:
                suloss = Focal_Loss(outputs_label, pngs, weights, num_classes=num_classes)
            else:
                suloss = CE_Loss(outputs_label, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs_label, labels)
                suloss = suloss + main_dice

            # ============= 双流扰动的核心实现 =============


            imgs_unlabel_strong1 = SA(imgs_unlabel, imgs_label)
            imgs_unlabel_strong2 = SA(imgs_unlabel, imgs_label)
            imgs_unlabel_strong1 = imgs_unlabel_strong1.cuda(local_rank)
            imgs_unlabel_strong2 = imgs_unlabel_strong2.cuda(local_rank)

            with torch.no_grad():

                # 集成两个弱扰动预测
                pseudo_logits = outputs_unlabel
                pseudo_labels = torch.softmax(pseudo_logits.detach(), dim=1)
                max_probs, targets_u = torch.max(pseudo_labels, dim=1)

                # 自适应阈值
                confidence_threshold = get_adaptive_threshold(epoch, args)
                mask = max_probs.ge(confidence_threshold).float()

            # 强扰动流的预测
            logits_strong1,outputs_encoder_s1, low_level_features_s1 = model_train(imgs_unlabel_strong1)
            logits_strong2,outputs_encoder_s2, low_level_features_s2= model_train(imgs_unlabel_strong2)
            # complementary Dropout
            outputs_encoder_s1, outputs_encoder_s2 = complementary_dropout_for_two_inputs(outputs_encoder_s1, outputs_encoder_s2, p=0.5)
            logits_strong1 = model_train.module.decoder(low_level_features_s1, outputs_encoder_s1)
            logits_strong2 = model_train.module.decoder(low_level_features_s2, outputs_encoder_s2)


            # 计算强扰动流与伪标签之间的一致性损失
            loss_unsup_strong1 = criterion_u(logits_strong1, targets_u)
            loss_unsup_strong2 = criterion_u(logits_strong2, targets_u)

            # strong之间的一致性损失（这是原文没有的，实测加上效果会稍微更好一点点）
            loss_consistency = F.mse_loss(
                torch.softmax(logits_strong1, dim=1),
                torch.softmax(logits_strong2, dim=1)
            )

            # 总的无监督损失
            loss_unsupervised = (
                (loss_unsup_strong1 + loss_unsup_strong2) * mask +
                args['lambda_c'] * loss_consistency
            ).mean()

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss = suloss + args['lambda_u'] * loss_unsupervised
            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()


        else:
            print('else')
        total_loss += loss.item()
        suloss_item += suloss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs, _, _ = model_train(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    print('total Loss: %.6f' % (total_loss / epoch_step))
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('su Loss: %.6f ||u Loss: %.6f || Total Loss: %.6f ||Val Loss: %.3f ' % (
        suloss_item / epoch_step, loss_u / epoch_step, total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

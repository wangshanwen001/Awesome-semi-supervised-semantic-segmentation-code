import os
import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,DiceLoss,softmax_mse_loss,
                                     weights_init)
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics import f_score
from torch.nn import functional as F
from torch import  nn
from utils.dataloader_unlabel import SA

def fit_one_epoch(model_train, model,model_train_unlabel,ema_model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,gen, gen_unlabel,gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    suloss_item=0
    loss_u=0
    val_loss        = 0
    val_f_score     = 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    dice_loss = DiceLoss(num_classes)
    for iteration, ((imgs_label, pngs, labels),imgs_unlabel) in enumerate(zip(gen,gen_unlabel)):
        if iteration >= epoch_step: 
            break
        imgs_unlabel = imgs_unlabel
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs_unlabel_s, lam = cutmix_images(imgs_unlabel, imgs_label, alpha=1.0)
                imgs_label = imgs_label.cuda(local_rank)
                imgs_unlabel    = imgs_unlabel.cuda(local_rank)
                imgs_unlabel_s = imgs_unlabel_s.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

                model_train.eval()
                pred_u_pseudo = model_train(imgs_unlabel)
                pred_u_pseudo = pred_u_pseudo.detach()
                model_train.module.set_pseudo_prob_map(pred_u_pseudo)
                pseudo_label = pred_u_pseudo.argmax(dim=1)
                model_train.module.set_pseudo_label(pseudo_label)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if fp16:
            # ----------------------#
            #
            # ----------------------#
            model_train.train()
            model_train_unlabel.train()
            num_lb, num_ulb = imgs_label.shape[0], imgs_unlabel_s.shape[0]
            outputs_total = model_train(torch.cat((imgs_label, imgs_unlabel_s)))
            outputs_label, outputs_unlabel = outputs_total.split([num_lb, num_ulb])
            #----------------------#
            #   监督学习
            #----------------------#
            if focal_loss:
                suloss = Focal_Loss(outputs_label, pngs, weights, num_classes = num_classes)
            else:
                suloss = CE_Loss(outputs_label, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs_label, labels)
                suloss      = suloss + main_dice

            ema_inputs, _ = SA(imgs_unlabel, imgs_label)
            with torch.no_grad():
                ema_output = model_train_unlabel(ema_inputs)
            consistency_loss = softmax_mse_loss(
                outputs_unlabel, ema_output)
            ##########################
            #----------------------#
            #   计算损失
            #----------------------#
            loss = suloss +consistency_loss
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()


        else:
            print('else')
        total_loss      += loss.item()
        suloss_item  += suloss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
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
            outputs = model_train(imgs)
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
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('su Loss: %.6f ||u Loss: %.6f || Total Loss: %.6f ||Val Loss: %.3f ' % (suloss_item / epoch_step, loss_u / epoch_step, total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
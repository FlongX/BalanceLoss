import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss

ce_loss = CrossEntropyLoss()


def get_tp_fp_fn(predict, target):
    tp = (predict * target).sum()
    fp = (predict * (1 - target)).sum()
    fn = ((1 - predict) * target).sum()
    return tp, fp, fn


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice
        return loss / self.n_classes


dice_loss = DiceLoss(2)


class TopKLoss(nn.Module):
    def __init__(self, k=90):
        super(TopKLoss, self).__init__()
        self.k = k

    def forward(self, probs, gt, img_size, epsilon=1e-7):
        pc = torch.softmax(probs, dim=1)[:, 1, :, :]
        pc = torch.clamp(pc, epsilon, 1.0 - epsilon)

        loss = -(gt * torch.log(pc)) - ((1 - gt) * torch.log(1 - pc))
        loss = loss.view(pc.size(0), -1)
        loss, _ = torch.sort(loss, dim=1, descending=True)
        topkloss = torch.zeros_like(loss)
        k = int(img_size[0] * img_size[1] * self.k / 100)
        topkloss[:, 0:k] = loss[:, 0:k]
        return topkloss.mean()


class TTopKLoss(nn.Module):
    def __init__(self):
        super(TTopKLoss, self).__init__()

    def forward(self, probs, gt, epsilon=1e-7, device=0, q=0.9):
        pc = torch.softmax(probs, dim=1)[:, 1, :, :]
        pc = torch.clamp(pc, epsilon, 1.0 - epsilon)

        loss = (-gt * torch.log(pc) - (1 - gt) * torch.log(1 - pc))

        difficult_f_loss = torch.where((gt * pc <= q) & (gt * pc != 0), loss, 0)
        difficult_b_loss = torch.where(((1 - gt) * (1 - pc) <= q) & ((1 - gt) * (1 - pc) != 0), loss, 0)
        difficult_loss = difficult_f_loss + difficult_b_loss

        difficult_num = (torch.count_nonzero(difficult_loss))
        difficult_loss = difficult_loss.sum() / (torch.tensor(difficult_num).cuda(device=device) + 1e-16)

        loss = difficult_loss
        print(
            'TtopK:', round(difficult_loss.item(), 2),
        )
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, n_class=2, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.n_classes = n_class
        self.reduction = reduction

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, predict, target):
        pt = torch.softmax(predict, dim=1)
        target = self._one_hot_encoder(target)
        ce_loss = F.binary_cross_entropy_with_logits(pt, target, reduction="none")
        p_t = predict * target + (1 - predict) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(TverskyLoss, self).__init__()
        self.alpha = 0.3
        self.beta = 1 - self.alpha
        self.n_classes = n_classes

    def tverskyloss(self, predict, target):
        target = target.float()
        smooth = 1
        tp, fp, fn = get_tp_fp_fn(predict, target)
        loss = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        loss = 1 - loss
        return loss

    def forward(self, predict, target):
        predict = torch.softmax(predict, dim=1)[:, 1, :, :]
        loss = self.tverskyloss(predict, target)

        return loss


class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()

    def forward(self, probs, gt, alpha=0.5, beta=0.5, epsilon=1e-7):
        pc = torch.softmax(probs, dim=1)[:, 1, :, :]
        pc = torch.clamp(pc, epsilon, 1.0 - epsilon)

        f_loss = -(gt * torch.log(pc))
        b_loss = -((1 - gt) * torch.log(1 - pc))

        cross_entropy = beta * f_loss + (1 - beta) * b_loss
        cross_entropy = cross_entropy.mean()

        tp, fp, fn = get_tp_fp_fn(pc, gt)
        dice = (tp + epsilon) / (tp + 0.5 * fp + 0.5 * fn + epsilon)
        loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        return loss


class AsymmetricFocalLoss(nn.Module):
    '''
    This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    '''

    def __init__(self, weight=None, size_average=True):
        super(AsymmetricFocalLoss, self).__init__()

    def forward(self, predict, target, delta=0.7, gamma=2, epsilon=1e-7):
        '''
        y_pred : the shape should be [batch,1,H,W], and the input should be the logits by a sigmoid in the forward function.
        y_true : the shape should be [batch,1,H,W].
        '''
        predict = torch.clamp(predict, epsilon, 1.0 - epsilon)
        back_ce = torch.pow(predict, gamma) * -((1 - target) * torch.log(1 - predict))
        back_ce = (1 - delta) * back_ce
        fore_ce = -target * torch.log(predict)
        fore_ce = delta * fore_ce
        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=-1), dim=-1))

        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    '''
    This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    '''

    def __init__(self, weight=None, size_average=True):
        super(AsymmetricFocalTverskyLoss, self).__init__()

    def forward(self, predict: torch.Tensor, target: torch.Tensor, delta=0.7, gamma=0.75, epsilon=1e-7):
        '''
        y_pred : the shape should be [batch,1,H,W], and the input should be the original logits by a sigmoid in the forward function.
        y_true : the shape should be [batch,1,H,W].
        '''

        predict = torch.clamp(predict, epsilon, 1.0 - epsilon)
        tp_f, fp_f, fn_f = get_tp_fp_fn(predict, target)
        tp_b, fp_b, fn_b = get_tp_fp_fn(1 - predict, 1 - target)
        dice_f = (tp_f + epsilon) / (tp_f + delta * fn_f + (1 - delta) * fp_f + epsilon)
        dice_b = (tp_b + epsilon) / (tp_b + delta * fn_b + (1 - delta) * fp_b + epsilon)

        back_dice = 1 - dice_b
        fore_dice = (1 - dice_f) * torch.pow(1 - dice_f, -gamma)
        loss = torch.mean(torch.stack([back_dice, fore_dice], dim=-1))
        return loss


class AsymmetricUnifiedFocalLoss(nn.Module):
    '''
    This is the implementation for binary segmentation.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    '''

    def __init__(self, weight: float = 0.5, gamma: float = 0.5, delta: float = 0.6, reduction='mean', ):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.gamma = gamma
        self.delta = delta
        self.weight: float = weight
        self.reduction = reduction
        self.asy_focal_loss = AsymmetricFocalLoss()
        self.asy_focal_tversky_loss = AsymmetricFocalTverskyLoss()

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        y_pred : the shape should be [batch,1,H,W], and the input should be the logits by a sigmoid in the forward function.
        y_true : the shape should be [batch,1,H,W].
        '''
        predict = torch.softmax(predict, dim=1)[:, 1, :, :]
        asy_focal_loss = self.asy_focal_loss(predict, target, delta=self.delta, gamma=self.gamma)
        asy_focal_tversky_loss = self.asy_focal_tversky_loss(predict, target, delta=self.delta, gamma=self.gamma)

        loss: torch.Tensor = self.weight * asy_focal_loss + (1 - self.weight) * asy_focal_tversky_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyWithL1(nn.Module):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """

    def forward(self, probs, gt_f_num, gt, epsilon=1e-10, alpha=1):
        # ce term
        loss_ce = ce_loss(probs, gt[:].long())
        pc = torch.softmax(probs, dim=1)[:, 1, :, :]
        pc = torch.clamp(pc, epsilon, 1.0 - epsilon)

        f_p = gt * pc
        b_p = (1 - gt) * (1 - pc)
        # regularization
        n = gt.shape[1] * gt.shape[2]
        gt_f_proportion = gt_f_num / n
        gt_b_proportion = (n - gt_f_num) / n
        pred_f_proportion = f_p.sum(dim=(1, 2)) / n
        pred_b_proportion = b_p.sum(dim=(1, 2)) / n
        loss_reg = ((pred_f_proportion - gt_f_proportion).abs() + (pred_b_proportion - gt_b_proportion).abs()).mean()

        loss = loss_ce + alpha * loss_reg
        print(loss_ce.item(), loss_reg.item(), loss.item())
        return loss


class InterCBLoss(nn.Module):
    def __init__(self):
        super(InterCBLoss, self).__init__()

    def forward(self, probs, gt_f_num, gt, img_size, epsilon=1e-7, device=0):
        pc = torch.softmax(probs, dim=1)[:, 1, :, :]
        pc = torch.clamp(pc, epsilon, 1.0 - epsilon)

        f_loss = -(gt * torch.log(pc)).view(pc.size(0), -1)
        b_loss = -((1 - gt) * torch.log(1 - pc)).view(pc.size(0), -1)

        difficult_b_loss = torch.zeros_like(b_loss)
        simple_b_loss = torch.zeros_like(b_loss)

        difficult_b_num = []
        simple_b_num = []
        _, idx = torch.sort(b_loss, dim=1, descending=True)

        for i in range(len(gt_f_num)):
            difficult_b_loss[i, 0: int(gt_f_num[i])] = b_loss[i, idx[i, 0: int(gt_f_num[i])]]
            simple_b_loss[i, int(gt_f_num[i]):] = b_loss[i, idx[i, int(gt_f_num[i]):]]
            difficult_b_num.append(torch.count_nonzero(difficult_b_loss[i]))
            simple_b_num.append(img_size[0] * img_size[1])
        #
        difficult_b_loss = difficult_b_loss.sum(dim=1) / (torch.tensor(difficult_b_num).cuda(device=device) + 1e-16)
        simple_b_loss = simple_b_loss.sum(dim=1) / (torch.tensor(simple_b_num).cuda(device=device) + 1e-16)
        f_loss = f_loss.sum(dim=1) / (gt_f_num + 1e-16)

        loss = (difficult_b_loss.mean() + f_loss.mean() + simple_b_loss.mean())

        print(
            'InterCB:', round(f_loss.mean().item(), 2), round(difficult_b_loss.mean().item(), 2),
        )
        return loss


class IntraCBLoss(nn.Module):
    def __init__(self):
        super(IntraCBLoss, self).__init__()

    def forward(self, probs, gt_f_num, gt, bs, trans=False, epsilon=1e-7, device=0, t=0.9):
        pc = torch.softmax(probs, dim=1)[:, 1, :, :]
        pc = torch.clamp(pc, epsilon, 1.0 - epsilon)

        loss = -(gt * torch.log(pc)) - ((1 - gt) * torch.log(1 - pc))

        difficult_b_loss = torch.zeros_like(loss)
        simple_b_loss = torch.zeros_like(loss)
        difficult_f_loss = torch.zeros_like(loss)
        simple_f_loss = torch.zeros_like(loss)
        difficult_b_num = []
        difficult_f_num = []
        simple_b_num = []
        simple_f_num = []

        for i in range(len(gt_f_num)):
            difficult_f_loss[i] = torch.where((gt[i] * pc[i] <= t) & (gt[i] * pc[i] != 0), loss[i], 0)
            difficult_b_loss[i] = torch.where(((1 - gt[i]) * (1 - pc[i]) <= t) & ((1 - gt[i]) * (1 - pc[i]) != 0),
                                              loss[i], 0)

            simple_f_loss[i] = torch.where((gt[i] * pc[i] > t) & (gt[i] * pc[i] != 0), loss[i], 0)
            simple_b_loss[i] = torch.where(((1 - gt[i]) * (1 - pc[i]) > t) & ((1 - gt[i]) * (1 - pc[i]) != 0), loss[i],
                                           0)

            difficult_b_num.append(torch.count_nonzero(difficult_b_loss[i]))
            difficult_f_num.append(torch.count_nonzero(difficult_f_loss[i]))

            simple_b_num.append(torch.count_nonzero(simple_b_loss[i]))
            simple_f_num.append(torch.count_nonzero(simple_f_loss[i]))

        difficult_b_loss = difficult_b_loss.sum(dim=(1, 2)) / (
                    torch.tensor(difficult_b_num).cuda(device=device) + 1e-16)
        difficult_f_loss = difficult_f_loss.sum(dim=(1, 2)) / (
                    torch.tensor(difficult_f_num).cuda(device=device) + 1e-16)
        simple_b_loss = simple_b_loss.sum(dim=(1, 2)) / (torch.tensor(simple_b_num).cuda(device=device) + 1e-16)
        simple_f_loss = simple_f_loss.sum(dim=(1, 2)) / (torch.tensor(simple_f_num).cuda(device=device) + 1e-16)

        loss = (difficult_b_loss.mean() + difficult_f_loss.mean() + simple_f_loss.mean() + simple_b_loss.mean())
        print(
            'IntraCB:', round(difficult_f_loss.mean().item(), 2), round(difficult_b_loss.mean().item(), 2),
        )

        pc_f = (gt * (1 - pc))
        pc_b = ((1 - gt) * pc)
        pc_f, _ = torch.sort(pc_f.view(pc.size(0), -1), dim=1, descending=True)
        pc_b, _ = torch.sort(pc_b.view(pc.size(0), -1), dim=1, descending=True)

        if not trans:
            sum_b = 0
            sum_f = 0
            for i in range(len(gt_f_num)):
                if pc_b[i, int(gt_f_num[i] - 1)] <= (1 - t):
                    sum_b += 1
                if pc_f[i, int(gt_f_num[i] - 1)] <= (1 - t):
                    sum_f += 1
            print(sum_b, sum_f)
            if sum_b >= len(gt_f_num) / 2 and sum_f >= len(gt_f_num) / 2 and len(gt_f_num) == bs:
                trans = True

        return loss, trans


def show_pc(pc, gt, gt_f_num, epsilon=1e-7):
    with torch.no_grad():
        pc = torch.softmax(pc, dim=1)[:, 1, :, :]
        pc = torch.clamp(pc, epsilon, 1.0 - epsilon)
        pc_f = (gt * (1 - pc))[0, :, :].view(1, -1)
        pc_b = ((1 - gt) * pc)[0, :, :].view(1, -1)
        pc_f, _ = torch.sort(pc_f, descending=True)
        pc_b, _ = torch.sort(pc_b, descending=True)
        pc_b_a = pc_b[0, : int(gt_f_num[0])]
    return pc_f, pc_b, pc_b_a

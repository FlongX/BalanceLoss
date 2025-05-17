import sys
import torch
import logging
import numpy as np
import pandas as pd
from medpy import metric
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def list2csv(list_, column_name, index_name, csv_save_path):
    column = [column_name]  # 列表头名称
    data = pd.DataFrame(columns=column, data=list_)  # 将数据放进表格
    data.index.name = index_name
    data.to_csv(csv_save_path)  # 数据存入csv,存储位置及文件名称


def show_reward_curve(csv_path, png_save_path, xlab, ylab, Smoothing):
    df = pd.read_csv(csv_path)

    # 设置图片大小
    plt.figure(figsize=(20, 12))

    # 设置背景网格
    plt.grid(linestyle='-.', linewidth=1.5, zorder=0)  # zorder控制绘图顺序，值越大绘图越晚

    # 设置x, y轴标签
    plt.xlabel(xlab, fontproperties='Times New Roman', fontweight='bold', fontsize=35, labelpad=15)
    plt.ylabel(ylab, fontweight='bold', fontproperties='Times New Roman', fontsize=35, labelpad=15)

    # 设置坐标轴刻度值
    plt.xticks(fontproperties='Times New Roman', size=30)
    plt.yticks(fontproperties='Times New Roman', size=30)

    # 局部多项式拟合
    spline = UnivariateSpline(df.iloc[:, 0], df.iloc[:, 1], s=Smoothing)  # 调整参数 s 控制平滑度
    fit_epochs = np.linspace(1, len(df.iloc[:, 0]), 2 * len(df.iloc[:, 0]))  # 更密集的点来绘制曲线
    fit_loss = spline(fit_epochs)

    # 绘制
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], linestyle='dashdot', linewidth=2, marker='o', ms=12, alpha=0.3)
    plt.plot(fit_epochs, fit_loss, linestyle='-', linewidth=6, ms=12, color='#fa5050')

    # 保存和展示
    plt.savefig(png_save_path, dpi=300)
    plt.close()


def gard(probs, gt_f_num, gt_b_num, gt):
    with torch.no_grad():
        pc = torch.softmax(probs, dim=1)[:, 1, :, :]
        pc = pc.type(torch.float32)
        gard = pc
        gard_b = gard * (1 - gt)
        gard_bfp = gard * (1 - gt)
        gard_f = (1 - gard) * gt
        gard_b = gard_b.sum(dim=(1, 2)) / (gt_b_num + 1e-16)
        gard_bfp = gard_bfp.sum(dim=(1, 2)) / (gt_b_num - (gt_b_num - gt_f_num) + 1e-16)
        gard_f = gard_f.sum(dim=(1, 2)) / (gt_f_num + 1e-16)

        return gard_b.mean().cpu().detach().numpy(), gard_bfp.mean().cpu().detach().numpy(), gard_f.mean().cpu().detach().numpy()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, device):
    label = label.squeeze(0)
    label = label.numpy()
    image = image.cuda(device=device)

    net.eval()
    with torch.no_grad():
        out = net(image)
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        out = torch.tensor(out)
        out = out.cpu().detach().numpy()

    metric_list = calculate_metric_percase(out, label)
    return metric_list, out


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%m%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def pc_image(pc_f, pc_b, pc_b_a, idx):
    pc_f = pc_f.cpu().numpy().tolist()[0]
    pc_f = list(filter(lambda x: x > 0, pc_f))
    pc_f_x = []
    pc_b_x = []
    pc_b_a_x = []
    pc_b = pc_b.cpu().numpy().tolist()[0]
    pc_b = list(filter(lambda x: x > 0, pc_b))
    pc_b_a = pc_b_a.cpu().numpy().tolist()
    pc_b_a = list(filter(lambda x: x > 0, pc_b_a))
    for i in range(len(pc_f)):
        pc_f_x.append(i)
    for i in range(len(pc_b)):
        pc_b_x.append(i)
    for i in range(len(pc_b_a)):
        pc_b_a_x.append(i)
    # 设置图片大小
    plt.figure(figsize=(20, 12))

    # 设置背景网格
    plt.grid(linestyle='-.', linewidth=1.5, zorder=0)  # zorder控制绘图顺序，值越大绘图越晚

    # 设置x, y轴标签
    plt.xlabel('num', fontproperties='Times New Roman', fontweight='bold', fontsize=35, labelpad=15)
    plt.ylabel('pc', fontweight='bold', fontproperties='Times New Roman', fontsize=35, labelpad=15)

    # 设置坐标轴刻度值
    plt.xticks(fontproperties='Times New Roman', size=30)
    plt.yticks(fontproperties='Times New Roman', size=30)

    # 绘制
    plt.plot(pc_f_x, pc_f, linestyle='dashdot', linewidth=1, marker='o', ms=5, color='r')
    plt.savefig(
        r'./model/pc/' + str(idx / 50) + 'f.png',
        dpi=300)
    plt.plot(pc_b_a_x, pc_b_a, linestyle='dashdot', linewidth=1, marker='o', ms=5, color='g')
    plt.savefig(
        r'./model/pc/' + str(idx / 50) + 'b_a.png',
        dpi=300)
    plt.plot(pc_b_x, pc_b, linestyle='dashdot', linewidth=1, marker='o', ms=5, color='b')
    # 保存和展示
    plt.savefig(r'./model/pc/' + str(int(idx / 50)) + '.png', dpi=300)
    plt.close()
    return 0


class EarlyStopping:
    def __init__(self, patience=20, patience_lr=10, patience_t=2):
        self.patience = patience
        self.patience_lr = patience_lr
        self.counter = 0
        self.counter_lr = 0
        self.counter_t = 0
        self.best_dice = 0

    def EStop(self, val_dice):
        decay = 1
        early_stop = False
        save_model = False
        if val_dice > self.best_dice:
            self.best_dice = val_dice
            self.counter = 0
            self.counter_lr = 0

            save_model = True
        else:
            self.counter += 1
            self.counter_lr += 1
            if self.counter_lr >= self.patience_lr:
                decay = 0.5
                self.counter_lr = 0
            if self.counter >= self.patience:
                early_stop = True
        return decay, early_stop, save_model


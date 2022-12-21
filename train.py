import argparse
import logging.config
from utils.load_conf import ConfigLoader
from pathlib import Path

import os
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net, eval_3d

from model.ourNet import ourNet5

from utils.dataset import KneeDataset, DriveDataset, StareDataset, ChasedDataset, LungDataset, GLASDataset
from torch.utils.data import DataLoader

logger_path = Path("./configs/logger.yaml")
conf = ConfigLoader(logger_path)
_logger = logging.getLogger(__name__)

dir_checkpoint = './checkpoint/'
device = torch.device(1)


def get_args():
    parser = argparse.ArgumentParser(description='train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=3000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    # 选择对应的数据集 Knee、DRIVE、STARE、CHASEDB1、Lung、GLAS
    parser.add_argument('-d', '--data_path', type=str, default='./data/STARE/', help='Load data')
    parser.add_argument('--channels', type=int, default=3, help='image channels', dest='channels')
    parser.add_argument('--classes', type=int, default=1, help='masks_2st nums', dest='classes')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,  # 0.0001
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-m', '--model', type=str, default="ourNet5", help='Load model')
    parser.add_argument('-s', '--save_cp', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-scal', '--scal', type=bool, default=False,
                        help='test for scal or not')

    return parser.parse_args()


def save_pth(model, name, epoch):
    torch.save(net.state_dict(), dir_checkpoint + '{}_{}_{}.pth'.format(model, name, epoch))


def delete_pth(model, name, epoch):
    path = dir_checkpoint + '{}_{}_{}.pth'.format(model, name, epoch)
    if (os.path.exists(path)):
        os.remove(path)


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_net(net, device, root_dir, epochs=100, batch_size=1, lr=0.001, model='U-Net', save_cp=True, scal=False):
    dataname = root_dir.split('/')[2]

    if dataname == "Knee":
        train_data = KneeDataset(root_dir, train=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        n_train = int(len(train_loader) * batch_size)
        n_test = 20

    if dataname == "DRIVE":
        train_data = DriveDataset(root_dir, train=True)
        test_data = DriveDataset(root_dir, train=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
        n_train = int(len(train_loader) * batch_size)
        n_test = int(len(test_loader) * batch_size)

    if dataname == "CHASEDB1":
        train_data = ChasedDataset(root_dir, train=True)
        test_data = ChasedDataset(root_dir, train=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
        n_train = int(len(train_loader) * batch_size)
        n_test = int(len(test_loader) * batch_size)

    if dataname == "STARE":
        train_data = StareDataset(root_dir, train=True)
        test_data = StareDataset(root_dir, train=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
        n_train = int(len(train_loader) * batch_size)
        n_test = int(len(test_loader) * batch_size)

    if dataname == "Lung":
        train_data = LungDataset(root_dir, train=True)
        test_data = LungDataset(root_dir, train=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
        n_train = int(len(train_loader) * batch_size)
        n_test = int(len(test_loader) * batch_size)

    if dataname == "GLAS":
        train_data = GLASDataset(root_dir, train=True)
        test_data = GLASDataset(root_dir, train=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
        n_train = int(len(train_loader) * batch_size)
        n_test = int(len(test_loader) * batch_size)
    _logger.info('train/test number: {}/{}'.format(n_train, n_test))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)

    # TODO: Dice loss/IOU loss?
    if net.n_classes > 1:
        # 使用普通交叉熵
        criterion = nn.CrossEntropyLoss()
    else:
        # 二分类使用二值交叉熵
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.MSELoss()
        # criterion = nn.BCELoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['masks_2st']
                # imgs 形状应为 [BatchSize, Channel, Height, Width]
                assert imgs.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # 梯度值清零
                optimizer.zero_grad()

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                pred = net(imgs)
                loss = criterion(pred, true_masks)
                # loss = dice_coeff_loss(pred, true_masks, device)
                # loss = dice_coeff_loss(pred, true_masks, device) + criterion(pred, true_masks)
                epoch_loss += loss.item()
                # 反向传播计算梯度值
                loss.backward()
                # 防止梯度消失和梯度爆炸
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # 通过梯度下降更新参数
                optimizer.step()
                # 进度条右边显示内容
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
        epoch_loss = epoch_loss / len(train_loader)
        adjust_lr(optimizer, base_lr=lr, iter=epoch, max_iter=epochs, power=0.9)
        _logger.info('Epoch_{}_loss: {}'.format(epoch + 1, epoch_loss))

        # 验证集评估模型
        if dataname == "Knee":
            dice, hd95 = eval_3d(net, device)
            _logger.info(f'''Starting testing:
                                            Validation Dice:               {dice}
                                            Validation Hd95:               {hd95}
                                            ''')
        else:
            f1, acc, sen, auc, dice, hd95 = eval_net(net, test_loader, batch_size, device, dataname, scal=scal)

            _logger.info(f'''Starting testing:
                                Validation F1:                 {f1}
                                Validation acc:                {acc}
                                Validation sen:                {sen}
                                Validation auc:                {auc}
                                Validation dice:               {dice}
                                Validation Hd95:               {hd95}
                                ''')
        if (epoch + 1) % 10 == 0:
            if save_cp:
                try:
                    if os.path.exists(dir_checkpoint):
                        torch.save(net.state_dict(),
                                   dir_checkpoint + f'CP_epoch{epoch + 1}_{model}.pth')
                        _logger.info(f'Checkpoint {epoch + 1}_{model} saved !')
                    else:
                        os.mkdir(dir_checkpoint)
                        _logger.info('Created checkpoint directory')
                        torch.save(net.state_dict(),
                                   dir_checkpoint + f'CP_epoch{epoch + 1}_{model}_{dataname}.pth')
                        _logger.info(f'Checkpoint {epoch + 1}_{model}_{dataname} saved !')
                except OSError:
                    _logger.error('Failed to created checkpoint directory!')


if __name__ == '__main__':
    args = get_args()
    remark = ''
    is_scal = False
    start = time.perf_counter()
    _logger.info(f'''Starting training:
                                epochs:                        {args.epochs}
                                Batch size:                    {args.batchsize}
                                Learning rate:                 {args.lr}
                                Checkpoints:                   {args.save_cp}
                                input channels:                {args.channels}
                                output channels (classes):     {args.classes}
                                model:                         {args.model}
                                DataSet:                       {args.data_path.split('/')[2]}
                                Scal:                          {is_scal}
                                Device:                        {device.type}
                                Remarks:                       {remark}
                            ''')

    net = ourNet5(in_channels=args.channels, n_classes=args.classes)

    # 是否加载预训练模型
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        _logger.info(f'Model loaded from {args.load}')

    net.to(device=device)

    train_net(net=net,
              epochs=args.epochs,
              root_dir=args.data_path,
              batch_size=args.batchsize,
              lr=args.lr,
              model=args.model,
              save_cp=args.save_cp,
              device=device,
              scal=is_scal)

    torch.save(net.state_dict(), './{}_last.pth'.format(args.model))
    _logger.info('Train End!')
    end = time.perf_counter()
    runTime = round(((end - start) / 3600), 2)
    print(f"Time:{runTime}/n")

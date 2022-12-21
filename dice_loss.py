import cv2
import numpy as np
import nibabel as nib
import torch
from medpy.metric import binary
import sklearn.metrics as AUC


def predict_img(net,
                img,
                device,
                out_threshold=0.5):
    img = torch.from_numpy(img).to(device).type(torch.float32)
    net.eval()
    with torch.no_grad():
        output = net(img)
        # output.shape: [1,1,512,512]
        output = (output > out_threshold).float()
    output = torch.squeeze(output, dim=0)
    # output.shape: [1,512,512]
    output = torch.squeeze(output, dim=0)
    # output.shape: [512,512]
    return output


def normalization(data):  # 归一化
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def numeric_score(pred, gt):
    FP = np.float(np.sum((pred == 1) & (gt == 0)))
    FN = np.float(np.sum((pred == 0) & (gt == 1)))
    TP = np.float(np.sum((pred == 1) & (gt == 1)))
    TN = np.float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def get_AUC(prediction, label):
    result_1D = prediction.flatten()
    label_1D = label.flatten()

    auc = AUC.roc_auc_score(label_1D, result_1D)
    return auc


def metrics(pred, label, batch_size, dataset, scal=False):
    AUC_pred = pred
    outputs_auc = (AUC_pred.data.cpu().numpy()).astype(np.float32)
    outputs_auc = outputs_auc.squeeze(1)

    one_pred = (pred > 0.5).float()
    outputs = (one_pred.data.cpu().numpy()).astype(np.uint8)
    labels = (label.data.cpu().numpy()).astype(np.uint8)

    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)

    F1, Acc, Sen, AUC, Dice, Hd95 = 0., 0., 0., 0., 0., 0.
    for i in range(batch_size):
        img = outputs[i, :, :]
        gt = labels[i, :, :]
        auc_pred = outputs_auc[i, :, :]

        if scal:
            if dataset == 'DRIVE':
                img = cv2.resize(img, (565, 584))
                auc_pred = cv2.resize(auc_pred, (565, 584))

            if dataset == 'STARE':
                img = cv2.resize(img, (700, 605))
                auc_pred = cv2.resize(auc_pred, (700, 605))

            if dataset == 'CHASEDB1':
                img = cv2.resize(img, (999, 960))
                auc_pred = cv2.resize(auc_pred, (999, 960))

            img[img >= 0.5] = 1
            img[img < 0.5] = 0

        f1, acc, sen, dic, hd = get_acc(img, gt)
        auc = get_AUC(auc_pred, gt)
        F1 += f1
        Acc += acc
        Sen += sen
        AUC += auc
        Dice += dic
        Hd95 += hd

    return F1 / batch_size, Acc / batch_size, Sen / batch_size, AUC / batch_size, Dice / batch_size, Hd95 / batch_size


def get_acc(image, label):
    FP, FN, TP, TN = numeric_score(image, label)
    num = 0.0000001
    Hd = 0
    Pre = TP / (TP + FP + num)  # 精确度Precision
    acc = (TP + TN) / (TP + TN + FP + FN + num)  # 准确率Accuracy
    sen = TP / (TP + FN + num)  # 灵敏度Sensitive  or  召回率Recall
    F1 = 2 * Pre * sen / ((Pre + sen) + num)  # F1_score
    Dice = (2 * TP) / (2 * TP + FP + FN + num)
    if np.any(label):
        if np.any(image):
            Hd = binary.hd95(image, label)

    return F1, acc, sen, Dice, Hd


def get3d_acc(image, label):
    FP, FN, TP, TN = numeric_score(image, label)
    num = 0.0000001
    dice = (2 * TP) / (2 * TP + FP + FN + num)
    return dice


def metrics_3d(net, test_nii_path, mask_nii_path, device):
    Dice = 0.
    Hd95 = 0.
    image_nii = nib.load(test_nii_path)  # 为了得到affine信息
    image = image_nii.get_fdata()

    mask_nii = nib.load(mask_nii_path)
    mask = mask_nii.get_fdata()

    i_height, i_width, i_depth = image.shape

    d_depth = i_depth
    h_depth = i_depth
    image = np.array(image)
    mask = np.array(mask)
    for i in range(i_depth):
        image[:, :, i][image[:, :, i] < 100] = 0
        image[:, :, i] = normalization(image[:, :, i])
        test_image = image[:, :, i][np.newaxis, :][np.newaxis, :]
        label = mask[:, :, i]
        pred = predict_img(net=net,
                           img=test_image,
                           out_threshold=0.5,
                           device=device)
        pred = pred.cpu().numpy()
        if np.float(np.sum((label == 0) & (pred == 0))) == 262144:
            d_depth = d_depth - 1
        else:
            dice = get3d_acc(pred, label)
            Dice += dice

        if np.any(label):
            if np.any(pred):
                hd95 = binary.hd95(pred, label)
                Hd95 += hd95
        else:
            h_depth = h_depth - 1
    return Dice / d_depth, Hd95 / h_depth

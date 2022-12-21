import os

import cv2

import numpy as np

import torch
from PIL import Image

from model.ourNet import ourNet5


def predict_img(net,
                img,
                device,
                out_threshold=0.5):
    img = torch.from_numpy(img).to(device).type(torch.float32)
    net.eval()
    with torch.no_grad():
        output = net(img)
        # output.shape: [1,1,512,512]
        output = (output >= out_threshold).float()
    output = torch.squeeze(output, dim=0)
    # output.shape: [1,512,512]
    output = torch.squeeze(output, dim=0)

    # output.shape: [512,512]
    return output


if __name__ == "__main__":
    dataset = 'GLAS'

    if dataset == 'DRIVE':
        resize_scal = (512, 512)
    if dataset == 'STARE':
        resize_scal = (512, 512)
    if dataset == 'CHASEDB1':
        resize_scal = (960, 960)
    if dataset == 'GLAS':
        resize_scal = (512, 512)
    img_file_path = 'data/' + dataset + '/test/images/'
    save_path = 'test_2D_results/predict/' + dataset + '/'
    weight_path = './checkpoint/ourNet4_f1_1177.pth'

    for file in os.listdir(img_file_path):
        device = torch.device(1)

        model = 'ourNet4'  # U_Net、AttU_Net、ourNet1、EUNet

        img_path = img_file_path + file
        image = np.array(Image.open(img_path))
        # image = cv2.imread(img_path)

        w = image.shape[0]
        h = image.shape[1]

        image = cv2.resize(image, resize_scal)

        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1)) / 255.0  # (3, 512, 512)

        image = np.expand_dims(image, axis=0)

        net = ourNet5(n_classes=1, in_channels=3)

        net.to(device=device)
        net.load_state_dict(torch.load(weight_path, map_location=device))

        output = predict_img(net, image, device)
        output = output.cpu().numpy()

        output_resize = cv2.resize(output, (h, w))
        save = save_path + file.split('.')[0] + model + '.tiff'
        # save = save_path + file.split('.')[0] + 'TransUnet' + '.tiff'

        output_resize[output_resize > 0.5] = 1
        output_resize[output_resize < 0.5] = 0
        cv2.imwrite(save, output_resize * 255)
        print(output_resize.shape)
        # save = save_path + file.split('.')[0] + '.tiff'
        # cv2.imwrite(save, output_resize * 255)

        print(f"{file} :已完成！")

import random
from os.path import splitext
from os import listdir
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import os
import logging

from torchvision import transforms

_logger = logging.getLogger(__name__)


def RandomCrop(image, label, crop_size):
    crop_width, crop_height = crop_size

    w, h = label.size

    left = random.randint(0, w - crop_width)
    top = random.randint(0, h - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    new_image = image.crop((left, top, right, bottom))
    new_label = label.crop((left, top, right, bottom))
    return new_image, new_label


class KneeDataset(Dataset):
    def __init__(self, root_dir, train=False, mask='_mask'):
        if train:
            path = 'train'
        else:
            path = 'test'
        self.imgs_dir = os.path.join(root_dir, path, 'images', '')
        self.masks_dir = os.path.join(root_dir, path, 'masks', '')
        self.mask = mask
        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = os.path.join(self.masks_dir, '{}'.format(idx) + self.mask + '.npy')
        img_file = os.path.join(self.imgs_dir, '{}'.format(idx) + '.npy')
        img = np.load(img_file, allow_pickle=True)[np.newaxis, :]
        # img = np.concatenate((img, img, img), axis=0)

        mask = np.load(mask_file, allow_pickle=True)[np.newaxis, :]
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'masks_2st': torch.from_numpy(mask).type(torch.FloatTensor)
        }


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


# 随机仿射变换
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


# 随机水平翻转
def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


# 随机垂直翻转
def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def read_DRIVE_datasets(root_path, train=True):
    images = []
    masks = []
    if train:
        image_root = os.path.join(root_path, 'train/images')
        gt_root = os.path.join(root_path, 'train/masks')
    else:
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/masks')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0].split('.')[0] + '_mask.gif')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def default_DRIVE_loader(img_path, mask_path, scal=(512, 512)):
    is_train = img_path.split('/')[3]

    if is_train == "train":
        img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path)
        img = cv2.resize(img, scal)

        mask = np.array(Image.open(mask_path))
        mask = cv2.resize(mask, scal)
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if is_train == "test":
        img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path)

        img = cv2.resize(img, scal)

        mask = np.array(Image.open(mask_path))
        mask = cv2.resize(mask, scal)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    # masks_2st = abs(masks_2st-1)
    return img, mask


def default_GLAS_loader(img_path, mask_path, scal=(512, 512)):
    is_train = img_path.split('/')[3]

    if is_train == "train":
        img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path)
        img = cv2.resize(img, scal)

        mask = np.array(Image.open(mask_path))
        mask = cv2.resize(mask, scal)
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if is_train == "test":
        img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path)

        img = cv2.resize(img, scal)

        mask = np.array(Image.open(mask_path))
        mask = cv2.resize(mask, scal)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    # masks_2st = abs(masks_2st-1)
    return img, mask


def default_STARE_loader(img_path, mask_path, scal=(512, 512)):
    is_train = img_path.split('/')[3]

    if is_train == "train":
        img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path)
        img = cv2.resize(img, scal)

        mask = np.array(Image.open(mask_path))
        mask = cv2.resize(mask, scal)
        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if is_train == "test":
        img = np.array(Image.open(img_path))
        # img = cv2.imread(img_path)

        img = cv2.resize(img, scal)

        mask = np.array(Image.open(mask_path))
        mask = cv2.resize(mask, scal)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    # masks_2st = abs(masks_2st-1)
    return img, mask


def default_scalDRIVE_loader(img_path, mask_path, crop_scal=(512, 512), resize_scal=(512, 512)):
    is_train = img_path.split('/')[3]

    if is_train == "train":
        image = Image.open(img_path)
        label = Image.open(mask_path)

        image, label = RandomCrop(image, label, crop_scal)

        img = np.array(image)
        mask = np.array(label)

        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if is_train == "test":
        img = np.array(Image.open(img_path))
        img = cv2.resize(img, resize_scal)

        mask = np.array(Image.open(mask_path))

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    # masks_2st = abs(masks_2st-1)
    return img, mask


def default_scalSTARE_loader(img_path, mask_path, crop_scal=(512, 512), resize_scal=(592, 592)):
    is_train = img_path.split('/')[3]

    if is_train == "train":
        image = Image.open(img_path)
        label = Image.open(mask_path)

        image, label = RandomCrop(image, label, crop_scal)

        img = np.array(image)
        mask = np.array(label)

        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if is_train == "test":
        img = np.array(Image.open(img_path))
        img = cv2.resize(img, resize_scal)

        mask = np.array(Image.open(mask_path))

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    # masks_2st = abs(masks_2st-1)
    return img, mask


def default_scalCHABSE_loader(img_path, mask_path, crop_scal=(512, 512), resize_scal=(960, 960)):
    is_train = img_path.split('/')[3]

    if is_train == "train":
        image = Image.open(img_path)
        label = Image.open(mask_path)

        image, label = RandomCrop(image, label, crop_scal)

        img = np.array(image)
        mask = np.array(label)

        img = randomHueSaturationValue(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    if is_train == "test":
        img = np.array(Image.open(img_path))
        img = cv2.resize(img, resize_scal)

        mask = np.array(Image.open(mask_path))

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    # masks_2st = abs(masks_2st-1)
    return img, mask


def read_STARE_datasets(root_path, train=True):
    images = []
    masks = []
    if train:
        image_root = os.path.join(root_path, 'train/images')
        gt_root = os.path.join(root_path, 'train/masks')
    else:
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/masks')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0].split('.')[0] + '_mask.tiff')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_Lung_datasets(root_path, train=True):
    images = []
    masks = []
    if train:
        image_root = os.path.join(root_path, 'train/images')
        gt_root = os.path.join(root_path, 'train/masks')
    else:
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/masks')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0].split('.')[0] + '_mask.tif')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_GLAS_datasets(root_path, train=True):
    images = []
    masks = []
    if train:
        image_root = os.path.join(root_path, 'train/images')
        gt_root = os.path.join(root_path, 'train/masks')
    else:
        image_root = os.path.join(root_path, 'test/images')
        gt_root = os.path.join(root_path, 'test/masks')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('.')[0].split('.')[0] + '_anno.tif')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


class DriveDataset(data.Dataset):

    def __init__(self, root_path, train=True):
        self.root = root_path
        self.images, self.labels = read_DRIVE_datasets(self.root, train=train)

    def __getitem__(self, index, train=True):
        img, mask = default_DRIVE_loader(self.images[index], self.labels[index])

        # img, mask = default_scalDRIVE_loader(self.images[index], self.labels[index])

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return {
            'image': img,
            'masks_2st': mask
        }

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)


class StareDataset(data.Dataset):

    def __init__(self, root_path, train=True):
        self.root = root_path
        self.images, self.labels = read_STARE_datasets(self.root, train=train)

    def __getitem__(self, index, train=True):
        img, mask = default_STARE_loader(self.images[index], self.labels[index])

        # img, mask = default_scalSTARE_loader(self.images[index], self.labels[index])

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return {
            'image': img,
            'masks_2st': mask
        }

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)


class ChasedDataset(data.Dataset):

    def __init__(self, root_path, train=True):
        self.root = root_path
        self.images, self.labels = read_STARE_datasets(self.root, train=train)

    def __getitem__(self, index, train=True):
        # img, mask = default_DRIVE_loader(self.images[index], self.labels[index])

        img, mask = default_scalCHABSE_loader(self.images[index], self.labels[index])

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return {
            'image': img,
            'masks_2st': mask
        }

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)


class LungDataset(data.Dataset):

    def __init__(self, root_path, train=True):
        self.root = root_path
        self.images, self.labels = read_Lung_datasets(self.root, train=train)

    def __getitem__(self, index, train=True):
        img, mask = default_DRIVE_loader(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return {
            'image': img,
            'masks_2st': mask
        }

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)


class GLASDataset(data.Dataset):

    def __init__(self, root_path, train=True):
        self.root = root_path
        self.images, self.labels = read_GLAS_datasets(self.root, train=train)

    def __getitem__(self, index, train=True):
        img, mask = default_GLAS_loader(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return {
            'image': img,
            'masks_2st': mask
        }

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)


if __name__ == '__main__':
    data = ChasedDataset('../data/CHASEDB1', True)
    for n, i in enumerate(data):
        print(i)

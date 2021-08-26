import os
import torch
import numpy as np
import cv2
import random
def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def resize_label_batch(label, size):
    label_resized = np.zeros((size, size, 1, label.shape[3]))
    interp = torch.nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = torch.autograd.Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>21] = 255
    return label_resized


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy').item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]

def RandomCrop(imgarr, gt, cropsize):

    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    img_container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)
    gt_container = np.zeros((cropsize, cropsize), np.float32)
    cropping = np.zeros((cropsize, cropsize), np.bool)

    img_container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    gt_container[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
        gt[img_top:img_top + ch, img_left:img_left + cw]
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1
    return img_container, gt_container, cropping


def generate_original_images(images_temp, batch_size):
    ori_images = np.zeros_like(images_temp)
    for i in range(batch_size):
        img_temp = images_temp[i].transpose(1, 2, 0)
        img_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        img_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        img_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[i] = img_temp.transpose(2, 0, 1)
    return ori_images
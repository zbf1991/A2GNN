import os

import xml.etree.ElementTree as ET
import torch.nn.functional as F
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import numpy as np

obj_dict = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4, 'bottle':5,
            'bus':6,'car':7,'cat':8,'chair':9,'cow':10,'diningtable':11,
            'dog':12,'horse':13,'motorbike':14,'person':15,'pottedplant':16,
            'sheep':17,'sofa':18,'train':19,'tvmonitor':20}

def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []
    objects = []
    for object in root.findall('object'):

        bndbox = object.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append(object.find('name').text)
        bndboxlist.append([xmin,ymin,xmax,ymax])
    return objects, bndboxlist

def box_mil_loss(pred, boxes, objs):
    mil_loss = 0
    for i, box in enumerate(boxes):
        obj = objs[i]
        label_box_index = obj_dict[obj]
        pred_box = pred[:, :, box[1]:box[3] + 1, box[0]:box[2] + 1]
        pred_row_max, ind_row = torch.max(pred_box, dim=2)
        pred_col_max, ind_col = torch.max(pred_box, dim=3)
        pred_box = torch.cat([pred_row_max, pred_col_max], dim=2)
        label_box = label_box_index * torch.ones([1,pred_box.shape[2]])
        mil_loss += F.cross_entropy(pred_box,label_box.cuda().long())

    mil_loss = mil_loss/len(boxes)
    return mil_loss


def box_mp_loss_with_consistency_checking(pred, fts, boxes, objs, init_seed_label):
    mil_loss = 0
    mil_sum = 0
    dis_loss = 0
    dis_sum = 0
    for i, box in enumerate(boxes):
        obj = objs[i]
        label_box_index = obj_dict[obj]
        pred_box = pred[:,:, box[1]:box[3] + 1, box[0]:box[2] + 1]
        pred_class_box = pred[:, label_box_index, box[1]:box[3] + 1, box[0]:box[2] + 1]
        fts_box = fts[:, :, box[1]:box[3] + 1, box[0]:box[2] + 1]
        pred_row_max, ind_row = torch.max(pred_class_box, dim=2)
        pred_col_max, ind_col = torch.max(pred_class_box, dim=1)
        indice_row = torch.arange(0,pred_row_max.shape[-1]).cuda()
        indice_col = torch.arange(0,pred_col_max.shape[-1]).cuda()
        ind_row = torch.squeeze(ind_row[:, :])
        ind_col = torch.squeeze(ind_col[:, :])
        pred_row = pred_box[:,:,indice_row,ind_row]
        pred_col = pred_box[:,:,ind_col,indice_col]

        fts_row = fts_box[:,:,indice_row, ind_row]
        fts_col = fts_box[:, :, ind_col, indice_col]

        pred_box = torch.cat([pred_row, pred_col], dim=2)
        fts_max_box = torch.cat([fts_row, fts_col], dim=2).squeeze().transpose(1,0)

        norm2 = torch.norm(fts_max_box, 2, 1).view(-1, 1)

        cos = torch.div(torch.mm(fts_max_box, fts_max_box.t()), torch.mm(norm2, norm2.t()) + 1e-7)
        dis_loss += torch.sum(1-cos+1e-5)
        dis_sum += cos.shape[0]*cos.shape[1]

        label_box = label_box_index * torch.ones([1,pred_box.shape[2]])
        mil_loss += F.cross_entropy(pred_box,label_box.cuda().long(),reduction='sum')
        mil_sum += pred_box.shape[2]

        fts_max_box_prototype = torch.mean(fts_max_box, dim=0)
        init_seed_label_box = init_seed_label[:, box[1]:box[3] + 1, box[0]:box[2] + 1]
        init_seed_fts_box = fts[:, :, box[1]:box[3] + 1, box[0]:box[2] + 1]
        init_seed_fts_box = init_seed_fts_box.contiguous().view(256,-1)
        fts_seed_dis = F.cosine_similarity(init_seed_fts_box, fts_max_box_prototype.unsqueeze(dim=1), dim=0)
        fts_seed_dis = fts_seed_dis.view(1, init_seed_label_box.shape[1], init_seed_label_box.shape[2])
        new_init_seed_label_box = init_seed_label_box.clone()
        new_init_seed_label_box[fts_seed_dis<0] = 255
        new_init_seed_label_box[init_seed_label_box!=label_box_index] = init_seed_label_box[init_seed_label_box!=label_box_index]

        init_seed_label[:, box[1]:box[3] + 1, box[0]:box[2] + 1] = new_init_seed_label_box

    mil_loss = mil_loss/ mil_sum
    dis_loss = dis_loss/dis_sum
    return mil_loss+dis_loss, init_seed_label


def generate_aff(f_w, f_h, aff_mat, radius):
    aff = torch.zeros([f_w * f_h, f_w * f_h])
    aff_mask = torch.zeros([f_w, f_h])
    aff_mask_pad = F.pad(aff_mask, (radius, radius, radius, radius),'constant')
    aff_mat = torch.squeeze(aff_mat)
    for i in range(f_w):
        for j in range(f_h):
            ind = i*f_h+j
            center_x = i+radius
            center_y = j+radius
            aff_mask_pad[center_x - radius: (center_x+radius+1), center_y-radius : (center_y+radius+1)] = aff_mat[:,:,i,j]
            aff_mask_nopad = aff_mask_pad[radius:-radius, radius:-radius]
            aff[ind] = aff_mask_nopad.reshape(1,-1)
            aff_mask_pad = 0*aff_mask_pad

    return aff


def generate_aff_cropping(f_w, f_h, radius):
    aff = torch.zeros([f_w * f_h, f_w * f_h])
    aff_mask = torch.zeros([f_w, f_h])
    aff_mask_pad = F.pad(aff_mask, (radius, radius, radius, radius),'constant')

    for i in range(f_w):
        for j in range(f_h):
            ind = i*f_h+j
            center_x = i+radius
            center_y = j+radius
            aff_mask_pad[center_x - radius: (center_x+radius+1), center_y-radius : (center_y+radius+1)] = 1
            aff_mask_nopad = aff_mask_pad[radius:-radius, radius:-radius]
            aff[ind] = aff_mask_nopad.reshape(1,-1)
            aff_mask_pad = 0*aff_mask_pad

    return aff

def train_with_mp_nocrf(epoch, model, features, adj, labels, critersion, f_h, f_w, boxes,objs, aff_cropping, optimizer):
    model.train()
    optimizer.zero_grad()
    x, fts = model(features.cuda(), adj.cuda(), aff_cropping.cuda())
    label_b, label_w, label_h = labels.shape
    x = x.view(1, f_h, f_w, 21)
    x = x.permute(0, 3, 1, 2)
    pred = F.interpolate(x, [label_w, label_h], mode='bilinear', align_corners=False)

    fts = fts.view(1, f_h, f_w, -1)
    fts = fts.permute(0, 3, 1, 2)
    fts = F.interpolate(fts, [label_w, label_h], mode='bilinear', align_corners=False)

    mil_loss, _ = box_mp_loss_with_consistency_checking(pred, fts, boxes, objs, labels)

    bg_label = labels.clone()
    fg_label = labels.clone()
    bg_label[labels != 0] = 255
    fg_label[labels == 0] = 255

    bgloss = critersion(pred, bg_label.cuda())
    fgloss = critersion(pred, fg_label.cuda())
    loss_train = bgloss + fgloss + 1*mil_loss

    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'bg_loss: {:.4f}'.format(bgloss.item()),
          'fg_loss:{:.4f}'.format(fgloss.item()),
          'mil_loss:{:.4f}'.format(mil_loss.item()))
    return loss_train, model


def train_with_mp_crf(epoch, model, features, adj, labels, critersion, f_h, f_w, densecrflosslayer, normalized_input,
                            boxes, objs, aff_cropping, optimizer):
    model.train()
    optimizer.zero_grad()
    x, fts= model(features.cuda(), adj.cuda(), aff_cropping.cuda())
    label_b, label_w, label_h = labels.shape
    x = x.view(1, f_h, f_w, 21)
    x = x.permute(0, 3, 1, 2)
    pred_soft = F.softmax(x, dim=1)
    pred = F.interpolate(x, [label_w, label_h], mode='bilinear', align_corners=False)

    fts = fts.view(1, f_h, f_w, -1)
    fts = fts.permute(0, 3, 1, 2)
    fts = F.interpolate(fts, [label_w, label_h], mode='bilinear', align_corners=False)
    mil_loss, init_seed_label = box_mp_loss_with_consistency_checking(pred, fts, boxes, objs, labels)

    out_gatedcrf = densecrflosslayer(pred_soft, [{'weight': 1, 'xy':6, 'rgb': 0.1}], 5, normalized_input, f_h, f_w, mask_src=None)
    crfloss = out_gatedcrf['loss']

    labels = init_seed_label
    bg_label = labels.clone()
    fg_label = labels.clone()
    bg_label[labels != 0] = 255
    fg_label[labels == 0] = 255

    bgloss = critersion(pred, bg_label.cuda())
    fgloss = critersion(pred, fg_label.cuda())

    loss_train = bgloss + fgloss + 0.01*crfloss + 1*mil_loss

    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'bg_loss: {:.4f}'.format(bgloss.item()),
          'fg_loss:{:.4f}'.format(fgloss.item()),
          'crf_loss:{:.4f}'.format(crfloss.item()),
          'mil_loss:{:.4f}'.format(mil_loss.item()))
    return loss_train,model


def infer_nocrf(model, features, adj, f_h, f_w, aff_cropping):
    model.eval()
    x, output = model(features.cuda(), adj.cuda(), aff_cropping.cuda())
    x = x.reshape([f_h, f_w, 21])
    pred = x.permute(2, 0, 1)

    return pred


def crf_inference_inf(img, probs, t=10, scale_factor=1, labels=21):

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=2/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=50/scale_factor, srgb=5, rgbim=np.copy(img_c), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def train_with_nocrf(epoch, model, features, adj, labels, critersion, f_h, f_w, aff_cropping, optimizer):
    model.train()
    optimizer.zero_grad()
    x, fts = model(features.cuda(), adj.cuda(), aff_cropping.cuda())
    label_b, label_w, label_h = labels.shape
    x = x.view(1, f_h, f_w, 21)
    x = x.permute(0, 3, 1, 2)
    pred = F.interpolate(x, [label_w, label_h], mode='bilinear', align_corners=False)

    bg_label = labels.clone()
    fg_label = labels.clone()
    bg_label[labels != 0] = 255
    fg_label[labels == 0] = 255

    bgloss = critersion(pred, bg_label.cuda())
    fgloss = critersion(pred, fg_label.cuda())
    loss_train = bgloss + fgloss

    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'bg_loss: {:.4f}'.format(bgloss.item()),
          'fg_loss:{:.4f}'.format(fgloss.item()))
    return loss_train, model


def train_with_crf(epoch, model, features, adj, labels, critersion, f_h, f_w, densecrflosslayer, normalized_input, aff_cropping,
                   optimizer):
    model.train()
    optimizer.zero_grad()
    x, _= model(features.cuda(), adj.cuda(), aff_cropping.cuda())
    label_b, label_w, label_h = labels.shape
    x = x.view(1, f_h, f_w, 21)
    x = x.permute(0, 3, 1, 2)
    pred_soft = F.softmax(x, dim=1)
    pred = F.interpolate(x, [label_w, label_h], mode='bilinear', align_corners=False)

    out_gatedcrf = densecrflosslayer(pred_soft, [{'weight': 1, 'xy':6, 'rgb': 0.1}], 5, normalized_input, f_h, f_w, mask_src=None)
    crfloss = out_gatedcrf['loss']

    bg_label = labels.clone()
    fg_label = labels.clone()
    bg_label[labels != 0] = 255
    fg_label[labels == 0] = 255

    bgloss = critersion(pred, bg_label.cuda())
    fgloss = critersion(pred, fg_label.cuda())

    loss_train = bgloss + fgloss + 0.3*crfloss

    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'bg_loss: {:.4f}'.format(bgloss.item()),
          'fg_loss:{:.4f}'.format(fgloss.item()),
          'crf_loss:{:.4f}'.format(crfloss.item()))
    return loss_train,model


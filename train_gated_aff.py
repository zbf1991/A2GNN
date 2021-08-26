import os
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
cudnn.enabled = True
from torchvision import transforms
import voc12.data
import tool.pyutils as pyutils
import tool.torchutils as torchutils
import tool.imutils as imutils
import argparse
import importlib
import torch.nn.functional as F
from model_fts_gated_regularized import ModelLossSemsegGatedCRF
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def _unfold(img, radius):
    assert img.dim() == 4, 'Unfolding requires NCHW batch'
    N, C, H, W = img.shape
    diameter = 2 * radius + 1
    return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=12, type=int)
    parser.add_argument("--network", default="network.resnet38_aff_gated", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument('--radius', type=int, default=4)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--session_name", default="res_aff_box", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    # parser.add_argument("--weights", default='./netWeights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--weights", default='./netWeights/resnet38_aff_SEAM.pth', type=str)
    parser.add_argument("--voc12_root", default='/data/zbf_data/dataset/VOCdevkit/VOC2012', type=str)
    parser.add_argument("--label_dir",
                        default='./data/Init_Label/SEAM_Box',
                        type=str)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'Net')()

    print(model)

    train_dataset = voc12.data.VOC12AffGtDataset_NoExtract(args.train_list, label_dir=args.label_dir,
                                               voc12_root=args.voc12_root, cropsize=args.crop_size, radius=5,
                   joint_transform_list=[
                       None,
                       None,
                       imutils.RandomCrop(args.crop_size),
                       imutils.RandomHorizontalFlip()
                   ],
                   img_transform_list=[
                       transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                              saturation=0.3, hue=0.1),
                       np.asarray,
                       model.normalize,
                       imutils.HWC_to_CHW
                   ],
                   label_transform_list=[
                       None,
                       None,
                       None,
                       imutils.LabelResize(args.crop_size//8)
                   ])

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert args.network == "network.resnet38_aff_gated"
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss', 'crf_loss', 'bg_cnt', 'fg_cnt',
                                     'neg_cnt')

    gatedcrf = ModelLossSemsegGatedCRF()

    timer = pyutils.Timer("Session started: ")
    radius = args.radius

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            features, aff, aff_crf = model.forward(pack[0], radius)
            aff = aff.unsqueeze(dim=1)
            label = pack[1].cuda().unsqueeze(dim=1)
            label = _unfold(label,radius=radius)
            N, _, H, W = features.shape

            crf_img = F.interpolate(pack[0], [H, W], mode='bilinear', align_corners=False)
            crf_input = {'rgb': crf_img.cuda()}

            label_center = label[:, :, radius, radius, :, :].view(N, 1, 1, 1, H, W)
            label_center = label_center.expand_as(label)

            aff_label = torch.eq(label, label_center)
            aff_label_ignore = torch.eq(label,255)

            aff_label[aff_label_ignore == 1] = 255
            aff_label[label_center == 255] = 255

            bg_pos_label_need = torch.zeros_like(aff_label)
            bg_pos_label_need[aff_label == 1] = 1
            bg_pos_label_need[label_center != 0] = 0
            bg_pos_label_need[aff_label_ignore == 1] = 0

            fg_pos_label_need = torch.zeros_like(aff_label)
            fg_pos_label_need[aff_label == 1] = 1
            fg_pos_label_need[label_center == 0] = 0
            fg_pos_label_need[aff_label_ignore ==1] = 0
            fg_pos_label_need[label_center == 255] = 0

            neg_label_need = torch.zeros_like(aff_label)
            neg_label_need[aff_label==0] =1
            neg_label_need[aff_label_ignore==1] = 0
            neg_label_need[label_center==255] = 0

            bg_count = torch.sum(bg_pos_label_need).float() + 1e-5
            fg_count = torch.sum(fg_pos_label_need).float() + 1e-5
            neg_count = torch.sum(neg_label_need).float() + 1e-5

            bg_loss = torch.sum(- bg_pos_label_need.float() * torch.log(aff + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_pos_label_need.float() * torch.log(aff + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label_need.float() * torch.log(1. + 1e-5 - aff)) / neg_count

            croppings = pack[2].cuda().unsqueeze(dim=1)
            croppings = _unfold(croppings, radius=radius)
            croppings_center = croppings[:, :, radius, radius, :, :].view(N, 1, 1, 1, H, W)
            croppings_center = croppings_center.expand_as(croppings)
            croppings_ignore = torch.eq(croppings, 0)
            croppings = torch.ones_like(croppings)
            croppings[croppings_ignore == 1] = 0
            croppings[croppings_center == 0] = 0

            out_gatedcrf = gatedcrf(aff_crf, [{'weight': 1, 'xy': 6, 'rgb': 0.1}], radius, crf_input, H, W,
                                    mask_src=croppings.float())
            crf_loss = 3*out_gatedcrf['loss']

            loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2 + crf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(),
                'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                'crf_loss': crf_loss.item(),
                'bg_cnt': bg_count.item(), 'fg_cnt': fg_count.item(), 'neg_cnt': neg_count.item()
            })

            if (optimizer.global_step - 1) % 5 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'crf_loss', 'bg_loss', 'fg_loss',
                                                                      'neg_loss'),
                      'cnt:%.0f %.0f %.0f' % avg_meter.get('bg_cnt', 'fg_cnt', 'neg_cnt'),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()


        else:
            print('')
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')

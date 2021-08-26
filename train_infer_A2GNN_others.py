import torch
import torchvision
from tool import imutils
import argparse
import importlib
import numpy as np
from pygcn.A2GNN import A2GNN
import voc12.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path
import torch.optim as optim
import cv2
from model_loss_semseg_gatedcrf import ModelLossSemsegGatedCRF
import imageio
import mytool


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='./netWeights/final_model/aff_scribble.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_aff_gated", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='/your/path/VOCdevkit/VOC2012', type=str)
    parser.add_argument("--cam_dir", default='/your/path/cam', type=str)
    parser.add_argument("--seed_label_root", default='./data/Init_Label/Scribble_SuperPixel', type=str)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument("--alpha", default=6, type=float)
    parser.add_argument("--rw_weight", default=3, type=int)
    parser.add_argument("--logt", default=6, type=int)
    parser.add_argument("--beta", default=8, type=int)
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.03, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--num_class", default=21, type=int)
    parser.add_argument("--save_path", default="./out/scribble_pred", type=str)
    parser.add_argument("--rw", default=False, type=bool)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    model.load_state_dict(torch.load(args.weights), strict=False)

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ImageDataset(args.infer_list, voc12_root=args.voc12_root,
                                               transform=torchvision.transforms.Compose(
        [np.asarray,
         model.normalize,
         imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for iter, (name, img) in enumerate(infer_data_loader):
        name = name[0]
        if os.path.exists(os.path.join(args.save_path, name+'.png')):
            print(iter, 'has finished')
        else:
            print(iter)
            img_nopad = img.clone()
            orig_shape = img.shape
            padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

            p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
            img = F.pad(img, p2d)

            dheight = int(np.ceil(img.shape[2]/8))
            dwidth = int(np.ceil(img.shape[3]/8))

            with torch.no_grad():
                features, aff_mat, _ = model.forward(img.cuda(), radius=args.radius, to_dense=False)
                aff_mat = torch.pow(aff_mat, 1).squeeze(dim=0)
                f_h, f_w = features.shape[-2], features.shape[-1]
                aff_mat = mytool.generate_aff(f_h, f_w, aff_mat, radius=args.radius)
                aff_cropping = mytool.generate_aff_cropping(f_h, f_w, radius=args.radius)

            normalized_img = F.interpolate(img, [f_h, f_w], mode='bilinear', align_corners=False)
            normalized_input = {'rgb': normalized_img.cuda()}

            seed_label = cv2.imread(os.path.join(args.seed_label_root,name+'.png'),cv2.IMREAD_GRAYSCALE)
            gt_h, gt_w = seed_label.shape[-2], seed_label.shape[-1]
            seed_label = np.pad(seed_label, ((0, p2d[3]), (0, p2d[1])), mode='constant')

            seed_label = torch.from_numpy(seed_label).unsqueeze(dim=0)
            seed_label = seed_label.long()

            adj = aff_mat.cuda()
            adj = adj - torch.diag(torch.diag(adj))
            adj = adj + torch.eye(adj.shape[0]).cuda()
            features = features.squeeze(dim=0)
            features = features.permute(1,2,0)
            features = features.view(-1,5632)
            features = features/torch.sum(features, dim=1, keepdim=True)

            gnn_model = A2GNN(nfeat=features.shape[1],
                         nhid=args.hidden,
                         nclass=args.num_class,
                         nlayers=3,
                         dropout_rate=args.dropout)

            gatedcrf = ModelLossSemsegGatedCRF()
            critersion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean').cuda()
            optimizer = optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            gnn_model.cuda()
            optimizer.zero_grad()

            for epoch in range(args.epochs):
                loss_train, gnn_model = mytool.train_with_nocrf(epoch, gnn_model, features, adj, seed_label, critersion,
                                                                f_h, f_w, aff_cropping, optimizer)

            for epoch in range(args.epochs):
                loss_train, gnn_model = mytool.train_with_crf(epoch, gnn_model, features, adj, seed_label, critersion,
                                                              f_h, f_w, gatedcrf, normalized_input, aff_cropping, optimizer)

            pred = mytool.infer_nocrf(gnn_model, features, adj, f_h, f_w, aff_cropping)

            pred = F.interpolate(torch.unsqueeze(pred, dim=0), [gt_h, gt_w], mode='bilinear', align_corners=False)
    #------------------------------------RW-----------------------------------------------------
            if args.rw==True:
                aff_mat = torch.pow(aff_mat, args.beta)
                aff_mat = torch.squeeze(aff_mat)
                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                for _ in range(args.logt):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                cam = np.load(os.path.join(args.cam_dir, name + '.npy'), allow_pickle=True).item()

                cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
                for k, v in cam.items():
                    cam_full_arr[k + 1] = v
                cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False)) ** args.alpha

                cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

                cam_full_arr = torch.from_numpy(cam_full_arr)
                cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

                cam_vec = cam_full_arr.view(args.num_class, -1)
                cam_rw = torch.matmul(cam_vec.cuda(), trans_mat.cuda())

                cam_rw = cam_rw.view(1, args.num_class, dheight, dwidth)
                cam_rw = F.interpolate(cam_rw, [img.shape[2], img.shape[3]], mode='bilinear',
                                     align_corners=False)

                cam_rw_pred = (cam_rw[:,:,:orig_shape[2], :orig_shape[3]]).cpu().data.numpy()
                cam_rw_pred = np.squeeze(cam_rw_pred)

                pred_probs = F.softmax(pred, dim=1).cpu().data.numpy()
                pred_probs = np.squeeze(pred_probs)

                original_img = np.array(imageio.imread(os.path.join(args.voc12_root + '/JPEGImages', name + '.jpg'))).astype(
                    np.uint8)
                cam_pred = mytool.crf_inference_inf(original_img, pred_probs, labels=args.num_class)

                cam_img = np.argmax(cam_pred, 0)

                pred_probs = pred_probs + args.rw_weight*cam_rw_pred
                pred_probs = pred_probs / np.sum(pred_probs, axis=0, keepdims=True)

                crf_pred = mytool.crf_inference_inf(original_img, pred_probs, labels=args.num_class)
                crf_pred = np.argmax(crf_pred, 0)
                imageio.imsave(os.path.join(args.save_path, name + '.png'), crf_pred.astype(np.uint8))
            else:
                pred_probs = F.softmax(pred, dim=1).cpu().data.numpy()
                pred_probs = np.squeeze(pred_probs)
                original_img = np.array(imageio.imread(os.path.join(args.voc12_root + '/JPEGImages', name + '.jpg'))).astype(
                    np.uint8)
                crf_pred = mytool.crf_inference_inf(original_img, pred_probs, labels=args.num_class)
                crf_pred = np.argmax(crf_pred, 0)

                imageio.imsave(os.path.join(args.save_path, name + '.png'), crf_pred.astype(np.uint8))
                torch.cuda.empty_cache()
        



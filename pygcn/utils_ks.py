import os
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve2d
import torch
from skimage import io
from skimage.segmentation.boundaries import find_boundaries
import argparse
import cv2

from scipy.spatial.distance import cdist as Distance
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="/home/zbf/pythonProject/pygcn/data/voc", dataset="F_ks_bg_onehot.npy"):
    idx_features = np.load(os.path.join(path, dataset))
    features = sp.csr_matrix(idx_features[:,1:], dtype=np.float32)
    idx = np.array(idx_features[:,0], dtype=np.int16)
    adj = np.load('/home/zbf/pythonProject/pygcn/data/voc/M_ks_bg(th=0.01,p=0.2).npy')
    adj = sp.csr_matrix(adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def compute_sp_adj(sp_img):
    sp_unique = np.unique(sp_img)
    adj = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    sp_adj = {}
    for key in sp_unique:
        mask = convolve2d((sp_img == key).astype(np.float), adj, 'same') > 0
        value = np.unique(sp_img[mask]).astype(np.int)
        sp_adj[str(key)] = value[value != key]
    return sp_adj

def build_dict_index(sp_features):
    sp_index = {}
    num = 0
    for k in sp_features.keys():
        sp_index[k] = num
        num+=1
    return sp_index


def compute_weight(index_feature, adj_feature):
    ed = np.linalg.norm((index_feature - adj_feature), ord =1)
    h = len(index_feature)
    weight_ij = np.exp(-ed/(2*h))
    return weight_ij

def add_center_node(sp_features, adj, sp_label):
    ori_features = sp_features
    ori_labels = np.array(sp_label).astype(np.int)
    ori_features_w, ori_features_h = ori_features.shape
    # center_features = ori_features[ori_labels != 255]
    # center_labels = ori_labels[ori_labels != 255]
    # index = np.argwhere(ori_labels != 255)
    unique_labels = np.unique(ori_labels)
    if unique_labels[-1] == 255:
        unique_labels = unique_labels[:-1]
    else:
        unique_labels = unique_labels
    center_features = np.zeros([unique_labels.shape[0], ori_features_h])
    center_label = np.zeros_like(unique_labels)
    for i,cls in enumerate(unique_labels):
        single_class_feature = ori_features[ori_labels == cls]
        center_features[i,:] = np.mean(single_class_feature, axis=0)
        center_label[i] = cls
    center_label_w = center_label.shape[0]
    new_features = np.vstack((ori_features, center_features))
    new_labels = np.hstack((ori_labels, unique_labels))
    center_adj = np.identity(center_label_w)
    new_adj_shape = ori_features_w + center_label_w
    new_adj = np.ones([new_adj_shape, new_adj_shape])
    new_adj[0:ori_features_w, 0:ori_features_w] = adj
    new_adj[ori_features_w:,ori_features_w:] = center_adj

    return new_features,new_adj, new_labels

def add_spatial_center_node(features, adj, sp_label):
    ori_features = features
    ori_labels = np.squeeze(sp_label).astype(np.int)
    ori_features_w, ori_features_h = ori_features.shape
    # center_features = ori_features[ori_labels != 255]
    # center_labels = ori_labels[ori_labels != 255]
    # index = np.argwhere(ori_labels != 255)
    unique_labels = np.unique(ori_labels)
    if unique_labels[-1] == 255:
        unique_labels = unique_labels[:-1]
    else:
        unique_labels = unique_labels
    center_features = np.zeros([unique_labels.shape[0], ori_features_h])
    center_label = np.zeros_like(unique_labels)
    for i,cls in enumerate(unique_labels):
        single_class_feature = ori_features[ori_labels == cls]
        center_features[i,:] = np.mean(single_class_feature, axis=0)
        center_label[i] = cls
    center_label_w = center_label.shape[0]
    new_features = np.vstack((ori_features, center_features))
    new_labels = np.hstack((ori_labels, unique_labels))
    # center_adj = np.identity(center_label_w)
    center_adj = np.zeros([center_label_w,center_label_w])
    new_adj_shape = ori_features_w + center_label_w
    new_adj = np.ones([new_adj_shape, new_adj_shape])
    new_adj[0:ori_features_w, 0:ori_features_w] = adj
    new_adj[ori_features_w:,ori_features_w:] = center_adj

    return new_features,new_adj, new_labels


def build_graph(sp_features, sp_img, sp_label):
    sp_adj = compute_sp_adj(sp_img)
    sp_index = build_dict_index(sp_features)
    features = []
    labels = []
    weight_matrix = np.zeros((len(sp_index),len(sp_index)))

    for k, v in sp_index.items():
        index = v
        index_features = sp_features[k]
        index_label = sp_label[k]
        index_adj = sp_adj[k]
        features.append(index_features)
        labels.append(index_label)
        for index_adj_j_k in index_adj:
            index_adj_j_v = sp_index[str(index_adj_j_k)]
            adj_feature = sp_features[str(index_adj_j_k)]
            weight_matrix[index, index_adj_j_v] = compute_weight(index_features,adj_feature)
    features = np.array(features)

    adj = np.ones_like(weight_matrix)
    th = np.abs(np.mean(weight_matrix) - np.std(weight_matrix))
    for i in range(weight_matrix.shape[0]):
        max_row = np.max(weight_matrix[i,:])
        not_region = np.logical_and(weight_matrix[i, :] < max_row,weight_matrix[i,:] < th)
        adj[i,:][not_region] = 0

    new_features, new_adj, new_labels = add_center_node(features, adj, labels)
    features = sp.csr_matrix(new_features)
    adj = sp.csr_matrix(new_adj)

    adj = adj + adj.T.multiply(adj.T > adj) - - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features, adj, new_labels


def build_adding_graph(features, labels, M_ks):
    ori_features = features.numpy()
    ori_labels = np.array(labels).astype(np.int)
    add_features = ori_features[ori_labels != 255]
    add_labels = ori_labels[ori_labels != 255]
    index = np.argwhere(ori_labels != 255)
    unique_labels = np.unique(add_labels)
    w, h = add_features.shape
    add_adj = np.ones((w, w))

    for first_i in unique_labels:
        for second_i in unique_labels:
            first_index = np.argwhere(add_labels == first_i)
            second_index = np.argwhere(add_labels == second_i)
            first_value = M_ks[first_i, second_i]
            second_value = M_ks[first_i, second_i]
            for i in first_index:
                for j in second_index:
                    add_adj[i,j] = first_value
                    add_adj[j,i] = second_value

    # zero_index = np.argwhere(add_labels == 0)
    # non_zero_index = np.argwhere(add_labels != 0)
    # for i in zero_index:
    #     for j in non_zero_index:
    #         add_adj[i,j] = 0
    #         add_adj[j,i] = 0

    print('111')

# def softmax(x):
#     e_x = np.exp(x - np.max(x, axis=1))
#     probs = e_x / np.sum(e_x, axis=0)
#     return probs


def generate_spatial_adj(features):
    features = np.squeeze(features.transpose(2,0,1))
    c = features.shape[0]
    features_a = features.copy().reshape([c,-1])
    features_b = features_a.copy().T

    spatial_attention = Distance(features_b, features_b, metric='euclidean')
    spatial_attention_ori = spatial_attention.copy()

    spatial_th = 1*np.abs(np.mean(spatial_attention_ori, axis=1) - np.std(spatial_attention_ori, axis=1))
    spatial_attention = sc_softmax(spatial_attention, axis=1)

    spatial_adj = np.ones_like(spatial_attention)
    spatial_adj[spatial_attention_ori > spatial_th] = 0
    spatial_adj = spatial_adj - np.diag(np.diag(spatial_adj))

    spatial_features = features_b

    return spatial_features, spatial_adj


def generate_spatial_channel_adj(features):
    features = np.squeeze(features.transpose(2,0,1))
    c = features.shape[0]
    features_a = features.copy().reshape([c,-1])
    features_b = features_a.copy().T

    spatial_attention = np.dot(features_b, features_a)
    spatial_attention = spatial_attention - np.diag(np.diag(spatial_attention))
    spatial_attention_ori = spatial_attention.copy()
    # a = np.mean(spatial_attention, axis=1)
    # b = np.std(spatial_attention, axis=1)
    # spatial_th = np.abs(np.mean(spatial_attention_ori,axis=1) - np.std(spatial_attention_ori,axis=1))
    spatial_th = np.mean(spatial_attention, axis=1) + 0 * (np.max(spatial_attention_ori, axis=1) - np.mean(spatial_attention_ori, axis=1))
    spatial_attention = sc_softmax(spatial_attention, axis=1)

    spatial_adj = np.ones_like(spatial_attention)
    spatial_adj[spatial_attention < spatial_th] = 0
    spatial_adj = spatial_adj - np.diag(np.diag(spatial_adj))

    channel_attention = np.dot(features_a, features_b)
    # channel_th = np.abs(np.mean(channel_attention, axis=1) - np.std(channel_attention, axis=1))
    channel_th = 0.5
    channel_attention = sc_softmax(channel_attention, axis=1)
    channel_adj = np.ones_like(channel_attention)
    channel_adj[channel_attention < channel_th] = 0
    channel_adj = channel_adj - np.diag(np.diag(channel_adj))

    spatial_features = features_b
    channel_features = features_a

    return spatial_features, channel_features, spatial_adj, channel_adj

def build_spatial_graph(features, adj):
    # new_features, new_adj, new_labels = add_spatial_center_node(features, adj, labels)
    new_features = features
    new_adj = adj
    features = sp.csr_matrix(new_features)
    adj = sp.csr_matrix(new_adj)

    # adj = adj + adj.T.multiply(adj.T > adj) - - adj.multiply(adj.T > adj)

    features = normalize(features)
    #adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features, adj

def build_channel_graph(features, adj):
    features = sp.csr_matrix(features)
    adj = sp.csr_matrix(adj)

    # adj = adj + adj.T.multiply(adj.T > adj) - - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features, adj

def normalize_img(img_temp):
    img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
    img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
    img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
    return img_temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--LISTpath", default="/home/zbf/pythonProject/pygcn/data/voclist/"
                                              "train_aug.txt", type=str)
    args = parser.parse_args()
    img_list = open(args.LISTpath).readlines()
    for i in img_list:
        sp_img_path = os.path.join('/home/zbf/pythonProject/pygcn/data/gcndata/test/sp/', i[:-1] + '.png')
        sp_features_path = os.path.join('/home/zbf/pythonProject/pygcn/data/gcndata/test/features/', i[:-1] + '.npy')
        sp_label_path = os.path.join('/home/zbf/pythonProject/pygcn/data/gcndata/test/label/', i[:-1] + '.npy')
        sp_img = np.load(sp_img_path, allow_pickle=True)
        sp_features = np.load(sp_features_path, allow_pickle=True)
        sp_label = np.load(sp_label_path, allow_pickle=True)

        spatial_features, channel_features, spatial_adj, channel_adj = generate_spatial_channel_adj(sp_features)
        new_spatial_features, new_spatial_adj, new_spatial_labels = build_spatial_graph(spatial_features, spatial_adj, sp_label)
        new_channel_features, new_channel_adj = build_channel_graph(channel_features, channel_adj)
        print('111')


        # M_ks= np.load('/home/zbf/pythonProject/pygcn/data/voc/M_ks_bg(th=0.4,p=0.2).npy',allow_pickle=True)
        # f, a, l = build_graph(sp_features, sp_img, sp_label)
        # build_adding_graph(f, l, M_ks)
        # add_center_node(f,a,l)
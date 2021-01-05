# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
import sound_frame_models
from util_sound_frame import AverageMeter, Logger, UnifLabelSampler
from tqdm import tqdm
from tensorflow.contrib.tensorboard.plugins import projector
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import Sound_Event_Detect_dc as SEDdc
import config as cfg
from PIL import Image
from PIL import ImageFile
from datetime import datetime
from torchsummary import summary

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='vgg16', help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=3,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=32, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=2500,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--label', type=int, default=None, help='select label')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform=transform
        # print(self.data.shape)
        # exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_data = torch.from_numpy(out_data).unsqueeze(0).float()
        return out_data

def main(args):
    labels = np.random.rand(100)
    labels = [int(i * 10) for i in labels]
    LABEL_CNT = {}
    
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = sound_frame_models.__dict__[args.arch]()
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # optimizer = torch.optim.Adam(
    #     filter(lambda x: x.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     weight_decay=10**args.wd,
    # )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict'].copy():
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # model_summary = "/".join([".", "model.tsv"])
    # model_summary_file = open(model_summary, "w")
    # print(model, file=model_summary_file)
    # model_summary_file.close()

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # load the data
    # print()

    end = time.time()
    dc_clustering = SEDdc.Clustering_dc(
        44100, 0.04, cfg.hop_length / cfg.sample_rate, 64, args.resume)
        
    dc_clustering.set_audio_path(train=False)
    dc_clustering.load_feature_name()

    key_list = list(dc_clustering.tag.keys())
    # print(key_list)
    # select_key = ["Cat", "Dog", "Running_water", "Dishes"]
    select_key = ["all"]
    # select_key = [""]

    for key_name in key_list:
        if key_name in select_key or "all" in select_key:
            print("\n### Start making strong label：{} ###".format(key_name))
            weak_frame = dc_clustering.tag[key_name]["data"]
            strong_frame = dc_clustering.strong_tag[key_name]["data"]

            dataset1 = Mydatasets(weak_frame)
            dataset2 = Mydatasets(strong_frame)

            if args.verbose:
                print('Load dataset: {0:.2f} s'.format(time.time() - end))

            dataloader1 = torch.utils.data.DataLoader(dataset1,
                                                    batch_size=args.batch,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
            dataloader2 = torch.utils.data.DataLoader(dataset2,
                                                    batch_size=args.batch,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
            # clustering algorithm to use
            # deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)
            deepcluster = New_Kmeans(args.nmb_cluster)

            # training convnet with DeepCluster
            # remove head
            model.top_layer = None
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

            # get the features for the whole dataset
            features1 = compute_features(dataloader1, model=model, N=len(dataset1))
            features2 = compute_features(dataloader2, model=model, N=len(dataset2))
            # features1 = compute_features(dataloader1, N=len(dataset1))
            # features2 = compute_features(dataloader2, N=len(dataset2))
            # cluster the features
            if args.verbose:
                print('Cluster the features')
            
            # normal_kmeans(dataloader1, dataloader2, len(dataset1), len(dataset2), args.nmb_cluster, key_name,DC=dc_clustering)

            # exit()

            _ = deepcluster.cluster(features1, features2, verbose=args.verbose)
            # embedding_projecter(deepcluster.feature, deepcluster.I, model, optimizer, deepcluster.spe_number)
            # exit()

            # 指定したキーの区間抽出
            dc_clustering.SED_Single(key_name, deepcluster.I, deepcluster.spe_number)
            # 区間抽出したデータを強ラベルデータ化
            dc_clustering.create_csv_wav_Single(key_name, resume=args.resume)

            # dc_clustering.merge_single(key_name)
            dc_clustering.save_csv()
        else:
            dc_clustering.create_csv_wav_Single(key_name, resume=args.resume, select=True)
            dc_clustering.save_csv()




def compute_features(dataloader, model=None, N=0):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, input_tensor in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        if model==None:
            aux = input_var            
        else:
            aux = model(input_var).data.cpu().numpy()
        # print(aux.shape)
        # exit()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    print(features.shape)
    return features

def normal_kmeans(dataloader1, dataloader2, N1, N2, k, key_name, DC=None):
    # batch_time = AverageMeter()
    # end = time.time()
    print("### Start Normal Kmeans：{}".format(key_name))
    features1 = normal_feature(dataloader1, N1)
    features2 = normal_feature(dataloader2, N2)
    print("weak:", features1.shape)
    print("strong:", features2.shape)

    I, J, loss = new_run_kmeans(features1, features2, k, verbose=args.verbose)
    # Clustering.cluster(features, verbose=args.verbose)
    LOG_DIR = "./normal/{}".format(current_time)
    writer = SummaryWriter(log_dir=LOG_DIR)
    label_cnt = "/".join([LOG_DIR, "label_cnt.tsv"])  

    maximum = 0
    for number in range(k):
        cnt = 0
        for j in range(len(J)):
            if number == J[j]: cnt=cnt+1
        print("{}:{}".format(number, cnt))
        if cnt > maximum:
            maximum = cnt
            spe_number = number
    # print("\nClustering:{}\n".format(key))
    print("Specified_number:{}\n".format(spe_number))
    # exit()

    LABEL_CNT = {}
    labels = I
    for i in labels:
        if i not in LABEL_CNT:
            LABEL_CNT[i] = 1
        else:
            LABEL_CNT[i] += 1
    LABEL_CNT = sorted(LABEL_CNT.items())
    print(LABEL_CNT)

    with open(label_cnt, "a") as label_cnt_file:
        label_cnt_file.write("{}\n".format(key_name))
        label_cnt_file.write("{}\n".format(LABEL_CNT))
        label_cnt_file.write("{}\n".format(spe_number))

    writer.add_embedding(features1, metadata=labels)
    writer.close()

    # DC.SED_Single(key_name, I, spe_number)
    # DC.create_csv_wav_file_Single(key_name)
    # DC.merge_single(key_name)

    # 指定したキーの区間抽出
    DC.SED_Single(key_name, I, spe_number)
    # 区間抽出したデータを強ラベルデータ化
    DC.create_csv_wav_file_Single(key_name, resume="normal")

    # exit()

def normal_feature(dataloader, N):
    batch_time = AverageMeter()
    end = time.time()
    for i, input_tensor in enumerate(dataloader):
        input_feature = input_tensor.reshape(input_tensor.shape[0], -1).numpy()
        if i == 0:
            features = np.zeros((N, input_feature.shape[1]), dtype='float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = input_feature
        else:
            # special treatment for final batch
            features[i * args.batch:] = input_feature

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features

def embedding_projecter(feature, pred, model, optimizer, spe_number):
    # LOG_DIR='logs'
    print("### Embedding_Projector ###")

    if args.label == None:
        LOG_DIR = "./runs/{}/{}/{}".format(args.arch, os.path.basename(args.resume), current_time)
        writer = SummaryWriter(log_dir=LOG_DIR)
        label_cnt = "/".join([LOG_DIR, "label_cnt.tsv"])
        model_summary = "/".join([LOG_DIR, "model.tsv"])
        optimizer_summary = "/".join([LOG_DIR, "optimizer.tsv"])

        LABEL_CNT = {}
        labels = pred
        with open(label_cnt, "w") as label_cnt_file:
            for i in labels:
                if i not in LABEL_CNT:
                    LABEL_CNT[i] = 1
                else:
                    LABEL_CNT[i] += 1
            LABEL_CNT = sorted(LABEL_CNT.items())
            label_cnt_file.write("{}\n".format(LABEL_CNT))
            label_cnt_file.write("{}\n".format(spe_number))

        model_summary_file = open(model_summary, "w")
        print(model, file=model_summary_file)
        model_summary_file.close()

    with open(optimizer_summary, "w") as op_sum:
        op_sum.write("Optimizer:\n")
        for i,j in optimizer.param_groups[0].items():
            if  not isinstance(j, list):
                op_sum.write("{}\n".format(i))
                op_sum.write("{}\n".format(j))  


        print(LABEL_CNT)
        writer.add_embedding(feature, metadata=labels)
        writer.close()

class New_Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data1, data2, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        if data1.shape[1] >= 256:
            xb = clustering.preprocess_features(data1)
            yb = clustering.preprocess_features(data2)
        else:
            xb = data1.astype('float32')
            row_sums = np.linalg.norm(xb, axis=1)
            xb = xb / row_sums[:, np.newaxis]

            yb = data2.astype('float32')
            row_sums = np.linalg.norm(yb, axis=1)
            yb = yb / row_sums[:, np.newaxis]             
        # cluster the data
        I, J, loss = new_run_kmeans(xb, yb, self.k, verbose)

        self.feature = xb
        self.I = I
        self.J = J

        print(len(I), len(J))

        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data1)):
            self.images_lists[I[i]].append(i)

        self.spe_number = self.cnt_strong_frame(self.J)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss
    
    def cnt_strong_frame(self, pred_strong):
        print("### 強ラベルデータによるクラスタ推定 ###")
        maximum = 0
        for number in range(self.k):
            cnt = 0
            ### term_listを使用しない場合 ###
            for i in range(len(pred_strong)):
                if number == pred_strong[i]: cnt=cnt+1
            print("{}:{}".format(number, cnt))
            if cnt > maximum:
                maximum = cnt
                spe_number = number
        # print("\nClustering:{}\n".format(key))
        # print("Specified_number:{}\n".format(spe_number))
        # exit()
        return spe_number

def new_run_kmeans(x, y, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    _, J = index.search(y, 1)

    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], [int(o[0]) for o in J], losses[-1]

if __name__ == '__main__':
    args = parse_args()
    main(args)


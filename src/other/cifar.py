import torch
from tqdm import tqdm
import os
import torchvision
import torchvision.transforms as transforms
from keras.datasets import cifar10
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2
import csv
import pandas as pd

def load_data(train=False):
    transform_test = transforms.Compose([
        transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10("../datasets", transform=transform_test, train=train, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False,
                                             num_workers=0)
    return testloader

def one_channel(trainX, trainy, testX, testy, rs=None):
    header = None
    splits = ['train', 'test']
    for split in splits:
        dt = trainX if split == 'train' else testX
        labels = trainy if split == 'train' else testy
        rows = []
        for i in range(dt.shape[0]):
            # change 3 channels to 1
            img = np.dot(dt[i], [0.2989, 0.5870, 0.1140])
            if rs:
                img = cv2.resize(img, rs)

            if header is None:
                header = []
                # for c in range(img.shape[2]):
                c = 0
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        header.append("i_{0}_{1}_{2}".format(c, x, y))
                header.append('class')

            row = []
            # for c in range(img.shape[2]):
            c = 0
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    row.append(img[x, y].item())
            row.append(labels[i].item())
            rows.append(row)

        saved_dir = "../datasets/cifar-10/{}all".format('{},{}/'.format(rs[0], rs[1]) if rs else '32,32/')
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        file = saved_dir + "/{0}.csv".format('train' if split == 'train' else 'test')
        with open(file, "w") as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerows(rows)

def comb(rs, ori=False):
    """
        combine train and test into one
    """
    saved_dir = "../datasets/cifar-10/{}all".format('{},{}/'.format(rs[0], rs[1]) if rs else '32,32/')
    file_train = saved_dir + "/train{}.csv".format('_ori' if ori else '')
    file_test = saved_dir + "/test{}.csv".format('_ori' if ori else '')

    df0 = pd.read_csv(file_train)
    df1 = pd.read_csv(file_test)
    df = pd.concat([df0, df1])
    df.to_csv(saved_dir + "/complete{}.csv".format('_ori' if ori else ''), index=False)

def ori_channel(trainX, trainy, testX, testy, rs=None):
    header = None
    splits = ['train', 'test']
    for split in splits:
        dt = trainX if split == 'train' else testX
        labels = trainy if split == 'train' else testy
        rows = []
        for i in range(dt.shape[0]):
            img = dt[i].copy()
            if rs:
                img = cv2.resize(img, rs)
            if header is None:
                header = []
                for c in range(img.shape[2]):
                    for x in range(img.shape[0]):
                        for y in range(img.shape[1]):
                            header.append("i_{0}_{1}_{2}".format(x, y, c))
                header.append('class')

            row = []
            # for c in range(img.shape[2]):
            for c in range(img.shape[2]):
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        row.append(img[x, y, c].item())
            row.append(labels[i].item())
            rows.append(row)

        saved_dir = "../datasets/cifar-10/{}all".format('{},{}/'.format(rs[0], rs[1]) if rs else '32,32/')
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        file = saved_dir + "/{0}_ori.csv".format('train' if split == 'train' else 'test')
        with open(file, "w") as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerows(rows)

def binarise(rs, cls, ori=False):
    """
        binarise classes
    """

    saved_dir = '../datasets/cifar-10/{}/all/'.format(rs)
    saved_dir_ = saved_dir.replace('/all/', '/{}/'.format(cls))
    if not os.path.isdir(saved_dir_):
        os.makedirs(saved_dir_)
    for split in ['complete', 'train', 'test']:
        file = saved_dir + split + '{}.csv'.format('_ori' if ori else '')
        df = pd.read_csv(file)
        cls0, cls1 = cls.split(',')
        df = df[(df.iloc[:, -1] == cls0) | (df.iloc[:, -1] == cls1)]
        df.to_csv(saved_dir_ + split + '{}.csv'.format('_ori' if ori else ''), index=False)

    """cls to 0,1 """
    saved_dir = '../datasets/cifar-10/{}/{}/'.format(rs, cls)
    cmd = 'python ./explain.py -p --pfiles complete{}.csv,complete{} {}/'.format('_ori' if ori else '',
                                                                                     '_ori' if ori else '',
                                                                                     saved_dir)
    #print(cmd)
    os.system(cmd)

    for split in ['train', 'test']:
        ori_file = '../datasets/cifar-10/{}/{}/{}{}.csv'.format(rs, cls, split, '_ori' if ori else '')
        data_file = ori_file.replace('.csv', '_data.csv')

        df0 = pd.read_csv(ori_file)

        dict0 = {}
        for i, v in enumerate(sorted(df0.iloc[:, -1].unique())):
            dict0[v] = i
        df0 = df0.replace({df0.columns[-1]: dict0})
        df0.to_csv(data_file, index=False)

        old_pkl = ori_file.replace('/' + split, '/complete').replace('.csv', '_data.csv.pkl')
        new_pkl = data_file + '.pkl'
        os.system('cp {} {}'.format(old_pkl, new_pkl))


if __name__ == '__main__':
    (trainX, trainy), (testX, testy) = cifar10.load_data()
    trainX = trainX/255
    testX = testX/255
    id2name = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
               4: 'deer', 5: 'dog', 6: 'frog', 7:'horse',
               8: 'ship', 9: 'truck'}
    trainy = np.squeeze(trainy)
    trainy = np.vectorize(id2name.get)(trainy)

    testy = np.squeeze(testy)
    testy = np.vectorize(id2name.get)(testy)

    resizes = [None]
    for rs in resizes:
        ori_channel(trainX, trainy, testX, testy, rs)
        comb(rs, ori=True)

    resizes = ['32,32']
    for rs in resizes:
        for cls in ['ship,truck']:
            binarise(rs, cls, ori=True)
    exit()

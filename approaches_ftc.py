import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from approaches import naive
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

from calibration_methods import BinningCalibration
from calibration_methods import IsotonicCalibration
from calibration_methods import BetaCalibration

from approaches import find_threshold


def ftc(dataset_name, feature, db_fold, nbins, calibration_method):

    if dataset_name == 'rfw':
        r = collect_error_embeddings_rfw(feature, db_fold['cal'])
        batch_size = 200
    elif 'bfw' in dataset_name:
        r = collect_error_embeddings_bfw(feature, db_fold['cal'])
        batch_size = 200
    elif 'ijbc' in dataset_name:
	# given that IJB-C is a larger dataset we increased the batch size
        r = collect_error_embeddings_ijbc(feature, db_fold['cal'])
        batch_size = 2048

    error_embeddings = r[0]
    ground_truth = r[1]
    subgroups_left = r[2]
    subgroups_right = r[3]
    train_dataloader = DataLoader(
        EmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    evaluate_train_dataloader = DataLoader(
        EmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    if dataset_name == 'rfw':
        r = collect_error_embeddings_rfw(feature, db_fold['test'])
    elif 'bfw' in dataset_name:
        r = collect_error_embeddings_bfw(feature, db_fold['test'])
    elif 'ijbc' in dataset_name:
        r = collect_error_embeddings_ijbc(feature, db_fold['test'])
    error_embeddings = r[0]
    ground_truth = r[1]
    subgroups_left = r[2]
    subgroups_right = r[3]

    evaluate_test_dataloader = DataLoader(
        EmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    # Initialize model
    model = NeuralNetwork()
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    if 'ijbc' in dataset_name:
        epochs = 10
    else:
        epochs = 50
        
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, dataset_name)
#         _, _ = test_loop(evaluate_test_dataloader, model, loss_fn)
    print("Done!")
    scores_cal, ground_truth_cal = test_loop(evaluate_train_dataloader, model, loss_fn)
    scores_cal = scores_cal[:, 1].numpy().reshape(-1)
    assert sum(np.array(ground_truth_cal == 1) != np.array(db_fold['cal']['same'])) == 0

    scores_test, ground_truth_test = test_loop(evaluate_test_dataloader, model, loss_fn)
    scores_test = scores_test[:, 1].numpy().reshape(-1)
    assert sum(np.array(ground_truth_test == 1) != np.array(db_fold['test']['same'])) == 0

    fair_scores = {'cal': scores_cal, 'test': scores_test}
    ground_truth = {'cal': ground_truth_cal, 'test': ground_truth_test}

    confidences = baseline(fair_scores, ground_truth, nbins, calibration_method, score_min=-1, score_max=1)

    return fair_scores, confidences, model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128*4, 256*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256*4, 512*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512*4, 512*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512*4, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        prob = self.softmax(logits)
        return logits, prob


def collect_error_embeddings_rfw(feature, db_cal):
    if feature != 'arcface':
        embeddings = {'left': torch.zeros((len(db_cal), 512)), 'right': torch.zeros((len(db_cal), 512))}
        ground_truth = np.array(db_cal['same']).astype(bool)
        subgroups = np.array(db_cal['ethnicity'])
        subgroup_old = ''
        for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
            folder_names = db_cal[id_face].values
            file_names = db_cal[id_face] + '_000' + db_cal[num_face].astype(str) + '.jpg'
            file_names = file_names.values
            i = 0
            for folder_name, file_name, subgroup in zip(folder_names, file_names, subgroups):
                if subgroup_old != subgroup:
                    temp = pickle.load(
                        open('raw_data/embeddings/rfw/' + subgroup + '_' + feature + '_embeddings.pickle', 'rb'))
                subgroup_old = subgroup

                key = 'rfw/data/' + subgroup + '_cropped/' + folder_name + '/' + file_name
                if id_face == 'id1':
                    embeddings['left'][i, :] = temp[key]
                elif id_face == 'id2':
                    embeddings['right'][i, :] = temp[key]
                i += 1
        difference_embeddings = embeddings['left'] - embeddings['right']
        error_embeddings = torch.abs(difference_embeddings)
    else:
        temp = pickle.load(open('raw_data/embeddings/rfw/rfw_' + feature + '_embeddings.pickle', 'rb'))
        embeddings = {'left': torch.zeros((len(db_cal), 512)), 'right': torch.zeros((len(db_cal), 512))}
        ground_truth = np.array(db_cal['same']).astype(bool)
        subgroups = np.array(db_cal['ethnicity'])
        for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
            folder_names = db_cal[id_face].values
            file_names = db_cal[id_face] + '_000' + db_cal[num_face].astype(str) + '.jpg'
            file_names = file_names.values
            i = 0
            for folder_name, file_name, subgroup in zip(folder_names, file_names, subgroups):
                key = 'rfw/data/' + subgroup + '/' + folder_name + '/' + file_name
                if id_face == 'id1':
                    embeddings['left'][i, :] =  torch.from_numpy(temp[key].reshape(1, -1))
                elif id_face == 'id2':
                    embeddings['right'][i, :] = torch.from_numpy(temp[key].reshape(1, -1))
                i += 1
        difference_embeddings = embeddings['left'] - embeddings['right']
        error_embeddings = torch.abs(difference_embeddings)
    return error_embeddings, ground_truth, subgroups, subgroups


def collect_error_embeddings_bfw(feature, db_cal):
    # collect embeddings of all the images in the calibration set
    embeddings = {'left': torch.zeros((len(db_cal), 512)), 'right': torch.zeros((len(db_cal), 512))}
    ground_truth = np.array(db_cal['same'].astype(bool))
    subgroups_left = np.array(db_cal['att1'])
    subgroups_right = np.array(db_cal['att2'])
    temp = pickle.load(open('raw_data/embeddings/bfw/' + feature + '_embeddings.pickle', 'rb'))
    for path in ['path1', 'path2']:
        file_names = db_cal[path].values
        for i, file_name in enumerate(file_names):
            if path == 'path1':
                if feature == 'arcface':
                    embeddings['left'][i, :] = torch.from_numpy(temp[file_name])
                else:
                    embeddings['left'][i, :] = temp[file_name]
            elif path == 'path2':
                if feature == 'arcface':
                    embeddings['right'][i, :] = torch.from_numpy(temp[file_name])
                else:
                    embeddings['right'][i, :] = temp[file_name]
    difference_embeddings = embeddings['left'] - embeddings['right']
    error_embeddings = torch.abs(difference_embeddings)
    return error_embeddings, ground_truth, subgroups_left, subgroups_right


def collect_error_embeddings_ijbc(feature, db_cal):
    # collect embeddings of all the images in the calibration set
#     print('Loading IJB-C')
    ground_truth = np.array(db_cal['same'].astype(bool))
    subgroups_left = np.array(db_cal['att1'])
    subgroups_right = np.array(db_cal['att2'])
    temp = np.load('raw_data/embeddings/ijbc/' + feature + '_embeddings.npy')
    npairs = len(db_cal)
    error_embeddings = torch.zeros((npairs, temp.shape[1]))
    start = 0
#     print(npairs//10000+1)
    for i in range(npairs//10000+1):
#         print(i)
        if start>npairs:
            break
        end = min(start+10000,npairs)
        aux_left = torch.from_numpy(temp[db_cal['MATRIX_ID_1'][start:end]])
#         print('aux_left')
        aux_right = torch.from_numpy(temp[db_cal['MATRIX_ID_2'][start:end]])
#         print('aux_right')
        error_embeddings[start:end] = torch.abs(aux_left-aux_right)
    return error_embeddings, ground_truth, subgroups_left, subgroups_right

class EmbeddingsDataset(Dataset):
    """Embeddings dataset."""

    def __init__(self, error_embeddings, ground_truth, subgroups_left, subgroups_right):
        """
        Arguments
        """
        self.subgroups_left = subgroups_left
        self.subgroups_right = subgroups_right
        self.error_embeddings = error_embeddings
        self.labels = torch.zeros(len(error_embeddings)).type(torch.LongTensor)
        self.labels[ground_truth] = 1

    def __len__(self):
        return len(self.error_embeddings)

    def __getitem__(self, idx):
        return self.error_embeddings[idx, :], self.subgroups_left[idx], self.subgroups_right[idx], self.labels[idx]


def fair_individual_loss(g1, g2, y, yhat, dataset_name):
    if dataset_name == 'rfw':
        subgroups = ['Asian', 'African', 'Caucasian', 'Indian']
    elif 'bfw' in dataset_name:
        subgroups = ['asian_females', 'asian_males', 'black_females', 'black_males', 'indian_females', 'indian_males',
                     'white_females', 'white_males']
    elif 'ijbc' in dataset_name:
        subgroups = ['dark_brown_female', 'dark_brown_male', 'light_pink_female',
           'light_pink_male', 'light_yellow_female', 'light_yellow_male',
           'medium_dark_brown_female', 'medium_dark_brown_male',
           'medium_pink_brown_female', 'medium_pink_brown_male',
           'medium_yellow_brown_female', 'medium_yellow_brown_male']
    else:
        subgroups = None
    loss = 0
    for i in subgroups:
        for j in subgroups:
            select_i = np.logical_and(np.array(g1) == i, np.array(g2) == i)
            select_j = np.logical_and(np.array(g1) == j, np.array(g2) == j)
            if (sum(select_i) > 0) and (sum(select_j) > 0):
                select = y[select_i].reshape(-1, 1) == y[select_j]
                aux = torch.cdist(yhat[select_i, :], yhat[select_j, :])[select].pow(2).sum()
                loss += aux/(sum(select_i)*sum(select_j))
    return loss


def train_loop(dataloader, model, loss_fn, optimizer, dataset_name):
    if dataset_name == 'rfw':
        batch_check = 50
    elif 'bfw' in dataset_name:
        batch_check = 500
    elif 'ijbc' in dataset_name:
        batch_check = 50
    model.cuda()
    size = len(dataloader.dataset)
    for batch, (X, g1, g2, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # Compute prediction and loss
        pred, prob = model(X)
        loss = 0.5*loss_fn(pred, y)+0.5*fair_individual_loss(g1, g2, y, pred, dataset_name)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_check == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    model.cpu()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    scores = torch.zeros(0, 2)
    ground_truth = torch.zeros(0)
    model.cuda()
    with torch.no_grad():
        for X, g1, g2, y in dataloader:
            X = X.cuda()
            y = y.cuda()
            pred, prob = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            scores = torch.cat((scores, prob.cpu()))
            ground_truth = torch.cat([ground_truth, y.cpu()], 0)
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    fpr, tpr, thr = roc_curve(ground_truth, scores[:, 1].numpy())
    print('FNR @ 0.1 FPR %1.2f'% (1-tpr[np.argmin(np.abs(fpr-1e-3))]))
    return scores, ground_truth
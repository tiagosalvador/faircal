import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from approaches_final import baseline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from calibration_methods import BinningCalibration
from calibration_methods import SplinesCalibration
from calibration_methods import IsotonicCalibration
from calibration_methods import BetaCalibration

from approaches_final import find_threshold


def agenda(dataset_name, feature, db_fold, nbins, calibration_method):
    
    if dataset_name == 'rfw':
        embeddings, subgroup_embeddings, id_embeddings = collect_embeddings_rfw_agenda(feature, db_fold['cal'])
    elif 'bfw' in dataset_name:
        embeddings, subgroup_embeddings, id_embeddings = collect_embeddings_bfw_agenda(feature, db_fold['cal'])
    elif 'ijbc' in dataset_name:
        embeddings, subgroup_embeddings, id_embeddings = collect_embeddings_ijbc_agenda(feature, db_fold['cal'])

    
    subgroup_embeddings = pd.Series(subgroup_embeddings, dtype="category").cat.codes.values
    embeddings_train, embeddings_test, id_train, id_test, subgroup_train, subgroup_test \
        = train_test_split(embeddings,id_embeddings,subgroup_embeddings, test_size=0.2)

    id_train = pd.Series(id_train, dtype="category").cat.codes.values

    train_dataloader = DataLoader(
            EmbeddingsDataset(embeddings_train, id_train, subgroup_train),
            batch_size=400,
            shuffle=True,
            num_workers=0)

    test_dataloader = DataLoader(
            EmbeddingsDataset(embeddings_test, id_test, subgroup_test),
            batch_size=400,
            shuffle=True,
            num_workers=0)

    n_id = len(np.unique(id_train))
    n_subgroup = len(np.unique(subgroup_train))

    
    Nep = 100
    Tep = 10
    epochs_stage1 = 50
    epochs_stage2 = 25
    epochs_stage3 = 5
    epochs_stage4 = 5

    loss_fn = nn.CrossEntropyLoss()
    # Initialize 
    modelM = NeuralNetworkM().cuda()
    modelC = NeuralNetworkC(n_id).cuda()

    optimizer_stage1 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-3)
    ## STAGE 1 ##  
    for epoch in tqdm(range(epochs_stage1)):
        if torch.cuda.is_available():
            modelM.train()
            modelC.train()
        for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                X = X.cuda()
                y_id = y_id.cuda()
            # Compute prediction and loss
            prob = modelM(X.float())
            prob = modelC(prob)
            loss = loss_fn(prob,y_id.long())

            # Backpropagation
            optimizer_stage1.zero_grad()
            loss.backward()
            optimizer_stage1.step()

    for i in tqdm(range(Nep)):
        ## STAGE 2 ##
        if i % Tep == 0:
            if torch.cuda.is_available():
                modelE = NeuralNetworkE(n_subgroup).cuda()
            optimizer_stage2 = optim.Adam(modelE.parameters(), lr=1e-3)
            for epoch in range(epochs_stage2):
                for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_subgroup = y_subgroup.cuda()                
                    prob = modelM(X.float())
                    prob = modelE(prob)
                    loss = loss_fn(prob,y_subgroup.long())

                    # Backpropagation
                    optimizer_stage2.zero_grad()
                    loss.backward()
                    optimizer_stage2.step()

        ## STAGE 3 ##
        optimizer_stage3 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-3)
        for epoch in range(epochs_stage3):
            for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_id = y_id.cuda()
                    y_subgroup = y_subgroup.cuda()
                f_out = modelM(X.float())
                prob_class = modelC(f_out)
                prob_subgroup = modelE(f_out)

                loss_class = loss_fn(prob_class,y_id.long())

                loss_deb = -torch.log(prob_subgroup)/prob_subgroup.shape[1]
                loss_deb = loss_deb.sum(axis=1).mean()

                loss = loss_class+10*loss_deb

                # Backpropagation
                optimizer_stage3.zero_grad()
                loss.backward()
                optimizer_stage3.step()

        ## STAGE 4 ##

        optimizer_stage2 = optim.Adam(modelE.parameters(), lr=1e-3)
        for epoch in range(epochs_stage4):
            modelM.eval()
            modelE.eval()
            size = len(test_dataloader.dataset)
            test_loss, correct = 0, 0

            scores = torch.zeros(0, 2)
            ground_truth = torch.zeros(0)
            with torch.no_grad():
                for X, y_id, y_subgroup in test_dataloader:
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_subgroup = y_subgroup.cuda()
                    prob = modelM(X.float())
                    prob = modelE(prob)
                    test_loss += loss_fn(prob,y_subgroup.long()).item()
                    correct += (prob.argmax(1) == y_subgroup).type(torch.float).sum().item()
            test_loss /= size
            correct /= size

            if correct > 0.9:
                break
            for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_id = y_id.cuda()
                    y_subgroup = y_subgroup.cuda()                
                prob = modelM(X.float())
                prob = modelE(prob)
                loss = loss_fn(prob,y_subgroup.long())

                # Backpropagation
                optimizer_stage2.zero_grad()
                loss.backward()
                optimizer_stage2.step()
    
    
    fair_scores = {}
    ground_truth = {}
    for dataset in ['cal', 'test']:
        if 'ijbc' in dataset_name:
            fair_scores[dataset], ground_truth[dataset] = collect_pair_embeddings_ijbc(feature, db_fold[dataset], modelM)
        else:
            if dataset_name == 'rfw':
                embeddings, ground_truth[dataset], subgroups_left, subgroups_right = collect_pair_embeddings_rfw(feature, db_fold[dataset])
            elif 'bfw' in dataset_name:
                embeddings, ground_truth[dataset], subgroups_left, subgroups_right = collect_pair_embeddings_bfw(feature, db_fold[dataset])
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            modelM.eval()
            modelM.cpu()
            with torch.no_grad():
                temp1 = modelM(embeddings['left'])
                temp2 = modelM(embeddings['right'])
            output = cos(temp1, temp2)
            fair_scores[dataset] = output.numpy()
    
    confidences = baseline(fair_scores, ground_truth, nbins, calibration_method, score_min=-1, score_max=1)
    
    return fair_scores, confidences, modelM, modelC, modelE


class EmbeddingsDataset(Dataset):
    """Embeddings dataset."""

    def __init__(self, embeddings, id_embeddings, subgroup_embeddings):
        """
        Arguments
        """
        self.embeddings = embeddings
        self.id_embeddings = id_embeddings
        self.subgroup_embeddings = subgroup_embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx, :], self.id_embeddings[idx], self.subgroup_embeddings[idx]



class NeuralNetworkM(nn.Module):
    def __init__(self):
        super(NeuralNetworkM, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)

class NeuralNetworkC(nn.Module):
    def __init__(self,nClasses):
        super(NeuralNetworkC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, nClasses)
        )

    def forward(self, x):
        return self.model(x)

class NeuralNetworkE(nn.Module):
    def __init__(self,nClasses):
        super(NeuralNetworkE, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128,nClasses),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


def collect_embeddings_rfw_agenda(feature, db_cal):
    # collect embeddings of all the images in the calibration set
    embeddings = np.zeros((0, 512))  # all embeddings are in a 512-dimensional space
    faces_id_num = []
    subgroup_embeddings = []
    id_embeddings = []
    if feature != 'arcface':
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            temp = pickle.load(open('raw_data/embeddings/rfw/' + subgroup + '_' + feature + '_embeddings.pickle', 'rb'))
            select = db_cal['ethnicity'] == subgroup

            for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                folder_names = db_cal[select][id_face].values
                file_names = db_cal[select][id_face] + '_000' + db_cal[select][num_face].astype(str) + '.jpg'
                file_names = file_names.values
                for folder_name, file_name, id_embedding in zip(folder_names, file_names, db_cal[select][id_face].values):
                    key = 'rfw/data/' + subgroup + '_cropped/' + folder_name + '/' + file_name
                    if file_name not in faces_id_num:
                        embeddings = np.concatenate((embeddings, temp[key]))
                        faces_id_num.append(file_name)
                        subgroup_embeddings.append(subgroup)
                        id_embeddings.append(id_embedding)
    else:
        temp = pickle.load(open('raw_data/embeddings/rfw/rfw_' + feature + '_embeddings.pickle', 'rb'))
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            select = db_cal['ethnicity'] == subgroup

            for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                folder_names = db_cal[select][id_face].values
                file_names = db_cal[select][id_face] + '_000' + db_cal[select][num_face].astype(str) + '.jpg'
                file_names = file_names.values
                for folder_name, file_name, id_embedding in zip(folder_names, file_names, db_cal[select][id_face].values):
                    key = 'rfw/data/' + subgroup + '/' + folder_name + '/' + file_name
                    if file_name not in faces_id_num:
                        embeddings = np.concatenate((embeddings, temp[key].reshape(1, -1)))
                        faces_id_num.append(file_name)
                        subgroup_embeddings.append(subgroup)
                        id_embeddings.append(id_embedding)
    subgroup_embeddings = np.array(subgroup_embeddings)
    id_embeddings = np.array(id_embeddings)
    return embeddings, subgroup_embeddings, id_embeddings


def collect_embeddings_bfw_agenda(feature, db_cal):
    # collect embeddings of all the images in the calibration set
    embeddings = np.zeros((0, 512))  # all embeddings are in a 512-dimensional space
    subgroup_embeddings = []
    id_embeddings = []

    file_names_visited = []
    temp = pickle.load(open('raw_data/embeddings/bfw/' + feature + '_embeddings.pickle', 'rb'))
    for path, att, id_name in zip(['path1', 'path2'],['att1', 'att2'], ['id1', 'id2']):
        file_names = db_cal[path].values
        subgroups = db_cal[att].values
        ids = db_cal[id_name].values
        for file_name, subgroup, id_embedding in zip(file_names, subgroups, ids):
            if file_name not in file_names_visited:
                embeddings = np.concatenate((embeddings, temp[file_name].reshape(1, -1)))
                file_names_visited.append(file_name)
                subgroup_embeddings.append(subgroup)
                id_embeddings.append(id_embedding)

    return embeddings, subgroup_embeddings, id_embeddings

def collect_embeddings_ijbc_agenda(feature, db_cal):
    # collect embeddings of all the images in the calibration set
    
    index_embeddings = np.concatenate((db_cal['MATRIX_ID_1'],db_cal['MATRIX_ID_2']))
    id_embeddings = np.concatenate((db_cal['ID_1'],db_cal['ID_2']))
    subgroup_embeddings = np.concatenate((db_cal['att1'],db_cal['att2']))
    index_embeddings,select = np.unique(index_embeddings, return_index=True)
    id_embeddings = id_embeddings[select]
    subgroup_embeddings = subgroup_embeddings[select]
    
    temp = np.load('raw_data/embeddings/ijbc/' + feature + '_embeddings.npy')
    embeddings = temp[index_embeddings]
    
    return embeddings, subgroup_embeddings, id_embeddings


def collect_pair_embeddings_rfw(feature, db_cal):
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
    return embeddings, ground_truth, subgroups, subgroups


def collect_pair_embeddings_bfw(feature, db_cal):
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
    return embeddings, ground_truth, subgroups_left, subgroups_right


def collect_pair_embeddings_ijbc(feature, db_cal, modelM):
    ground_truth = np.array(db_cal['same'].astype(bool))
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    modelM.eval()
    modelM.cuda()
    npairs = len(db_cal)
    temp = np.load('raw_data/embeddings/ijbc/' + feature + '_embeddings.npy')
    fair_scores = np.zeros(npairs)
    with torch.no_grad():
        start = 0
        for i in range(npairs//100+1):
            if start>npairs:
                break
            end = min(start+100,npairs)
            temp1 = modelM(torch.from_numpy(temp[db_cal['MATRIX_ID_1'][start:end]]).cuda().float())
            temp2 = modelM(torch.from_numpy(temp[db_cal['MATRIX_ID_2'][start:end]]).cuda().float())
            fair_scores[start:end] = cos(temp1, temp2).cpu().numpy()
            start = start+100

    return fair_scores, ground_truth

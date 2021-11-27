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

from approaches_agenda import collect_embeddings_rfw_agenda, collect_embeddings_bfw_agenda, collect_embeddings_ijbc_agenda
from approaches_agenda import collect_pair_embeddings_rfw, collect_pair_embeddings_bfw, collect_pair_embeddings_ijbc


def pass_att(dataset_name, feature, db_fold, nbins, calibration_method):
    
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
    id_test = pd.Series(id_test, dtype="category").cat.codes.values

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
    epochs_stage1 = 100
    epochs_stage2 = 100
    epochs_stage3 = 5
    epochs_stage4 = 5
    K = 2

    loss_fn = nn.CrossEntropyLoss()
    # Initialize 
    modelM = NeuralNetworkM().cuda()
    modelC = NeuralNetworkC(n_id).cuda()

    ## STAGE 1 ##  
    # Initialize 
    modelM = NeuralNetworkM().cuda()
    modelC = NeuralNetworkC(n_id).cuda()
    optimizer_stage1 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-2)

    for epoch in tqdm(range(epochs_stage1)):
        if torch.cuda.is_available():
            modelM.train()
            modelC.train()
        loss_list = []
        for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                X = X.cuda()
                y_id = y_id.cuda()
                y_subgroup = y_subgroup.cuda()
            # Compute prediction and loss
            prob = modelM(X.float())
            prob = modelC(prob)
            loss = loss_fn(prob,y_id.long())
            loss_list.append(loss)

            # Backpropagation
            optimizer_stage1.zero_grad()
            loss.backward()
            optimizer_stage1.step()


    for i in tqdm(range(Nep)):
        ## STAGE 2 ##
        if i % Tep == 0:
            if torch.cuda.is_available():
                modelE = {}

                for k in range(K):
                    modelE[k] = NeuralNetworkE(n_subgroup).cuda()
            optimizer_stage2_parameters = list(modelE[0].parameters())
            for k in range(1,K):
                optimizer_stage2_parameters += list(modelE[k].parameters())
            optimizer_stage2 = optim.Adam(optimizer_stage2_parameters, lr=1e-3)
            for epoch in tqdm(range(epochs_stage2)):
                loss_list = []
                for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_id = y_id.cuda()
                        y_subgroup = y_subgroup.cuda()                
                    prob = modelM(X.float())
                    loss = 0.0
                    for k in range(K):
                        loss += loss_fn(modelE[k](prob),y_subgroup.long())
                    loss_list.append(loss)

                    # Backpropagation
                    optimizer_stage2.zero_grad()
                    loss.backward()
                    optimizer_stage2.step()

        ## STAGE 3 ##
        optimizer_stage3 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-4)
        for epoch in range(epochs_stage3):
            loss_list = []
            for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_id = y_id.cuda()
                    y_subgroup = y_subgroup.cuda()
                f_out = modelM(X.float())
                prob_class = modelC(f_out)
                loss_class = loss_fn(prob_class,y_id.long())

                loss_deb_list = []
                for k in range(K):
                    prob_subgroup = modelE[k](f_out)
                    loss_deb = -torch.log(prob_subgroup)/prob_subgroup.shape[1]
                    loss_deb = loss_deb.sum(axis=1).mean()
                    loss_deb_list.append(loss_deb)

                loss = loss_class+10*max(loss_deb_list)
                loss_list.append(loss)
                # Backpropagation
                optimizer_stage3.zero_grad()
                loss.backward()
                optimizer_stage3.step()

        ## STAGE 4 ##
        k = i % K
        optimizer_stage2 = optim.Adam(modelE[k].parameters(), lr=1e-3)
        for epoch in range(epochs_stage4):
            modelM.eval()
            modelE[k].eval()
            size = len(test_dataloader.dataset)
            test_loss, correct = 0, 0

            scores = torch.zeros(0, 2)
            ground_truth = torch.zeros(0)
            with torch.no_grad():
                for X, y_id, y_subgroup in test_dataloader:
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_id = y_id.cuda()
                        y_subgroup = y_subgroup.cuda()
                    prob = modelM(X.float())
                    prob = modelE[k](prob)
                    test_loss += loss_fn(prob,y_subgroup.long()).item()
                    correct += (prob.argmax(1) == y_subgroup).type(torch.float).sum().item()
            test_loss /= size
            correct /= size
            modelM.train()
            modelE[k].train()
            if correct > 0.95:
                break
            for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_id = y_id.cuda()
                    y_subgroup = y_subgroup.cuda()                
                prob = modelM(X.float())
                prob = modelE[k](prob)
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
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, nClasses),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
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

from approaches_agenda import collect_pair_embeddings_rfw, collect_pair_embeddings_bfw, collect_pair_embeddings_ijbc


def multi_pass(dataset_name, feature, db_fold, nbins, calibration_method):

    if dataset_name == 'rfw':
        raise Exception('MultiPASS algorithm not available for the RFW dataset.')
    elif 'bfw' in dataset_name:
        embeddings, subgroup_gender_embeddings, subgroup_race_embeddings, subgroup_embeddings, id_embeddings = collect_embeddings_bfw_multipass(feature, db_fold['cal'])
    elif 'ijbc' in dataset_name:
        embeddings, subgroup_gender_embeddings, subgroup_race_embeddings, subgroup_embeddings, id_embeddings = collect_embeddings_ijbc_multipass(feature, db_fold['cal'])
    
    
    subgroup_gender_embeddings = pd.Series(subgroup_gender_embeddings, dtype="category").cat.codes.values
    subgroup_race_embeddings = pd.Series(subgroup_race_embeddings, dtype="category").cat.codes.values
    subgroup_embeddings = pd.Series(subgroup_embeddings, dtype="category").cat.codes.values
    embeddings_train, embeddings_test, id_train, id_test, subgroup_gender_train, subgroup_gender_test, \
        subgroup_race_train, subgroup_race_test, subgroup_train, subgroup_test = train_test_split(
            embeddings,id_embeddings,subgroup_gender_embeddings,subgroup_race_embeddings,subgroup_embeddings, test_size=0.2)

    id_train = pd.Series(id_train, dtype="category").cat.codes.values
    id_test = pd.Series(id_test, dtype="category").cat.codes.values

    train_dataloader = DataLoader(
                EmbeddingsDataset(embeddings_train, id_train, subgroup_gender_train, subgroup_race_train, subgroup_train),
                batch_size=400,
                shuffle=True,
                num_workers=0)

    test_dataloader = DataLoader(
            EmbeddingsDataset(embeddings_test, id_test, subgroup_gender_test, subgroup_race_test, subgroup_test),
            batch_size=400,
            shuffle=True,
            num_workers=0)

    n_id = len(np.unique(id_train))
    n_subgroup_gender = len(np.unique(subgroup_gender_train))
    n_subgroup_race = len(np.unique(subgroup_race_train))
    n_subgroup = len(np.unique(subgroup_train))
    

    Nep = 100
    Tep = 10
    epochs_stage1 = 100
    epochs_stage2 = 100
    epochs_stage3 = 5
    epochs_stage4 = 5
    K_gender = 3
    K_race = 2

    loss_fn = nn.CrossEntropyLoss()

    print(f"STAGE 1")
    # Initialize 
    modelM = NeuralNetworkM().cuda()
    modelC = NeuralNetworkC(n_id).cuda()
    optimizer_stage1 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-3)

    for epoch in tqdm(range(epochs_stage1)):
        if torch.cuda.is_available():
            modelM.train()
            modelC.train()
        loss_list = []
        for batch, (X, y_id, _, _, _) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                X = X.cuda()
                y_id = y_id.cuda()
            # Compute prediction and loss
            prob = modelM(X.float())
            prob = modelC(prob)
            loss = loss_fn(prob,y_id.long())
            loss_list.append(loss)

            # Backpropagation
            optimizer_stage1.zero_grad()
            loss.backward()
            optimizer_stage1.step()
        if epoch % 10 == 0:
            tqdm.write('Epoch %d: Loss: %1.2f'%(epoch,sum(loss_list)/len(loss_list)))


    for i in tqdm(range(Nep)):
        ## STAGE 2 ##
        if i % Tep == 0:
            if torch.cuda.is_available():
                modelE_gender = {}
                for k in range(K_gender):
                    modelE_gender[k] = NeuralNetworkE(n_subgroup_gender).cuda()
                modelE_race = {}
                for k in range(K_race):
                    modelE_race[k] = NeuralNetworkE(n_subgroup_race).cuda()
            optimizer_stage2_parameters_gender = list(modelE_gender[0].parameters())
            for k in range(1,K_gender):
                optimizer_stage2_parameters_gender += list(modelE_gender[k].parameters())
            optimizer_stage2_gender = optim.Adam(optimizer_stage2_parameters_gender, lr=1e-3)
            optimizer_stage2_parameters_race = list(modelE_race[0].parameters())
            for k in range(1,K_race):
                optimizer_stage2_parameters_race += list(modelE_race[k].parameters())
            optimizer_stage2_race = optim.Adam(optimizer_stage2_parameters_race, lr=1e-3)
            print(f"STAGE 2")
            for epoch in tqdm(range(epochs_stage2)):
                loss_list_gender = []
                loss_list_race = []
                for batch, (X, y_id, y_subgroup_gender, y_subgroup_race, y_subgroup) in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_id = y_id.cuda()
                        y_subgroup_gender = y_subgroup_gender.cuda()
                        y_subgroup_race = y_subgroup_race.cuda()
                        y_subgroup = y_subgroup.cuda()
                    prob = modelM(X.float())
                    loss_gender = 0.0
                    for k in range(K_gender):
                        loss_gender += loss_fn(modelE_gender[k](prob),y_subgroup_gender.long())
                    loss_list_gender.append(loss_gender)
                    optimizer_stage2_gender.zero_grad()
                    loss_gender.backward()
                    optimizer_stage2_gender.step()

                    prob = modelM(X.float())
                    loss_race = 0.0
                    for k in range(K_race):
                        loss_race += loss_fn(modelE_race[k](prob),y_subgroup_race.long())
                    loss_list_race.append(loss_race)
                    optimizer_stage2_race.zero_grad()
                    loss_race.backward()
                    optimizer_stage2_race.step()
                if epoch % 10 == 0:
                    tqdm.write('Epoch %d: Loss (gender): %1.2f Loss (race): %1.2f'%(epoch,sum(loss_list_gender)/len(loss_list_gender),sum(loss_list_race)/len(loss_list_race)))

        ## STAGE 3 ##
        optimizer_stage3 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-4)
        #     print(f"STAGE 3")
        for epoch in range(epochs_stage3):
            loss_list = []
            for batch, (X, y_id, y_subgroup_gender, y_subgroup_race, y_subgroup) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_id = y_id.cuda()
                    y_subgroup_gender = y_subgroup_gender.cuda()
                    y_subgroup_race = y_subgroup_race.cuda()
                    y_subgroup = y_subgroup.cuda()
                f_out = modelM(X.float())
                prob_class = modelC(f_out)
                loss_class = loss_fn(prob_class,y_id.long())

                loss_deb_list_gender = []
                for k in range(K_gender):
                    prob_subgroup = modelE_gender[k](f_out)
                    loss_deb = -torch.log(prob_subgroup)/prob_subgroup.shape[1]
                    loss_deb = loss_deb.sum(axis=1).mean()
                    loss_deb_list_gender.append(loss_deb)
                loss_deb_list_race = []
                for k in range(K_race):
                    prob_subgroup = modelE_race[k](f_out)
                    loss_deb = -torch.log(prob_subgroup)/prob_subgroup.shape[1]
                    loss_deb = loss_deb.sum(axis=1).mean()
                    loss_deb_list_race.append(loss_deb)

                loss = loss_class+10*max(loss_deb_list_gender)+10*max(loss_deb_list_race)
                loss_list.append(loss)
                # Backpropagation
                optimizer_stage3.zero_grad()
                loss.backward()
                optimizer_stage3.step()
            if epoch % 1 == 0:
                tqdm.write('Epoch %d: Loss: %1.2f'%(epoch,sum(loss_list)/len(loss_list)))

        ## STAGE 4 ##
        k_gender = i % K_gender
        k_race = i % K_race
        optimizer_stage2_gender = optim.Adam(modelE_gender[k_gender].parameters(), lr=1e-4)
        optimizer_stage2_race = optim.Adam(modelE_race[k_race].parameters(), lr=1e-4)
    #     print(f"STAGE 4")
        for epoch in range(epochs_stage4):
            modelM.eval()
            modelE_gender[k_gender].eval()
            modelE_race[k_race].eval()
            size = len(test_dataloader.dataset)


            test_loss_gender, correct_gender = 0, 0
            with torch.no_grad():
                for X, y_id, y_subgroup_gender, _, _ in test_dataloader:
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_id = y_id.cuda()
                        y_subgroup_gender = y_subgroup_gender.cuda()
                    prob = modelM(X.float())
                    prob = modelE_gender[k_gender](prob)
                    test_loss_gender += loss_fn(prob,y_subgroup_gender.long()).item()
                    correct_gender += (prob.argmax(1) == y_subgroup_gender).type(torch.float).sum().item()
            test_loss_gender /= size
            correct_gender /= size
            test_loss_race, correct_race = 0, 0
            with torch.no_grad():
                for X, y_id, _, y_subgroup_race, _ in test_dataloader:
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_id = y_id.cuda()
                        y_subgroup_race = y_subgroup_race.cuda()
                    prob = modelM(X.float())
                    prob = modelE_race[k_race](prob)
                    test_loss_race += loss_fn(prob,y_subgroup_race.long()).item()
                    correct_race += (prob.argmax(1) == y_subgroup_race).type(torch.float).sum().item()
            test_loss_race /= size
            correct_race /= size
    #         print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

            modelM.train()
            modelE_gender[k_gender].train()
            modelE_race[k_race].train()

            if (correct_gender > 0.95) and (correct_race > 0.95):
                break
            for batch, (X, y_id, y_subgroup_gender, y_subgroup_race, y_subgroup) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_id = y_id.cuda()
                    y_subgroup_gender = y_subgroup_gender.cuda()
                    y_subgroup_race = y_subgroup_race.cuda()
                    y_subgroup = y_subgroup.cuda()
                prob = modelM(X.float())
                loss_gender = loss_fn(modelE_gender[k_gender](prob),y_subgroup_gender.long())
                optimizer_stage2_gender.zero_grad()
                loss_gender.backward()
                optimizer_stage2_gender.step()
                prob = modelM(X.float())
                loss_race = loss_fn(modelE_race[k_race](prob),y_subgroup_race.long())
                optimizer_stage2_race.zero_grad()
                loss_race.backward()
                optimizer_stage2_race.step()
    
    fair_scores = {}
    ground_truth = {}
    for dataset in ['cal', 'test']:
        print(dataset)
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
    
    return fair_scores, confidences, modelM, modelC, modelE_gender, modelE_race


def collect_embeddings_bfw_multipass(feature, db_cal):
    # collect embeddings of all the images in the calibration set
    embeddings = np.zeros((0, 512))  # all embeddings are in a 512-dimensional space
    subgroup_gender_embeddings = []
    subgroup_race_embeddings = []
    subgroup_embeddings = []
    id_embeddings = []

    file_names_visited = []
    temp = pickle.load(open('raw_data/embeddings/bfw/' + feature + '_embeddings.pickle', 'rb'))
    for path, g, e, att, id_name in zip(['path1', 'path2'],['g1', 'g2'], ['e1', 'e2'], ['att1', 'att2'], ['id1', 'id2']):
        file_names = db_cal[path].values
        subgroups_gender = db_cal[g].values
        subgroups_race = db_cal[e].values
        subgroups = db_cal[att].values
        ids = db_cal[id_name].values
        for file_name, subgroup_gender, subgroup_race, subgroup, id_embedding in zip(file_names, subgroups_gender, subgroups_race, subgroups, ids):
            if file_name not in file_names_visited:
                embeddings = np.concatenate((embeddings, temp[file_name].reshape(1, -1)))
                file_names_visited.append(file_name)
                subgroup_gender_embeddings.append(subgroup_gender)
                subgroup_race_embeddings.append(subgroup_race)
                subgroup_embeddings.append(subgroup)
                id_embeddings.append(id_embedding)

    return embeddings, subgroup_gender_embeddings, subgroup_race_embeddings, subgroup_embeddings, id_embeddings


def collect_embeddings_ijbc_multipass(feature, db_cal):
    
    index_embeddings = np.concatenate((db_cal['MATRIX_ID_1'],db_cal['MATRIX_ID_2']))
    id_embeddings = np.concatenate((db_cal['ID_1'],db_cal['ID_2']))
    subgroup_embeddings = np.concatenate((db_cal['att1'],db_cal['att2']))
    subgroup_gender_embeddings = np.concatenate((db_cal['g1'],db_cal['g2']))
    subgroup_race_embeddings = np.concatenate((db_cal['e1'],db_cal['e2']))
    index_embeddings,select = np.unique(index_embeddings, return_index=True)
    id_embeddings = id_embeddings[select]
    subgroup_embeddings = subgroup_embeddings[select]
    subgroup_gender_embeddings = subgroup_gender_embeddings[select]
    subgroup_race_embeddings = subgroup_race_embeddings[select]
    
    temp = np.load('raw_data/embeddings/ijbc/' + feature + '_embeddings.npy')
    embeddings = temp[index_embeddings]

    return embeddings, subgroup_gender_embeddings, subgroup_race_embeddings, subgroup_embeddings, id_embeddings

class EmbeddingsDataset(Dataset):
    """Embeddings dataset."""

    def __init__(self, embeddings, id_embeddings, subgroup_gender_embeddings, subgroup_race_embeddings, subgroup_embeddings):
        """
        Arguments
        """
        self.embeddings = embeddings
        self.id_embeddings = id_embeddings
        self.subgroup_gender_embeddings = subgroup_gender_embeddings
        self.subgroup_race_embeddings = subgroup_race_embeddings
        self.subgroup_embeddings = subgroup_embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx, :], self.id_embeddings[idx], self.subgroup_gender_embeddings[idx], self.subgroup_race_embeddings[idx], self.subgroup_embeddings[idx]


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
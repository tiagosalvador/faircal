import numpy as np
import torch
import pickle
import os
import time

from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

from calibration_methods import BinningCalibration
from calibration_methods import IsotonicCalibration
from calibration_methods import BetaCalibration


def baseline(scores, ground_truth, nbins, calibration_method, score_min=-1, score_max=1):
    if calibration_method == 'binning':
        calibration = BinningCalibration(scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max,
                                         nbins=nbins)
    elif calibration_method == 'isotonic_regression':
        calibration = IsotonicCalibration(scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max)
    elif calibration_method == 'beta':
        calibration = BetaCalibration(scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max)
    else:
        raise ValueError('Calibration method %s not available' % calibration_method)
    confidences = {'cal': calibration.predict(scores['cal']), 'test': calibration.predict(scores['test'])}
    return confidences


def oracle(scores, ground_truth, subgroup_scores, subgroups, nbins, calibration_method):
    confidences = {}
    for dataset in ['cal', 'test']:
        confidences[dataset] = {}
        for att in subgroups.keys():
            confidences[dataset][att] = np.zeros(len(scores[dataset]))

    for att in subgroups.keys():
        for subgroup in subgroups[att]:
            select = {}
            for dataset in ['cal', 'test']:
                select[dataset] = np.logical_and(
                    subgroup_scores[dataset][att]['left'] == subgroup,
                    subgroup_scores[dataset][att]['right'] == subgroup
                )

            scores_cal_subgroup = scores['cal'][select['cal']]
            ground_truth_cal_subgroup = ground_truth['cal'][select['cal']]
            if calibration_method == 'binning':
                calibration = BinningCalibration(scores_cal_subgroup, ground_truth_cal_subgroup, nbins=nbins)
            elif calibration_method == 'isotonic_regression':
                calibration = IsotonicCalibration(scores_cal_subgroup, ground_truth_cal_subgroup)
            elif calibration_method == 'beta':
                calibration = BetaCalibration(scores_cal_subgroup, ground_truth_cal_subgroup)
            else:
                raise ValueError('Calibration method %s not available' % calibration_method)

            confidences['cal'][att][select['cal']] = calibration.predict(scores_cal_subgroup)
            confidences['test'][att][select['test']] = calibration.predict(scores['test'][select['test']])
    return confidences

def gst(scores, ground_truth, subgroup_scores, subgroups, nbins, calibration_method, fpr):
    fair_scores = {}
    confidences = {}
    for att in subgroups.keys():
        fair_scores[att] = {}
        confidences[att] = {}
        global_threshold = find_threshold(scores['cal'], ground_truth['cal'], fpr)
        local_threshold = {}
        for subgroup in subgroups[att]:
            select = {}
            for dataset in ['cal', 'test']:
                select[dataset] = np.logical_and(
                    subgroup_scores[dataset][att]['left'] == subgroup,
                    subgroup_scores[dataset][att]['right'] == subgroup
                )

            scores_cal_subgroup = scores['cal'][select['cal']]
            ground_truth_cal_subgroup = ground_truth['cal'][select['cal']]
            local_threshold[subgroup] = find_threshold(scores_cal_subgroup, ground_truth_cal_subgroup, fpr)

        

        for dataset in ['cal', 'test']:
            fair_scores[att][dataset] = np.zeros(len(scores[dataset]))
            for subgroup in subgroups[att]:
                select = np.logical_and(
                    subgroup_scores[dataset][att]['left'] == subgroup,
                    subgroup_scores[dataset][att]['right'] == subgroup
                )
                fair_scores[att][dataset][select] += local_threshold[subgroup]-global_threshold

            fair_scores[att][dataset] = scores[dataset]-fair_scores[att][dataset]

        # The fair scores are no longer cosine similarity scores so they may not lie in the interval [-1,1]
        fair_scores_max = 1 - min(local_threshold.values()) + global_threshold
        fair_scores_min = -1 - max(local_threshold.values()) + global_threshold

        confidences[att] = baseline(
            fair_scores[att],
            ground_truth,
            nbins,
            calibration_method,
            score_min=fair_scores_min,
            score_max=fair_scores_max
        ) 
    return fair_scores, confidences


def cluster_methods(nbins, calibration_method, dataset_name, feature, fold, db_fold, n_clusters,
                    score_normalization, fpr):
    # k-means algorithm
    saveto = f"experiments/kmeans/{dataset_name}_{feature}_nclusters{n_clusters}_fold{fold}.npy"
    if not os.path.exists(saveto):
        np.save(saveto, {})
        embeddings = None
        if dataset_name == 'rfw':
            embeddings = collect_embeddings_rfw(feature, db_fold['cal'])
        elif 'bfw' in dataset_name:
            embeddings = collect_embeddings_bfw(feature, db_fold['cal'])
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)
        np.save(saveto, kmeans)
    else:
        while True:
            kmeans = np.load(saveto, allow_pickle=True).item()
            if type(kmeans) != dict:
                break
            print('Waiting for KMeans to be computed')
            time.sleep(60)
    if dataset_name == 'rfw':
        r = collect_miscellania_rfw(n_clusters, feature, kmeans, db_fold)
    elif 'bfw' in dataset_name:
        r = collect_miscellania_bfw(n_clusters, feature, kmeans, db_fold)
    else:
        raise ValueError('Dataset %s not available' % dataset_name)
    scores = r[0]
    ground_truth = r[1]
    clusters = r[2]
    cluster_scores = r[3]

    print('Statistics Cluster K = %d' % n_clusters)
    stats = np.zeros(n_clusters)
    for i_cluster in range(n_clusters):
        select = np.logical_or(cluster_scores['cal'][:, 0] == i_cluster, cluster_scores['cal'][:, 1] == i_cluster)
        clusters[i_cluster]['scores']['cal'] = scores['cal'][select]
        clusters[i_cluster]['ground_truth']['cal'] = ground_truth['cal'][select]
        stats[i_cluster] = len(clusters[i_cluster]['scores']['cal'])

    print('Minimum number of pairs in clusters %d' % (min(stats)))
    print('Maximum number of pairs in clusters %d' % (max(stats)))
    print('Median number of pairs in clusters %1.1f' % (np.median(stats)))
    print('Mean number of pairs in clusters %1.1f' % (np.mean(stats)))

    if score_normalization:

        global_threshold = find_threshold(scores['cal'], ground_truth['cal'], fpr)
        local_threshold = np.zeros(n_clusters)

        for i_cluster in range(n_clusters):
            scores_cal = clusters[i_cluster]['scores']['cal']
            ground_truth_cal = clusters[i_cluster]['ground_truth']['cal']
            local_threshold[i_cluster] = find_threshold(scores_cal, ground_truth_cal, fpr)

        fair_scores = {}

        for dataset in ['cal', 'test']:
            fair_scores[dataset] = np.zeros(len(scores[dataset]))
            for i_cluster in range(n_clusters):
                for t in [0, 1]:
                    select = cluster_scores[dataset][:, t] == i_cluster
                    fair_scores[dataset][select] += local_threshold[i_cluster] - global_threshold

            fair_scores[dataset] = scores[dataset] - fair_scores[dataset] / 2

        # The fair scores are no longer cosine similarity scores so they may not lie in the interval [-1,1]
        fair_scores_max = 1 - min(local_threshold - global_threshold)
        fair_scores_min = -1 - max(local_threshold - global_threshold)

        confidences = baseline(
            fair_scores,
            ground_truth,
            nbins,
            calibration_method,
            score_min=fair_scores_min,
            score_max=fair_scores_max
        )
    else:
        fair_scores = {}
        confidences = {}

        # Fit calibration
        cluster_calibration_method = {}
        for i_cluster in range(n_clusters):
            scores_cal = clusters[i_cluster]['scores']['cal']
            ground_truth_cal = clusters[i_cluster]['ground_truth']['cal']
            if calibration_method == 'binning':
                cluster_calibration_method[i_cluster] = BinningCalibration(scores_cal, ground_truth_cal, nbins=nbins)
            elif calibration_method == 'isotonic_regression':
                cluster_calibration_method[i_cluster] = IsotonicCalibration(scores_cal, ground_truth_cal)
            elif calibration_method == 'beta':
                cluster_calibration_method[i_cluster] = BetaCalibration(scores_cal, ground_truth_cal)
            clusters[i_cluster]['confidences'] = {}
            clusters[i_cluster]['confidences']['cal'] = cluster_calibration_method[i_cluster].predict(scores_cal)

        for dataset in ['cal', 'test']:
            confidences[dataset] = np.zeros(len(scores[dataset]))
            p = np.zeros(len(scores[dataset]))
            for i_cluster in range(n_clusters):
                for t in [0, 1]:
                    select = cluster_scores[dataset][:, t] == i_cluster
                    aux = scores[dataset][select]
                    if len(aux) > 0:
                        aux = cluster_calibration_method[i_cluster].predict(aux)
                        confidences[dataset][select] += aux * stats[i_cluster]
                        p[select] += stats[i_cluster]
            confidences[dataset] = confidences[dataset] / p

    return scores, ground_truth, confidences, fair_scores


def find_threshold(scores, ground_truth, fpr_threshold):
    far, tar, thresholds = roc_curve(ground_truth, scores, drop_intermediate=True)
    aux = np.abs(far - fpr_threshold)
    return np.min(thresholds[aux == np.min(aux)])


def collect_embeddings_rfw(feature, db_cal):
    # collect embeddings of all the images in the calibration set
    embeddings = np.zeros((0, 512))  # all embeddings are in a 512-dimensional space
    faces_id_num = []
    if feature != 'arcface':
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            temp = pickle.load(open('data/rfw/' + subgroup + '_' + feature + '_embeddings.pickle', 'rb'))
            select = db_cal['ethnicity'] == subgroup

            for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                folder_names = db_cal[select][id_face].values
                file_names = db_cal[select][id_face] + '_000' + db_cal[select][num_face].astype(str) + '.jpg'
                file_names = file_names.values
                for folder_name, file_name in zip(folder_names, file_names):
                    key = 'rfw/data/' + subgroup + '_cropped/' + folder_name + '/' + file_name
                    if file_name not in faces_id_num:
                        embeddings = np.concatenate((embeddings, temp[key]))
                        faces_id_num.append(file_name)
    else:
        temp = pickle.load(open('data/rfw/rfw_' + feature + '_embeddings.pickle', 'rb'))
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            select = db_cal['ethnicity'] == subgroup

            for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                folder_names = db_cal[select][id_face].values
                file_names = db_cal[select][id_face] + '_000' + db_cal[select][num_face].astype(str) + '.jpg'
                file_names = file_names.values
                for folder_name, file_name in zip(folder_names, file_names):
                    key = 'rfw/data/' + subgroup + '/' + folder_name + '/' + file_name
                    if file_name not in faces_id_num:
                        embeddings = np.concatenate((embeddings, temp[key].reshape(1, -1)))
                        faces_id_num.append(file_name)

    return embeddings


def collect_embeddings_bfw(feature, db_cal):
    # collect embeddings of all the images in the calibration set
    embeddings = np.zeros((0, 512))  # all embeddings are in a 512-dimensional space
    file_names_visited = []
    temp = pickle.load(open('data/bfw/' + feature + '_embeddings.pickle', 'rb'))
    for path in ['path1', 'path2']:
        file_names = db_cal[path].values
        for file_name in file_names:
            if file_name not in file_names_visited:
                embeddings = np.concatenate((embeddings, temp[file_name].reshape(1, -1)))
                file_names_visited.append(file_name)

    return embeddings


def collect_miscellania_rfw(n_clusters, feature, kmeans, db_fold):
    # setup clusters
    clusters = {}
    for i_cluster in range(n_clusters):
        clusters[i_cluster] = {}

        for variable in ['scores', 'ground_truth']:
            clusters[i_cluster][variable] = {}
            for dataset in ['cal', 'test']:
                clusters[i_cluster][variable][dataset] = []
    scores = {}
    ground_truth = {}
    cluster_scores = {}
    for dataset in ['cal', 'test']:
        scores[dataset] = np.zeros(len(db_fold[dataset]))
        ground_truth[dataset] = np.zeros(len(db_fold[dataset])).astype(bool)
        cluster_scores[dataset] = np.zeros((len(db_fold[dataset]), 2)).astype(int)

    # collect scores, ground_truth per cluster for the calibration set

    if feature != 'arcface':
        subgroup_old = ''
        temp = None
        for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
            scores[dataset] = np.array(db[feature])
            ground_truth[dataset] = np.array(db['same'].astype(bool))
            for i in range(len(db)):
                subgroup = db['ethnicity'].iloc[i]
                if subgroup != subgroup_old:
                    temp = pickle.load(
                        open('data/rfw/' + subgroup + '_' + feature + '_embeddings.pickle', 'rb'))
                subgroup_old = subgroup

                t = 0
                for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                    folder_name = db[id_face].iloc[i]
                    file_name = db[id_face].iloc[i] + '_000' + str(db[num_face].iloc[i]) + '.jpg'
                    key = 'rfw/data/' + subgroup + '_cropped/' + folder_name + '/' + file_name
                    i_cluster = kmeans.predict(temp[key])[0]
                    cluster_scores[dataset][i, t] = i_cluster
                    t += 1
    else:
        temp = pickle.load(open('data/rfw/rfw_' + feature + '_embeddings.pickle', 'rb'))
        for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
            scores[dataset] = np.array(db[feature])
            ground_truth[dataset] = np.array(db['same'].astype(bool))
            for i in range(len(db)):
                subgroup = db['ethnicity'].iloc[i]
                t = 0
                for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                    folder_name = db[id_face].iloc[i]
                    file_name = db[id_face].iloc[i] + '_000' + str(db[num_face].iloc[i]) + '.jpg'
                    key = 'rfw/data/' + subgroup + '/' + folder_name + '/' + file_name
                    i_cluster = kmeans.predict(temp[key].reshape(1, -1).astype(float))[0]
                    cluster_scores[dataset][i, t] = i_cluster
                    t += 1

    return scores, ground_truth, clusters, cluster_scores


def collect_miscellania_bfw(n_clusters, feature, kmeans, db_fold):
    # setup clusters
    clusters = {}
    for i_cluster in range(n_clusters):
        clusters[i_cluster] = {}

        for variable in ['scores', 'ground_truth']:
            clusters[i_cluster][variable] = {}
            for dataset in ['cal', 'test']:
                clusters[i_cluster][variable][dataset] = []
    scores = {}
    ground_truth = {}
    cluster_scores = {}
    for dataset in ['cal', 'test']:
        scores[dataset] = np.zeros(len(db_fold[dataset]))
        ground_truth[dataset] = np.zeros(len(db_fold[dataset])).astype(bool)
        cluster_scores[dataset] = np.zeros((len(db_fold[dataset]), 2)).astype(int)

    # collect scores and ground_truth per cluster for the calibration set
    temp = pickle.load(open('data/bfw/' + feature + '_embeddings.pickle', 'rb'))
    for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
        scores[dataset] = np.array(db[feature])
        ground_truth[dataset] = np.array(db['same'].astype(bool))
        for i in range(len(db)):
            t = 0
            for path in ['path1', 'path2']:
                key = db[path].iloc[i]
                if feature == 'arcface':
                    i_cluster = kmeans.predict(temp[key].reshape(1, -1).astype(float))[0]
                else:
                    i_cluster = kmeans.predict(temp[key])[0]

                cluster_scores[dataset][i, t] = i_cluster
                t += 1

    return scores, ground_truth, clusters, cluster_scores

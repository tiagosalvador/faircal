import numpy as np
from sklearn.metrics import roc_curve


def determine_edges(scores, nbins, indicesQ=False, score_min=-1, score_max=1):
    if nbins > len(scores):
        print('More bins than values. Attempting to return bins with one value each.')
        return determine_edges(scores, len(scores), indicesQ=indicesQ, score_min=score_min, score_max=score_max)
    scores_sorted = scores.copy()
    scores_index = scores.argsort()
    scores_sorted = scores_sorted[scores_index]
    aux = np.linspace(0, len(scores_sorted) - 1, nbins + 1).astype(int) + 1
    bin_indices = np.zeros(len(scores_sorted)).astype(int)
    bin_indices[:aux[1]] = 0
    for i in range(1, len(aux) - 1):
        bin_indices[aux[i]:aux[i + 1]] = i
    bin_edges = np.zeros(nbins + 1)
    for i in range(0, nbins - 1):
        bin_edges[i + 1] = np.mean(np.concatenate((
            scores_sorted[bin_indices == i][scores_sorted[bin_indices == i] == max(scores_sorted[bin_indices == i])],
            scores_sorted[bin_indices == (i + 1)][
                scores_sorted[bin_indices == (i + 1)] == min(scores_sorted[bin_indices == (i + 1)])])))
    bin_edges[0] = score_min
    bin_edges[-1] = score_max
    bin_indices = bin_indices[np.argsort(scores_index)]
    if len(np.unique(bin_edges)) < len(bin_edges):
        return determine_edges(scores, nbins - 1, indicesQ=indicesQ, score_min=score_min, score_max=score_max)
    else:
        if indicesQ:
            return bin_edges, bin_indices
        else:
            return bin_edges


def bin_confidences_and_accuracies(confidences, ground_truth, bin_edges, indices):
    i = np.arange(0, bin_edges.size-1)
    aux = indices == i.reshape((-1, 1))
    counts = aux.sum(axis=1)
    weights = counts / np.sum(counts)
    correct = np.logical_and(aux, ground_truth).sum(axis=1)
    a = np.repeat(confidences.reshape(1, -1), bin_edges.size-1, axis=0)
    a[np.logical_not(aux)] = 0
    bin_accuracy = correct / counts
    bin_confidence = a.sum(axis=1) / counts
    return weights, bin_accuracy, bin_confidence


def get_ks(confidences, ground_truth):
    n = len(ground_truth)
    order_sort = np.argsort(confidences)
    ks = np.max(np.abs(np.cumsum(confidences[order_sort])/n-np.cumsum(ground_truth[order_sort])/n))
    return ks


def get_brier(confidences, ground_truth):
    # Compute Brier Score
    brier = np.zeros(confidences.shape)
    brier[ground_truth] = (1-confidences[ground_truth])**2
    brier[np.logical_not(ground_truth)] = (confidences[np.logical_not(ground_truth)])**2
    brier = np.mean(brier)
    return brier


def get_ece(confidences, ground_truth, nbins):
    # Repeated code from determine edges. Here it is okay if the bin edges are not uniquely defined
    confidences_sorted = confidences.copy()
    confidences_index = confidences.argsort()
    confidences_sorted = confidences_sorted[confidences_index]
    aux = np.linspace(0, len(confidences_sorted) - 1, nbins + 1).astype(int) + 1
    bin_indices = np.zeros(len(confidences_sorted)).astype(int)
    bin_indices[:aux[1]] = 0
    for i in range(1, len(aux) - 1):
        bin_indices[aux[i]:aux[i + 1]] = i
    bin_edges = np.zeros(nbins + 1)
    for i in range(0, nbins - 1):
        bin_edges[i + 1] = np.mean(np.concatenate((
            confidences_sorted[bin_indices == i][confidences_sorted[bin_indices == i] == max(confidences_sorted[bin_indices == i])],
            confidences_sorted[bin_indices == (i + 1)][
                confidences_sorted[bin_indices == (i + 1)] == min(confidences_sorted[bin_indices == (i + 1)])])))
    bin_edges[0] = 0
    bin_edges[-1] = 1
    bin_indices = bin_indices[np.argsort(confidences_index)]

    weights, bin_accuracy, bin_confidence = bin_confidences_and_accuracies(confidences, ground_truth, bin_edges,
                                                                           bin_indices)
    ece = np.dot(weights, np.abs(bin_confidence - bin_accuracy))
    return ece


def compute_scores(confidences, ground_truth, nbins):

    # Compute ECE
    ece = get_ece(confidences, ground_truth, nbins)

    # Compute Brier
    brier = get_brier(confidences, ground_truth)

    # Compute KS
    ks = get_ks(confidences, ground_truth)

    return ece, ks, brier


def get_binary_clf_stats(ground_truth, scores):
    thr = np.unique(scores.copy())
    # false positives
    fp = np.sum(scores[np.logical_not(ground_truth)] >= thr.reshape(-1, 1), axis=1)
    # true negative
    tn = np.sum(scores[np.logical_not(ground_truth)] < thr.reshape(-1, 1), axis=1)
    # false negative
    fn = np.sum(scores[ground_truth] < thr.reshape(-1, 1), axis=1)
    # true positive
    tp = np.sum(scores[ground_truth] >= thr.reshape(-1, 1), axis=1)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    return fpr, tpr, thr

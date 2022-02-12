import math
import numpy as np
import sklearn.metrics
import numba


def weighting_ranking_metric(get_weights, S, ks=[None]):
    T = np.argsort(S) == 0
    T = np.sum(T, axis=-2)
    T = T[::-1]
    
    weights = get_weights(len(T))
    C = T * weights
    
    scores = np.array([C[:k].sum() for k in ks])
    scores /= len(S)

    return scores


ndcg_weights = 1 / np.log2(np.arange(2, 2 + 100))


def _get_ndcg_weights(length):
    global ndcg_weights
    if length > len(ndcg_weights):
        ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
    return ndcg_weights[:length]


def ndcg(S, ks=[None]):
    scores = weighting_ranking_metric(_get_ndcg_weights, S, ks)
    return scores


mrr_weights = 1 / np.arange(1, 1 + 100)


def _get_mrr_weights(length):
    global mrr_weights
    if length > len(mrr_weights):
        mrr_weights = 1 / np.arange(2, length + 2)
    return mrr_weights[:length]


def mrr(S, ks=[None]):
    scores = weighting_ranking_metric(_get_mrr_weights, S, ks)
    return scores


def hit(S, ks=[1]):
    T = np.argsort(S) == 0
    scores = np.array([T[:, -k:].sum() for k in ks], dtype=float)
    scores /= len(S)
    return scores


def group_auc(S):
    # labels = np.concatenate(([1, ], np.zeros(S.shape[-1] - 1, dtype=int)))
    # score = np.mean(
    #     [
    #         sklearn.metrics.roc_auc_score(labels, s)
    #         for s in S
    #     ]
    # )
    
    # faster version (for only 1 pos)
    m = S.shape[-1]
    a = np.arange(0, m)
    T = np.argsort(S) == 0
    T = np.sum(T, axis=-2)
    score = (a * T).sum() / ((m - 1) * len(S))
    
    return score


# def all_metrics(S):
#     group_auc_score = group_auc(S)
    
#     T = np.argsort(S) == 0
    
#     hit_ks = [3, 10]
#     hit_scores = np.array([T[:, -k:].sum() for k in hit_ks], dtype=float)
#     hit_scores /= len(S)
    
#     T = np.sum(T, axis=-2)
#     T = T[::-1]
    
#     W = _get_ndcg_weights(len(T))
#     C = T * W
    
#     ndcg_ks = [None, 1, 3, 10]
#     if S.shape[-1] > 100:
#         ndcg_ks.extend([100])
#     ndcg_scores = np.array([C[:k].sum() for k in ndcg_ks])
#     ndcg_scores /= len(S)
    
#     W = _get_mrr_weights(len(T))
#     C = T * W
    
#     mean_mrr = C.sum() / len(S)
    
#     results = {
#         "group_auc":    group_auc_score,
#         "ndcg":         ndcg_scores[0],
#         "ndcg@1":       ndcg_scores[1],
#         "ndcg@3":       ndcg_scores[2],
#         "ndcg@10":      ndcg_scores[3],
#         "hit@3":        hit_scores[0],
#         "hit@10":       hit_scores[1],
#         "mrr":          mean_mrr
#     }
#     if S.shape[-1] > 100:
#         results['ndcg@100'] = ndcg_scores[4]
#     return results


@numba.jit(nopython=True)
def get_rank(A):
    rank = np.empty(len(A), dtype=np.int32)
    for i in range(len(A)):
        a = A[i]
        key = a[0]
        r = 0
        for j in range(1, len(a)):
            if a[j] > key:
                r += 1
        rank[i] = r
    return rank


def all_metrics(S):
    num_samples = S.shape[0]
    num_scores = S.shape[1]
    rank = get_rank(S)
    
    top1 = rank == 0
    top3 = rank < 3
    top10 = rank < 10
    top20 = rank < 20
    top100 = rank < 100
    top300 = rank < 300
    
    results = {
        "auc": (num_scores - 1 - rank).mean() / (num_scores - 1)
    }
    
    # ndcg
    w = _get_ndcg_weights(num_scores)
    results.update({
        "ndcg": w[rank].sum() / num_samples,
        "n1": top1.mean(),
        "n3": w[rank[top3]].sum() / num_samples,
        "n10": w[rank[top10]].sum() / num_samples,
        "n20": w[rank[top20]].sum() / num_samples,
        "n100": w[rank[top100]].sum() / num_samples,
        "n300": w[rank[top300]].sum() / num_samples,
    })
    
    # hit
    results.update({
        "h3": top3.mean(),
        "h10": top10.mean(),
        "h100": top100.mean(),
        "h300": top300.mean(),
    })
    
    # mrr
    w = _get_mrr_weights(num_scores)
    results.update({
        "mrr": w[rank].sum() / num_samples,
    })

    return results

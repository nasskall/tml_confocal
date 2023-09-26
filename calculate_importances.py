import numpy as np
import pandas as pd
from skimage.segmentation import felzenszwalb
from sklearn.metrics import euclidean_distances
from cv_utils.cv_utils import features
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed


def calculate_importances(df_sample, gmm, cls, params, sel_algo):
    kp, des, im = features(df_sample, show=False)
    vec_dim = len(des[0])
    distances = euclidean_distances(des, gmm.means_)
    if sel_algo == 'Felzenswalb':
        segments = felzenszwalb(im, scale=params[0], sigma=params[1], min_size=params[2])
    elif sel_algo == 'Slic':
        segments = slic(im, n_segments=params[0], compactness=params[1], sigma=params[2],
                        start_label=1, slic_zero=True)
    elif sel_algo == 'Quickshift':
        segments = quickshift(im, kernel_size=params[0], max_dist=params[1], ratio=params[2])
    feat_imp = cls.feature_importances_
    xg_coef = feat_imp.clip(min=0)
    weights = gmm.weights_
    keypoints = []
    idx = 0
    for i in distances:
        importances = np.take(xg_coef, [np.argmin(i)] +
                              list(range(5 + (np.argmin(i) - 1) * vec_dim, 5 + np.argmin(i) * vec_dim))
                              + list(range(325 + (np.argmin(i) - 1) * vec_dim, 325 + np.argmin(i) * vec_dim)), axis=0)
        importances = 10000*importances
        keypoints.append(
            [sum(importances * weights[np.argmin(i)]) / (np.min(i)), round(kp[idx].pt[1]),
             round(kp[idx].pt[0]),
             kp[idx].size, segments[round(kp[idx].pt[1])][round(kp[idx].pt[0])]])
        idx = idx + 1
    df = pd.DataFrame(keypoints)
    df.columns = ['Total_Importance', 'Xx', 'Yy', 'Size', 'Segment']
    df.to_csv("n_results.csv")
    df_min = df['Total_Importance'].min()
    df_max = df['Total_Importance'].max()
    df['Total_Importance'] = round(255 * df['Total_Importance'] - df_min) / (df_max - df_min)
    return des, df, segments, im

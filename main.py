from PIL import Image
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, euclidean_distances
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier
from skimage.segmentation import felzenszwalb, mark_boundaries, find_boundaries, slic, quickshift
import cv2
import pickle
from calculate_importances import calculate_importances
from heatmap_utils.heatmap_generator import generate_enh_map, generate_s_map
import streamlit as st
from ifv_utils.fisher_vector_impl import fisher_vector

le = preprocessing.LabelEncoder()
y = ['Benign', 'Malignant']
le.fit(y)
y = le.transform(y)
n_clusters = 5


def load_model():
    infile_g = open('models/n_gaussian_mixture' + str(n_clusters), 'rb')
    vocab = pickle.load(infile_g)
    infile_g = open('models/n_classifier' + str(n_clusters), 'rb')
    cls = pickle.load(infile_g)
    return vocab, cls


def main():
    vocab, cls = load_model()
    image_s = None
    st.title("Explaining predictions on skin confocal images")
    seg_algo = st.sidebar.radio('Type of segmentation algorithm', ('Felzenswalb', 'Slic', 'Quickshift'))
    with st.container():
        st.title("Fisher Vectors and " + seg_algo)
        st.sidebar.write('You selected Fisher Vectors and ' + seg_algo)
        imp_threshold, params = get_params(seg_algo)
        sample = st.file_uploader("Choose an sample image...")
        if sample is not None:
            image_s = Image.open(sample)
            image_s = image_s.resize((300, 300))
            st.image(image_s, caption='Sample Image', width=300)
        if sample is not None:
            heatmap = Image.open('./reds.png')
            if st.button('Predict and Explain'):
                with st.spinner("Explaining predictive model's results"):
                    image_s = np.array(image_s)
                    mega_his = np.zeros((1, 2 * n_clusters * 128 + n_clusters))
                    des, df, segments, im = calculate_importances(image_s, vocab, cls, params,seg_algo)
                    fv = fisher_vector(des, vocab)
                    mega_his[0, :] += fv
                    # predict the class of the image
                    lb = cls.predict(mega_his)
                    prob = cls.predict_proba(mega_his)
                    enh_map = generate_enh_map(im, segments, df,imp_threshold)
                    plain_map = generate_s_map(im, df)
                    output_image1 = Image.fromarray(plain_map)
                    output_image2 = Image.fromarray(enh_map)
                    if output_image1:
                        st.header("Prediction: {}".format(le.inverse_transform(lb)[0]))
                        st.header("Probability: {:.2f} for Fisher Vector".format(float(prob.max())))
                        col_gc, col_gcc = st.columns(2)
                        with col_gc:
                            st.subheader("Plain heatmap")
                            st.image(output_image1)
                        with col_gcc:
                            st.subheader("Enhanced Superpixels heatmap")
                            st.image(output_image2)
                            st.image(heatmap)
                        st.success('Done')
                if st.button(
                        'Try again'):
                    st.rerun()


def get_params(seg_algo=None):
    imp_threshold = st.sidebar.slider(
        'Importance threshold',
        0.01, 1.0, 0.55)
    if seg_algo == 'Felzenswalb':
        scale = st.sidebar.slider(
            'Scale',
            0, 1000, 100)
        sigma = st.sidebar.slider(
            'Sigma',
            0.01, 2.00, 0.5)
        min_size = st.sidebar.slider(
            'Minimum size',
            0, 2000, 50)
        params = [scale, sigma, min_size]
    elif seg_algo == 'Slic':
        n_segments = st.sidebar.slider(
            'Number of segments',
            1, 1000, 250)
        compactness = st.sidebar.slider(
            'Compactness',
            0.1, 100.0, 10.0)
        sigma = st.sidebar.slider(
            'Sigma',
            0.1, 10.0, 1.0)
        params = [n_segments, compactness, sigma]
    elif seg_algo == 'Quickshift':
        kernel_size = st.sidebar.slider(
            'Kernel size',
            1, 10, 3)
        max_dist = st.sidebar.slider(
            'Maximum distance',
            1, 20, 6)
        ratio = st.sidebar.slider(
            'Ratio',
            0.1, 1.0, 0.5)
        params = [kernel_size, max_dist, ratio]
    return imp_threshold, params


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

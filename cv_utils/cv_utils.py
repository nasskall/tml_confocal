import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
from ifv_utils.fisher_vector_impl import fisher_vector

clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
sift = cv2.SIFT_create()

def segment(img):
    img = clahe.apply(img)
    if img is not None:
        segments_fz = felzenszwalb(img, scale=1500, sigma=1, min_size=3500).astype('float64')
    return segments_fz, img

def features(image, show=False):
    #image = cv2.imread(img_path, 0)
    if show:
        imS = cv2.resize(image, (960, 540))
        cv2.imshow("Original", imS)
    #mask, image_cl = segment(image)
    """
    if show:
        maskS = cv2.resize(mask, (960, 540))
        cv2.imshow("Mask", maskS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    kp, des = sift.detectAndCompute(image.astype('uint8'), None)
    return [kp, des, image]

def formatND(l):
    vStack = np.array(l[0])
    for remaining in l[1:]:
        vStack = np.vstack((vStack, remaining))
    desc_vstack = vStack.copy()
    return vStack, desc_vstack

def develop_vocabulary(n_clusters, gmm, n_images, descriptor_list):
    mega_histogram = np.zeros((n_images, 2 * n_clusters * 128 + n_clusters))
    for i in range(n_images):
        des = descriptor_list[i]
        fv = fisher_vector(des, gmm)
        mega_histogram[i, :] += fv
    print("Vocabulary Histogram Generated")
    return mega_histogram






































































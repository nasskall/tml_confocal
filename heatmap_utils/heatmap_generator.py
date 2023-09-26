import cv2
import numpy as np
from matplotlib import pyplot as plt, colors, cm
from skimage.segmentation import mark_boundaries


def generate_s_map(im, df, limit=50):
    hmap = np.zeros((im.shape[0], im.shape[1]))
    for _, row in df.iterrows():
        if row['Total_Importance'] > limit:
            r = int(row['Size'].astype('int64'))
            for x in range(-r, r):
                for y in range(-r, r):
                    if row['Xx'].astype('int64') + x >= im.shape[0] or row['Yy'].astype('int64') + y >= \
                            im.shape[1]:
                        continue
                    else:
                        hmap[row['Xx'].astype('int64') + x, row['Yy'].astype('int64') + y] = hmap[row['Xx'].astype(
                            'int64') + x, row['Yy'].astype('int64') + y] + int(row['Total_Importance']) * ((r - abs(
                            x)) / r) * ((r - abs(y)) / r)
    heatmap = (hmap - hmap.min()) / (hmap.max() - hmap.min())
    hmap= cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_HOT),cv2.COLOR_BGR2RGB)
    output_image = cv2.addWeighted(im.astype('uint8'), 1, hmap,0.5, 0)
    output_image = cv2.resize(output_image, (300, 300))
    return output_image


def generate_enh_map(im, segments, df, limit=0):
    dominant_kp_list = []
    imp_segments = df.Segment.unique()
    for seg in imp_segments:
        df_sel = df[df['Segment'] == seg]
        dominant_kp_list.append(df_sel['Total_Importance'].idxmax())
    df = df.loc[df.index[dominant_kp_list]]
    norm = colors.Normalize(vmin=0.0, vmax=256.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    img = im
    for _, rw in df.iterrows():
        if rw['Total_Importance'] > limit:
            imp_seg = rw['Segment']
            seg_image = mark_boundaries(img, (segments == imp_seg).astype(int),
                                        color=mapper.to_rgba(rw['Total_Importance'])[0:3], mode='inner')
        img = seg_image
    img = np.uint8(255 * img)
    return img

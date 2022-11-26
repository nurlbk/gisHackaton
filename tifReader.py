import os
from statistics import mean
import numpy as np
import pandas as pd
import rasterio as rs
from matplotlib import pylab
from rasterio.plot import show
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from sklearn.cluster import KMeans




def tif_read(filenames: [str]):
    sample_tif = rs.open(filenames[0])
    reds = []
    nirs = []

    tif_height, tif_width = sample_tif.height, sample_tif.width

    for filename in filenames:
        image_file = r"C:\Users\Nurlybek\Desktop\nic\\" + filename
        with rs.open(image_file) as src:
            reds.append(src.read(3))
        with rs.open(image_file) as src:
            nirs.append(src.read(4))

    ndvi = []
    rows = [0] * tif_width
    for _ in range(tif_height):
        ndvi.append(rows.copy())

    for file_id in range(len(filenames)):
        sample_ndvi = np.where(
            (nirs[file_id] + reds[file_id]) == 0.,
            0,
            (nirs[file_id] - reds[file_id]) / (nirs[file_id] + reds[file_id]))
        for i in range(len(sample_ndvi)):
            for j in range(len(sample_ndvi[0])):
                ndvi[i][j] += sample_ndvi[i][j] / len(filenames)

    ndvi = np.array(ndvi)

    ndvi_class_bins = [-np.inf, 0, 0.35, 0.43, 0.5, np.inf]
    ndvi_landsat_class = np.digitize(ndvi, ndvi_class_bins)

    ndvi_landsat_class = np.ma.masked_where(
        np.ma.getmask(ndvi), ndvi_landsat_class
    )
    np.unique(ndvi_landsat_class)

    nbr_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
    nbr_cmap = ListedColormap(nbr_colors)

    classes = np.unique(ndvi_landsat_class)
    classes = classes.tolist()
    classes = classes[0:5]

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(ndvi_landsat_class, cmap=nbr_cmap)

    ep.draw_legend(im_ax=im, classes=classes)
    ax.set_title(
        "ndvi",
        fontsize=14,
    )
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig('plot.png')

    # data_frame = pd.DataFrame(result)
    # data_frame.to_csv('test2.csv', sep=";")

    # return to_png(result, tif_width, tif_height)
    return ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1)


def tif_read_old(filenames: [str], num_of_clusters=4):
    sample_tif = rs.open(filenames[0])
    tif_height, tif_width = sample_tif.height, sample_tif.width
    df = []
    for _ in range(tif_height * tif_width):
        df.append([])

    for filename in filenames:
        images = rs.open(filename)
        for img_id in range(images.count):
            for i in range(tif_height):
                for j in range(tif_width):
                    df[i * tif_width + j].append(images.read(img_id + 1)[i][j])

    km = KMeans(n_clusters=num_of_clusters)
    km.fit(df)

    avg_clusters_productivity = []
    for _ in range(num_of_clusters):
        avg_clusters_productivity.append([])

    result = []
    rows = [0] * tif_width
    for _ in range(tif_height):
        result.append(rows.copy())

    for i in range(tif_height):
        for j in range(tif_width):
            pixel = df[i * tif_width + j]
            result[i][j] = km.predict([pixel])[0]
            avg_clusters_productivity[result[i][j]].append(mean(pixel))

    for i in range(num_of_clusters):
        avg_clusters_productivity[i] = mean(avg_clusters_productivity[i])

    sort_index = []
    sorted_clusters = sorted(avg_clusters_productivity)
    for i in avg_clusters_productivity:
        sort_index.append(sorted_clusters.index(i))

    for i in range(tif_height):
        for j in range(tif_width):
            result[i][j] = sort_index[result[i][j]] + 1

    # show(np.array(result))
    data_frame = pd.DataFrame(result)
    data_frame.to_csv('test2.csv', sep=";")

    # return to_png(result, tif_width, tif_height)


def to_png(data, width, height):
    colors = np.array([[104, 104, 104], [255, 0, 0], [255, 255, 0], [0, 255, 0]], dtype=np.uint8)

    new_colors = []
    rows = [0] * width
    for _ in range(height):
        new_colors.append(rows.copy())

    for i in range(height):
        for j in range(width):
            new_colors[i][j] = colors[data[i][j] - 1]

    img = Image.fromarray(np.array(new_colors)).convert('RGB')
    img_resized = img.resize((width * 4, height * 4), resample=Image.NEAREST)
    return img_resized


def tif_dir_reader(dirname):
    filenames = os.listdir(dirname)
    sample_tif = rs.open(f"{dirname}/" + filenames[0])
    tif_height, tif_width = sample_tif.height, sample_tif.width

    df = []
    for _ in range(tif_height * tif_width):
        df.append([])

    for filename in filenames:
        images = rs.open(f"{dirname}/" + filename)
        for img_id in range(images.count):
            for i in range(tif_height):
                for j in range(tif_width):
                    df[i * tif_width + j].append(images.read(img_id + 1)[i][j])

    km = KMeans(n_clusters=4)
    km.fit(df)

    result = []
    rows = [0] * tif_width
    for _ in range(tif_height):
        result.append(rows.copy())

    for i in range(tif_height):
        for j in range(tif_width):
            result[i][j] = km.predict([df[i * tif_width + j]])[0]

    data_frame = pd.DataFrame(result)
    data_frame.to_csv('result.csv', sep=";")

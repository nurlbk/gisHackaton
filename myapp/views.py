import os

from django.shortcuts import redirect, render

from myproject.settings import BASE_DIR
from .models import Document
# from .forms import DocumentForm

from statistics import mean
import numpy as np
import rasterio as rs
from sklearn.cluster import KMeans
from PIL import Image


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


def tif_read(filenames: [str], num_of_clusters=4):
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
    # data_frame = pd.DataFrame(result)
    # data_frame.to_csv('test1.csv', sep=";")

    return to_png(result, tif_width, tif_height)


def my_view(request):
    message = 'Upload as many files as you want!'
    output = None
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        file_list = []
        for file in files:
            newfile = Document.objects.create(docfile=file)
            file_list.append(BASE_DIR + newfile.docfile.url)

        output = tif_read(file_list)

        output.save(BASE_DIR + '/media/output/pic.png', format='png')
        output = '/media/output/pic.png'

    context = {'message': message, 'output': output}
    return render(request, 'list.html', context)

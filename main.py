import os

import PIL.Image
import matplotlib.pyplot as plt
import tifReader


def main():
    filenames = os.listdir('field')
    for i, filename in enumerate(filenames):
        filenames[i] = "field/" + filename

    ph = tifReader.tif_read(filenames)


if __name__ == '__main__':
    main()

import glob
import os

from PIL import Image

for dataset in ['test', 'train', 'val']:

    glob_exp = 'C:/Users/marcinis/Politechnika/sem8/inz/dataset/full/%s/**/*.jpeg' % dataset
    resolution_count = dict()
    for filepath in glob.iglob(glob_exp, recursive=True):
        img = Image.open(filepath)
        img_res = img.size
        img_name = os.path.basename(filepath)
        img_class = img_name.split('-')[0]

        classXRes = img_class + '-' + str(img_res)
        resolution_count[classXRes] = resolution_count.get(classXRes, 0) + 1


    print('Resolution count for {} dataset: {}'.format(dataset, resolution_count))
    with open("class_resolution_count.txt", "a") as myfile:
        myfile.write('Resolution count for {} dataset: {}'.format(dataset, resolution_count))

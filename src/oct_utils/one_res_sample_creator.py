import glob
import os
from pathlib import Path
from shutil import copy2

from PIL import Image

for dataset in ['test', 'val', 'train']:

    glob_exp = 'dataset\\%s\\**\\*.jpeg' % dataset
    resolution_count = dict()
    for filepath in glob.iglob(glob_exp, recursive=True):

        img = Image.open(filepath)
        img_res = img.size
        resolution_count[img_res] = resolution_count.get(img_res, 0) + 1

        if img_res == (512, 496):
            img_path = Path(filepath)
            img_path_relative = img_path.relative_to('dataset')

            new_dataset_path = Path('small_dataset')
            new_full_img_path = new_dataset_path / img_path_relative
            os.makedirs(str(new_full_img_path.parent), exist_ok=True)
            copy2(filepath, new_full_img_path)

    print('Resolution count for {} dataset: {}'.format(dataset, resolution_count))


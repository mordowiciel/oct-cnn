import glob
import random

from PIL import Image

random.seed(666)

def generate_class_sample(dataset_path, class_name):
    search_path = '{}\\{}\\*.jpeg'.format(dataset_path, class_name)
    res_to_paths = dict()

    for counter, filepath in enumerate(glob.iglob(search_path, recursive=True)):
        img = Image.open(filepath)
        img_res = str(img.size)

        paths_list = res_to_paths.get(img_res, [])
        paths_list.append(filepath)
        res_to_paths[img_res] = paths_list

    merge_list = []
    for res, paths_with_res in res_to_paths.items():
        paths_with_res_sample = random.sample(paths_with_res, int(0.2 * len(paths_with_res)))
        merge_list.append(paths_with_res_sample)

    # flatten list
    return [item for sublist in merge_list for item in sublist]

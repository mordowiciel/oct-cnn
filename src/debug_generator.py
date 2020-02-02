import os

from keras_preprocessing.image import load_img, img_to_array

from generator_resolver import GeneratorResolver
from oct_config import OCTConfig
from oct_logger import print_cfg

cfg = OCTConfig('../config.ini')
print_cfg(cfg)

if not os.path.exists('../preview'):
    os.mkdir('../preview')

img = load_img('C:/Users/marcinis/Politechnika/sem8/inz/dataset/full/train/CNV/CNV-13823-1.jpeg',
               color_mode='grayscale')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0

gen_resolver = GeneratorResolver(cfg)
training_datagen, test_datagen, val_datagen = gen_resolver.provide_image_data_generators()

for batch in val_datagen.flow(x, batch_size=1,
                                   save_to_dir='../preview', save_prefix='cnv', save_format='jpeg'):
    i += 1
    if i > 50:
        break  # otherwise the generator would loop indefinitely

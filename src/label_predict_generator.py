'''
CANDIDATE TO DELETE
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from oct_data_generator import OCTDataGenerator


def result_resolver(res):
    concat_res = []
    for i in res:
        highest_class_idx = np.argmax(i)
        concat_res.append(highest_class_idx)
    return concat_res


IMG_SIZE = (124, 128)
INPUT_SHAPE = IMG_SIZE + (1,)
model = load_model('../models/LENET-FULL-2019-04-27T12-20-57-categorical_crossentropy-sgd.h5')
val_data_generator = OCTDataGenerator(dataset_path='C:/Users/marcinis/Politechnika/sem8/inz/dataset/full/test',
                                      batch_size=1,
                                      dim=IMG_SIZE,
                                      n_channels=1,
                                      n_classes=4,
                                      shuffle=False)


y_true = val_data_generator.item_labels
y_pred = result_resolver(model.predict_generator(val_data_generator, verbose=1))

# Calculate global precision and recall
global_precision = precision_score(y_true, y_pred, average='micro')
global_recall = recall_score(y_true, y_pred, average='micro')

print('Global precision: %.4f' % global_precision)
print('Global recall: %.4f' % global_recall)

# Generate classification report
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
report = classification_report(y_true, y_pred, target_names=labels)
print(report)

# Create and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, range(4),
                     range(4))
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

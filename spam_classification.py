import numpy as np
from classifier import Classification_Model


import pandas as pd
data = pd.read_csv("dataset/spam_ham_dataset.csv")

sentences = data["text"].values.tolist()
labels = data["label_num"].values.tolist()


split_data = int(len(sentences) * 0.85)

train_sentences = sentences[:split_data]
train_labels = labels[:split_data]

test_sentences = sentences[split_data:]
test_labels = labels[split_data:]


batch_size = 2


sparse_categorical = 0

n_epochs = [5, 5, 5]  ## DNN--RNN-CNN
Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN
no_of_classes = 2
Classification_Model(np.array(train_sentences), np.array(train_labels), np.array(test_sentences),
                         np.array(test_labels),
                         batch_size=batch_size,
                         sparse_categorical=sparse_categorical,
                         random_deep=Random_Deep,
                         epochs=n_epochs, no_of_classes=2, plot=True)
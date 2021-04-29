import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
# Google colab can't handle full length dataset
mode = np.random.choice(['train', 'val', 'test'])

indices = np.load('./smalldataset_indices.npz')['test']
"""
train = np.load('smalldataset_train.npz')['data']
val = np.load('smalldataset_val.npz')['data']
"""
data = np.load('smalldataset_test.npz')['data']

# train_df = pd.read_csv('./mnist_complex/train/train_labels.csv')
df = pd.read_csv('./mnist_complex/test/test_labels.csv')

plot_idx = np.random.choice(len(data), 16)
for i in range(16):
    plt.subplot(4, 4, i+1)
    idx = plot_idx[i]
    plt.imshow(data[idx], cmap='gray')
    idx = indices[idx]
    plt.axis('off')
    plt.title(f'Composite class: {df["Composite class"][idx]}\nComponent classes: {df["Component classes"][idx]}')

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()
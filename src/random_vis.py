import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
mode = 'test'

data = np.load('complex_mnist_test_binarized.npz')['data']
df = pd.read_csv('./mnist_complex/test/test_labels.csv')
print(data.shape)
plot_idx = np.random.choice(len(data), 16)

for i in range(16):
    plt.subplot(4, 4, i+1)
    idx = plot_idx[i]
    plt.imshow(data[idx])
    plt.axis('off')
    plt.title(f'Composite class: {df["Composite class"][idx]}\nComponent classes: {df["Component classes"][idx]}')

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()
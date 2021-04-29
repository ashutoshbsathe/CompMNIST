import numpy as np 
from post_process import post_process 

np.random.seed(1618033989) # billion times golden ratio

def main():
    # 72-12-12 split
    train = [12000, 12000, 12000, 36000] 
    val = [2000, 2000, 2000, 6000]
    test = [2000, 2000, 2000, 6000]

    train_indices = [
        np.random.choice(np.arange(i*60000, (i+1)*60000), train[i], replace=False)
        for i in range(4)
    ]
    val_indices = [
        np.random.choice(np.setdiff1d(np.arange(i*60000, (i+1)*60000), train_indices[i], assume_unique=True), val[i], replace=False)
        for i in range(4)
    ]
    test_indices = [
        np.random.choice(np.arange(i*10000, (i+1)*10000), test[i], replace=False)
        for i in range(4)
    ]

    for idx in train_indices:
        print(len(idx), idx.min(), idx.max(), idx.shape)
    print(64*'-')
    for idx in val_indices:
        print(len(idx), idx.min(), idx.max(), idx.shape)
    print(64*'-')
    for idx in test_indices:
        print(len(idx), idx.min(), idx.max(), idx.shape)
    print(64*'-')
    for i in range(4):
        print(f'train[{i}] - val[{i}] = {np.setdiff1d(train_indices[i], val_indices[i]).shape}')
        print(f'val[{i}] - train[{i}] = {np.setdiff1d(val_indices[i], train_indices[i]).shape}')
        print(16*'-')

    train_indices = np.concatenate(train_indices, axis=0)
    val_indices = np.concatenate(val_indices, axis=0)
    test_indices = np.concatenate(test_indices, axis=0)

    print(train_indices.shape)
    print(val_indices.shape)
    print(test_indices.shape)

    np.savez_compressed('./smalldataset_indices.npz', train=train_indices, val=val_indices, test=test_indices)

    post_process('./mnist_complex/train/train_labels.csv', 'smalldataset_train.npz', train_indices)
    post_process('./mnist_complex/train/train_labels.csv', 'smalldataset_val.npz', val_indices)
    post_process('./mnist_complex/test/test_labels.csv', 'smalldataset_test.npz', test_indices)

if __name__ == '__main__':
    main()
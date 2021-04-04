import os 
import gzip 
import codecs 
import string
from PIL import Image 
from multiprocessing import Pool
import numpy as np 
from tqdm import tqdm
from utils import check_integrity
def integer(bytepattern: bytes) -> int:
    return int(codecs.encode(bytepattern, 'hex'), 16)

def read_images_from_idx3(fname: str) -> np.ndarray:
    raw = gzip.open(fname, 'rb').read() 
    magic = integer(raw[0:4])
    assert magic == 2051
    size, rows, cols = map(integer, (raw[4:8], raw[8:12], raw[12:16]))
    raw = np.frombuffer(raw[16:], dtype=np.uint8)
    raw = raw.reshape(-1, rows, cols)
    assert raw.shape[0] == size 
    return raw

def read_labels_from_idx3(fname: str) -> np.ndarray:
    raw = gzip.open(fname, 'rb').read() 
    magic = integer(raw[0:4])
    assert magic == 2049
    size = integer(raw[4:8])
    raw = np.frombuffer(raw[8:], dtype=np.uint8)
    assert raw.shape[0] == size
    return raw

def read_mnist_into_memory(root_dir: str = './mnist_raw/') -> tuple:
    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    for f, md5 in resources:
        file_path = os.path.join(root_dir, f)
        if check_integrity(file_path, md5):
            print(f'File {file_path} is correct')
        else:
            print(f'File {file_path} is corrupt or does not exist, please recheck it')
    
    train_images = read_images_from_idx3(os.path.join(root_dir, resources[0][0]))
    train_labels = read_labels_from_idx3(os.path.join(root_dir, resources[1][0]))
    test_images = read_images_from_idx3(os.path.join(root_dir, resources[2][0]))
    test_labels = read_labels_from_idx3(os.path.join(root_dir, resources[3][0]))

    return train_images, train_labels, test_images, test_labels

def write_img(info: tuple) -> bool:
    name, img = info 
    img = Image.fromarray(img.astype(np.uint8))
    img.save(name)
    return True

def multiprocess_write(fpath_template: str, arr: np.ndarray, desc: str = '', nprocesses: int = 8) -> bool:
    size = arr.shape[0]
    fnames = [fpath_template.format(i) for i in range(size)]
    with Pool(processes=nprocesses) as p:
        with tqdm(total=size) as pbar:
            pbar.set_description(desc)
            for i, _ in enumerate(p.imap_unordered(write_img, zip(fnames, arr))):
                pbar.update()
    return True

def sort_mnist(root_dir: str = './mnist_raw', dest_dir: str = './mnist_sorted', invert: bool = True) -> bool:
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist_into_memory(root_dir)
    if invert:
        train_imgs = (train_imgs > 127) * 0 + (train_imgs <= 127) * 255
        test_imgs = (test_imgs > 127) * 0 + (test_imgs <= 127) * 255
    print('Saving train dataset')
    for i in range(10):
        idxs = np.where(train_lbls == i)
        dest = os.path.join(dest_dir, f'train/class_{i}')
        os.makedirs(dest, exist_ok=True)
        multiprocess_write(os.path.join(dest, '{}.bmp'), train_imgs[idxs], f'Class {i}')
    print('Saving test dataset')
    for i in range(10):
        idxs = np.where(test_lbls == i)
        dest = os.path.join(dest_dir, f'test/class_{i}')
        os.makedirs(dest, exist_ok=True)
        multiprocess_write(os.path.join(dest, '{}.bmp'), test_imgs[idxs], f'Class {i}')
    return True

if __name__ == '__main__':
    sort_mnist()


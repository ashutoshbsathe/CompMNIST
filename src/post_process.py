import pandas as pd 
import numpy as np 
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
target_size = 224
binarize = False 
def process_image(file_path: str):
    img = Image.open(file_path)
    w, h = img.size
    size = max(target_size, w, h)
    x = Image.new('RGB', (size, size), (255, 255, 255))
    x.paste(img, (int((size - w) / 2), int((size - h) / 2)))
    x = np.array(x.resize((target_size, target_size)).convert('L'))
    if binarize:
        x = (x >= (x.max() + x.min() / 2)).astype(np.bool)
    return x

def post_process(f: str, output:str, indices=None):
    df = pd.read_csv(f)
    assert indices is not None
    print(len(indices))
    if indices is None:
        file_list = df['Output File'].tolist()#[:10]
    else:
        # file_list = df.iloc[indices, dfd.columns.get_loc('Output File')].tolist()
        file_list = df.iloc[indices]['Output File'].tolist()
        print(len(file_list))
        assert len(file_list) == len(indices)
    with Pool(processes=4) as p:
        """
        with tqdm(total=10) as pbar:
            pbar.set_description('lol')
            for i, data in enumerate(p.imap_unordered(job, range(10))):
                pbar.update()
                print(data)
        """
        l =  list(tqdm(p.imap(process_image, file_list), total=len(file_list), desc='Post processing'))
        l = np.asarray(l)
        np.savez_compressed(output, data=l)
        print(l.shape, l.dtype)
if __name__ == '__main__':
    post_process('./mnist_complex/train/train_labels.csv', 'complex_mnist_train.npz')
    post_process('./mnist_complex/test/test_labels.csv', 'complex_mnist_test.npz')

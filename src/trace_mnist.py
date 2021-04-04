import os 
import subprocess
from multiprocessing import Pool
from tqdm import tqdm 

COMMAND = '{potrace} {input} -o {output} -b svg'
def trace(info: tuple) -> int:
    src, dest = info
    args = {
        'potrace': 'potrace',
        'input': src,
        'output': dest
    }
    potrace = subprocess.run(COMMAND.format(**args), shell=True)
    return potrace 

def multiprocess_trace(src_dir: str, dest_dir: str, desc: str = '', nprocesses: int = 8) -> bool:
    os.makedirs(dest_dir, exist_ok=True)
    src_file_list = os.listdir(src_dir)
    src_file_list = [x for x in src_file_list if x.endswith('bmp')]
    dest_file_list = [x.replace('bmp', 'svg') for x in src_file_list]
    src_file_list = [os.path.join(src_dir, x) for x in src_file_list]
    dest_file_list = [os.path.join(dest_dir, x) for x in dest_file_list]
    with Pool(processes=nprocesses) as p:
        with tqdm(total=len(src_file_list)) as pbar:
            pbar.set_description(desc)
            for i, _ in enumerate(p.imap_unordered(trace, zip(src_file_list, dest_file_list))):
                pbar.update()
    return True

def trace_mnist(src_dir: str = './mnist_sorted', dest_dir: str = './mnist_traced') -> bool:
    print(f'Tracing {src_dir}/train')
    for i in range(10):
        src  = os.path.join(src_dir,  f'train/class_{i}')
        dest = os.path.join(dest_dir, f'train/class_{i}')
        multiprocess_trace(src, dest, f'Class {i}')
    print(f'Tracing {src_dir}/test')
    for i in range(10):
        src  = os.path.join(src_dir,  f'test/class_{i}')
        dest = os.path.join(dest_dir, f'test/class_{i}')
        multiprocess_trace(src, dest, f'Class {i}')
    return True

if __name__ == '__main__':
    trace_mnist()

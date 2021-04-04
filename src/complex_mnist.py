import os
from glob import glob 
from multiprocessing import Pool
from svgpathtools import svg2paths
import numpy as np 
from tqdm import tqdm
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 
TOTAL_SAMPLES = 32
SAMPLES_PER_PX = 1 
def generate_an_image(info: tuple) -> bool:
    svg_file, bitmap_file, output_file, first, second = info 
    paths, attributes = svg2paths(svg_file)
    # Divide total sample points across all the various paths in SVG 
    total_length = 0
    for path in paths:
        total_length += path.length() # https://stackoverflow.com/a/38297053
    samples_per_path = []
    for path in paths:
        samples_per_path.append(int(path.length() * TOTAL_SAMPLES / total_length))
    # Enumerate over paths and sample points 
    points_on_paths = {}
    for i, (nsamples, path, attr) in enumerate(zip(samples_per_path, paths, attributes)):
        nsamples = nsamples if nsamples > 1 else 2
        points = []
        length = path.length()
        colour = attr['stroke'] if 'stroke' in attr.keys() else 'black' # maybe use this for better visualizations 
        for j in range(nsamples):
            points.append(path.point(j / (nsamples - 1)))
        points_on_paths[colour + f'{i}'] = np.array(points)
    anno_img = plt.imread(bitmap_file)
    w, h = anno_img.shape 
    anno = np.empty((w, h, 3), dtype=np.uint8)
    anno[:, :, 0] = anno[:, :, 1] = anno[:, :, 2] = anno_img 
    if anno.max() < 255:
        anno = anno * 255
    fig, ax = plt.subplots()
    for k, v in points_on_paths.items():
        x = [e.real for e in v]
        y = [e.imag for e in v]
        ax.scatter(x, y)
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(OffsetImage(anno), (x0, y0), frameon=False)
            ax.add_artist(ab)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()
    plt.close()
    return True
def generate_dataset(sorted_dir: str = './mnist_sorted', traced_dir: str = './mnist_traced', dest_dir: str = './mnist_complex', mode: str = 'train', nprocesses: int = 8) -> bool:
    class_count = []
    total_count = 0
    for i in range(10):
        subdir = mode + f'/class_{i}/'
        bitmap_count = len(glob(os.path.join(sorted_dir, subdir + '*.bmp')))
        traced_count = len(glob(os.path.join(traced_dir, subdir + '*.svg')))
        assert bitmap_count == traced_count, f'Bitmap count ({bitmap_count}) and traced count ({traced_count}) unequal for class {i}' 
        class_count.append(bitmap_count)
        total_count += traced_count 
    if mode == 'train':
        assert total_count == 60000, f'Expected 60000 images in train dataset, got {total_count}'
    elif mode == 'test':
        assert total_count == 10000, f'Expected 10000 images in test dataset, got {total_count}'
    # Generate listings
    ops_list = []
    for i in range(10):
        # For class `i`
        subdir = mode + f'/class_{i}/'
        # What are other classes other than this class ?
        other_classes = list(range(0, i)) + list(range(i+1, 10))
        # Generate second class for every image 
        second_class_list = np.random.choice(other_classes, class_count[i])
        # Create the subdir at the destination 
        os.makedirs(os.path.join(dest_dir, subdir), exist_ok=True)
        for j in range(class_count[i]):
            # For every svg of this class 
            svg_file = os.path.join(traced_dir, subdir + f'{j}.svg')
            # What image should I pick from the second class ?
            second_class = second_class_list[j]
            k = np.random.randint(0, class_count[second_class])
            bitmap_file = os.path.join(sorted_dir, mode + f'/class_{second_class}/{k}.bmp')
            # Where should I save this image ?
            dest_file = os.path.join(dest_dir, subdir + f'{j}.png')
            ops_list.append((svg_file, bitmap_file, dest_file, i, second_class))
    df = pd.DataFrame(ops_list, columns=['SVG File', 'Bitmap File', 'Output File', 'Overall class', 'Inner class'])
    df.to_csv(os.path.join(dest_dir, mode + '/' + mode + '_labels.csv'))
    with Pool(processes=nprocesses) as p:
        with tqdm(total=len(ops_list)) as pbar:
            pbar.set_description(mode)
            for i, _ in enumerate(p.imap_unordered(generate_an_image, ops_list)):
                pbar.update()
    return True 
    

if __name__ == '__main__':
    generate_dataset(mode='train')
    generate_dataset(mode='test')


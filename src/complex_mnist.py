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
np.random.seed(1618033989) # billion times golden ratio

def read_img(f: str) -> np.ndarray:
    anno_img = plt.imread(f)
    w, h = anno_img.shape 
    anno = np.empty((w, h, 3), dtype=np.uint8)
    anno[:, :, 0] = anno[:, :, 1] = anno[:, :, 2] = anno_img 
    if anno.max() < 255:
        anno = anno * 255
    return anno

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
    if len(bitmap_file) == 1:
        anno = read_img(bitmap_file[0])
        anno_imgs = [anno] * TOTAL_SAMPLES
    else:
        anno_imgs = []
        for f in bitmap_file:
            anno = read_img(f)
            anno_imgs.append(anno)
    assert len(anno_imgs) == TOTAL_SAMPLES
    fig, ax = plt.subplots()
    for k, v in points_on_paths.items():
        x = [e.real for e in v]
        y = [e.imag for e in v]
        ax.scatter(x, y)
        for i, (x0, y0) in enumerate(zip(x, y)):
            ab = AnnotationBbox(OffsetImage(anno_imgs[i]), (x0, y0), frameon=False)
            ax.add_artist(ab)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()
    plt.close()
    return True

def generate_dataset(sorted_dir: str = './mnist_sorted', traced_dir: str = './mnist_traced', dest_dir: str = './mnist_complex', mode: str = 'train', nprocesses: int = 64) -> bool:
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
    if not os.path.exists(os.path.join(dest_dir, mode + '/' + mode + '_labels.csv')):
        ops_list = []

        def get_ops(composite, component_classes, composition_mode, composite_indices, dataset_mode):
            ret = []
            subdir = dataset_mode + f'/class_{composite}/'
            os.makedirs(os.path.join(dest_dir, subdir), exist_ok=True)
            if composition_mode == 'same':
                component_classes = np.random.choice(component_classes, len(composite_indices))
                for i in composite_indices:
                    component = component_classes[i]
                    svg_file = os.path.join(traced_dir, subdir + f'{i}.svg')
                    component_idx = np.random.randint(0, class_count[component])
                    bitmap_file = os.path.join(sorted_dir, dataset_mode + f'/class_{component}/{component_idx}.bmp')
                    output_file = os.path.join(dest_dir, subdir + f'{i}_{component}_{composition_mode}.png')
                    ret.append((svg_file, [bitmap_file], output_file, composite, [component]))
            elif composition_mode == 'diff':
                for i in composite_indices:
                    component = np.random.choice(component_classes, TOTAL_SAMPLES)
                    svg_file = os.path.join(traced_dir, subdir + f'{i}.svg')
                    bitmap_files = []
                    for c in component:
                        component_idx = np.random.randint(0, class_count[c])
                        bitmap_files.append(os.path.join(sorted_dir, dataset_mode + f'/class_{c}/{component_idx}.bmp'))
                    output_file = os.path.join(dest_dir, subdir + f'{i}_{list(np.unique(component))}_{composition_mode}.png')
                    ret.append((svg_file, bitmap_files, output_file, composite, list(np.unique(component))))
            return ret 

        # Same class, same image - type 0 
        for i in range(10):
            subdir = mode + f'/class_{i}/'
            composite = i
            component_classes = [i]
            composition_mode = 'same'
            # composite_indices can be less or more than class counts depending on arg passed 
            composite_indices = range(class_count[composite])
            ops_list.extend(get_ops(composite, component_classes, composition_mode, composite_indices, mode)) 
        # Diff class, same image - type 1 
        for i in range(10):
            subdir = mode + f'/class_{i}/'
            composite = i
            component_classes = list(range(0, i)) + list(range(i+1, 10)) 
            composition_mode = 'same'
            # composite_indices can be less or more than class counts depending on arg passed 
            composite_indices = range(class_count[composite])
            ops_list.extend(get_ops(composite, component_classes, composition_mode, composite_indices, mode)) 
        # Same classes, diff images - type 2
        for i in range(10):
            subdir = mode + f'/class_{i}/'
            composite = i
            component_classes = [i]
            composition_mode = 'diff'
            # composite_indices can be less or more than class counts depending on arg passed 
            composite_indices = range(class_count[composite])
            ops_list.extend(get_ops(composite, component_classes, composition_mode, composite_indices, mode)) 
        # Diff classes, diff images - type 3 
        for i in range(10):
            subdir = mode + f'/class_{i}/'
            composite = i
            component_classes = list(range(0, i)) + list(range(i+1, 10)) 
            composition_mode = 'diff'
            # composite_indices can be less or more than class counts depending on arg passed 
            composite_indices = range(class_count[composite])
            ops_list.extend(get_ops(composite, component_classes, composition_mode, composite_indices, mode)) 
        df = pd.DataFrame(ops_list, columns=['SVG File', 'Bitmap File', 'Output File', 'Composite class', 'Component classes'])
        print(len(df.index))
        df.to_csv(os.path.join(dest_dir, mode + '/' + mode + '_labels.csv'))
        df.to_csv(mode + '_labels_new.csv')
    else:
        df = pd.read_csv(os.path.join(dest_dir, mode + '/' + mode + '_labels.csv'))
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        df['Bitmap File'] = df['Bitmap File'].apply(lambda x: [i.strip()[1:-1] for i in x[1:-1].split(',')])
        df['Component classes'] = df['Component classes'].apply(lambda x: [int(i) for i in x[1:-1].split(',')])
        ops_list = list(df.itertuples(index=False, name=None))
        print(ops_list[120000])
        print(len(ops_list), type(ops_list), type(ops_list[0][1]), type(ops_list[0][-1]), type(ops_list[0]))
    with Pool(processes=nprocesses) as p:
        with tqdm(total=len(ops_list)) as pbar:
            pbar.set_description(mode)
            for i, _ in enumerate(p.imap_unordered(generate_an_image, ops_list)):
                pbar.update()
    return True 
    

if __name__ == '__main__':
    generate_dataset(mode='train')
    generate_dataset(mode='test')


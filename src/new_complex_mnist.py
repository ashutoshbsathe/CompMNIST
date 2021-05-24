import os
from glob import glob 
from multiprocessing import Pool
from svgpathtools import svg2paths
import numpy as np 
from tqdm import tqdm
import pandas as pd 
from PIL import Image
import json
import re
TOTAL_SAMPLES = 32
SAMPLES_PER_PX = 1 
width = 28 
height = 28 
padding = 4
scale_multiplier = 0
np.random.seed(1618033989) # billion times golden ratio

def generate_an_image(info: tuple) -> bool:
    svg_file, bitmap_file, output_file, composite, components = info 
    assert isinstance(components, list) or isinstance(components, np.ndarray)
    if len(components) == 1:
        components = components * TOTAL_SAMPLES
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
    all_x = []
    all_y = []
    for k, v in points_on_paths.items():
        x = [i.real for i in v]
        y = [i.imag for i in v]
        all_x.extend(x)
        all_y.extend(y)
    x_min = min(all_x) - width / 2 - padding 
    x_max = max(all_x) + width / 2 + padding 
    y_min = min(all_y) - height / 2 - padding 
    y_max = max(all_y) + height / 2 + padding 

    new_x = all_x - x_min 
    new_y = all_y - y_min 

    n = len(new_x)
    assert len(new_y) == n 
    if scale_multiplier != 0:
        assert 0 < scale_multiplier <= 1
        min_dist = x_max - x_min 
        for i in range(n):
            for j in range(i+1, n):
                computes += 1
                dist = ((new_x[i] - new_x[j]) ** 2 + (new_y[i] - new_y[j]) ** 2) ** 0.5
                if dist != 0 and dist < min_dist:
                    min_dist = dist 
        scale = scale_multiplier * width / min_dist 
        """
        0.5 multiplier gives total scaling factor of 1.448
        0.75 multiplier gives total scaling factor of 2.173
        1.0 multiplier gives total scaling factor of 2.897
        """
    else:
        scale = 1
    
    out = np.ones((int(scale * (y_max - y_min)), int(scale * (x_max - x_min)))) * 255
    if len(bitmap_file) == 1:
        component_images = [np.array(Image.open(bitmap_file[0]))] * n 
    else:
        component_images = [np.array(Image.open(f)) for f in bitmap_file]
    assert len(component_images) >= n
    assert len(components) >= n
    labels = []
    for i in range(n):
        x = int(scale * new_x[i])
        y = int(scale * (y_max - y_min - new_y[i]))
        out[y-height//2:y+height//2, x-width//2:x+width//2] = np.minimum(
            out[y-height//2:y+height//2, x-width//2:x+width//2], 
            component_images[i]
        )
        labels.append({
            # to show this, just do plt.imshow(image) and then 
            # add patch with ((x, y), width, height)
            'class': int(components[i]),
            'x': int(x-width//2),
            'y': int(y-height//2),
            'width': width,
            'height': height
        })
    # post process 
    # make images square 
    w, h = out.shape 
    if w > h:
        square = np.ones((w, w)) * 255 
        square[:, (w-h)//2:(w+h)//2] = out 
        for i in range(n):
            labels[i]['x'] += w//2 - h//2
    elif h > w:
        square = np.ones((h, h)) * 255 
        square[(h-w)//2:(h+w)//2, :] = out 
        for i in range(n):
            labels[i]['x'] += h//2 - w//2
    else:
        square = out 
    # convert bounding boxes into ratios of their sides 
    s = square.shape[0]
    for i in range(n):
        labels[i]['x'] /= s
        labels[i]['y'] /= s
        labels[i]['width'] /= s 
        labels[i]['height'] /= s 
    Image.fromarray(square.astype(np.uint8)).save(output_file)
    # bounding boxes
    with open(output_file + '_labels.json', 'w') as f:
        json.dump({'labels': labels}, f)
    return True

def generate_dataset(sorted_dir: str = './mnist_sorted', traced_dir: str = './mnist_traced', dest_dir: str = './mnist_complex', mode: str = 'train', nprocesses: int = 4) -> bool:
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
                    ret.append((svg_file, bitmap_files, output_file, composite, component))
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
        df['Component classes'] = df['Component classes'].apply(lambda x: [int(i) for i in re.split(',|\s', x[1:-1])])
        ops_list = list(df.itertuples(index=False, name=None))
        print(ops_list[len(ops_list) // 2])
        print(len(ops_list), type(ops_list), type(ops_list[0][1]), type(ops_list[0][-1]), type(ops_list[0]))
    with Pool(processes=nprocesses) as p:
        with tqdm(total=len(ops_list)) as pbar:
            pbar.set_description(mode)
            for i, _ in enumerate(p.imap_unordered(generate_an_image, ops_list)):
                pbar.update()
    return True 
    

if __name__ == '__main__':
    # generate_dataset(mode='train', dest_dir='new_mnist_complex')
    generate_dataset(mode='test', dest_dir='new_mnist_complex')


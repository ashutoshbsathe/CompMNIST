from svgpathtools import svg2paths 
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 
TOTAL_SAMPLES = 32 
svg_file = './mnist_traced/train/class_9/0.svg'
width = 28
height = 28
padding = 4

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
    print(k, len(v))
    x = [i.real for i in v]
    y = [i.imag for i in v]
    all_x.extend(x)
    all_y.extend(y)
print(min(all_x), max(all_x))
print(min(all_y), max(all_y))
print(len(all_x), len(all_y), all_x, all_y)

x_min = min(all_x) - width / 2 - padding 
x_max = max(all_x) + width / 2 + padding 
y_min = min(all_y) - height / 2 - padding 
y_max = max(all_y) + height / 2 + padding 

new_x = all_x - x_min 
new_y = all_y - y_min 
print(new_x, new_y)
print(min(new_x), min(new_y))

n = len(new_x)
assert len(new_y) == n 
min_dist = x_max - x_min 
computes = 0
for i in range(n):
    for j in range(i+1, n):
        print(i, j)
        computes += 1
        dist = ((new_x[i] - new_x[j]) ** 2 + (new_y[i] - new_y[j]) ** 2) ** 0.5
        if dist != 0 and dist < min_dist:
            min_dist = dist 
            print(f'Updating min_dist = {min_dist} at i = {i} j = {j}')
    print(16*'-')
print(dist, min_dist, computes)
scale = 1
"""
0.5 multiplier gives total scaling factor of 1.448
0.75 multiplier gives total scaling factor of 2.173
1.0 multiplier gives total scaling factor of 2.897
"""
# scale = 1

out = np.ones((int(scale * (y_max - y_min)), int(scale * (x_max - x_min)))) * 255
print(out.shape)
from PIL import Image
import numpy as np
im = Image.open("./sample_mnist.bmp")
p = np.array(im)
print(p.shape, p.min(), p.max())
labels = []
for i in range(n):
    # x = int(scale * (x_max - x_min - new_x[i]))
    x = int(scale * (new_x[i]))
    # y = int(scale * new_y[i])
    y = int(scale * (y_max - y_min - new_y[i]))
    print(x, y, new_x[i], new_y[i])
    out[y-height//2:y+height//2, x-width//2:x+width//2] = np.minimum(out[y-height//2:y+height//2, x-width//2:x+width//2], p)
    labels.append({
        'x': x-width//2,
        'y': y-width//2,
        'width': width,
        'height': height
    })
print(out.shape, out.min(), out.max())


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
        labels[i]['y'] += h//2 - w//2
else:
    square = out 
out = square 
Image.fromarray(out.astype(np.uint8)).save('new_algo.bmp')

from matplotlib import patches, text, patheffects
plt.set_cmap('Purples')
fig, ax = plt.subplots()
ax.imshow(out)
for i in range(n):
    """
    # x = int(scale * (x_max - x_min - new_x[i]))
    x = int(scale * new_x[i])
    y = int(scale * (y_max - y_min - new_y[i]))
    # y = int(scale * new_y[i])
    """
    x = labels[i]['x']
    y = labels[i]['y']
    width = labels[i]['width']
    height = labels[i]['height']
    ax.add_patch(patches.Rectangle((x, y), height, width, edgecolor='red', facecolor='none'))
plt.savefig(f'new_image_with_scale_factor_{scale}.pdf')
"""
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
    fig.tight_layout(pad=0)
    plt.savefig(output_file, bbox_inches='tight')
    a = plt.Figure().canvas.get_width_height()
    print(type(a), a)
    plt.clf()
    plt.close()
    return True

generate_an_image(('2.svg', ['./sample_mnist.bmp'], 'somethingsomething.png', 0, 0))
"""

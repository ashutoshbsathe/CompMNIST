from svgpathtools import svg2paths, wsvg
from pprint import pprint
import numpy as np 
import matplotlib.pyplot as plt 
# FILE = '/home/ashutosh/Pictures/mnist.svg'
FILE = './mnist.svg'
numSamples = 20 
paths, attributes = svg2paths(FILE)

for i, path in enumerate(paths):
    print(f'i = {i}')
    pprint(path)
    print(64*'-')
for i, attr in enumerate(attributes):
    print(f'i = {i}')
    pprint(attr)
    print(64*'-')
SAMPLES_PER_PX = 1
myPaths = {}
for path,attr in zip(paths, attributes):
    myPathList = []
    pathLength = path.length()
    pathColour = attr['stroke'] if 'stroke' in attr.keys() else 'black'
    # numSamples = int(pathLength * SAMPLES_PER_PX)
    for i in range(numSamples):
        #parametric length = ilength(geometric length)
        myPathList.append(path.point(path.ilength(pathLength * i / (numSamples-1))))
    myPaths[pathColour] = np.array(myPathList)
pprint(myPaths)
for k, v in myPaths.items():
    x = [e.real for e in v]
    y = [e.imag for e in v]
    plt.plot(x, y, marker='v', label=k)
plt.legend()
# !!!!! WE ARE DONE !!!!!
# SEE
# https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
# ALSO THIS, ESPECIALLY THIS
# https://stackoverflow.com/a/53851017/7914234
"""
img_path = './sample_imageone.png'
file_paths = [img_path for _ in range(numSamples)]
fig, ax = plt.subplots()
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
for k, v in myPaths.items():
    x = [e.real for e in v]
    y = [e.imag for e in v]
    ax.plot(x, y)
    for x0, y0, path in zip(x, y, file_paths):
        ab = AnnotationBbox(OffsetImage(plt.imread(path)), (x0, y0))
        ax.add_artist(ab)
"""
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# !!!!! PARALLELIZING THE CODE !!!!!
# https://urban-institute.medium.com/using-multiprocessing-to-make-python-code-faster-23ea5ef996ba
# https://towardsdatascience.com/a-hands-on-guide-to-multiprocessing-in-python-48b59bfcc89e
# https://stackoverflow.com/questions/23537037/python-multicore-programming/23537302

# !!!!! FUTHER EFFORTS !!!!!
# TRY RESCALING THE COORDINATE SYSTEM TO FIT DESIRED SIZE
# REST EVERYTHING SHOULD BE DONE BY THEN 

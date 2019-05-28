import numpy as np

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

def visualize_images(filepaths, probs, figure_path, title=None):
    num_images = len(filepaths)

    fig = plt.figure()
    for i, (filepath, prob) in enumerate(zip(filepaths, probs)):
        ax = plt.subplot(num_images // 5, 5, i + 1)
        ax.axis('off')
        ax.set_title('score: %.3f' % prob)
        
        img = Image.open(filepath).convert('RGB')
        plt.imshow(img)
    
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=600)


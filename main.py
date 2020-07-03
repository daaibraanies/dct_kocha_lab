import sys

import numpy as np
from skimage.util import view_as_blocks
from scipy.fftpack import idctn, dctn
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image


def read_img(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    return img


def eval_hologram(data):
    hologram = dctn(data, norm='ortho')
    min_h = data.min()
    max_h = data.max()
    
    min_condition = hologram < min_h
    max_condition = hologram > max_h
    other_condition = (min_h < hologram) & (hologram < max_h)
     
    hologram[min_condition] = 0
    hologram[max_condition] = 1 
    hologram[other_condition] = (hologram[other_condition] - min_h)/(max_h - min_h)

    return hologram


def restore_img(hologram):
    img = idctn(hologram, norm='ortho')
    img /= np.sum(img.shape)/2
    img = (img - img.min())/(img.max() - img.min())
    return img


def img_to_bits(img):
    img = np.uint8(img*255)
    bits = np.unpackbits(img)
    return bits


def bits_to_img(bits, w, h):
    values = np.packbits(bits)
    img = values.reshape((w, h))/255.
    return img 


u1, v1 = 4, 5
u2, v2 = 5, 4
n = 8
P = 25


def double_to_byte(arr):
    return np.uint8(np.round(np.clip(arr, 0, 255), 0))


def increment_abs(x):
    return x + 1 if x >= 0 else x - 1


def decrement_abs(x):
    if np.abs(x) <= 1:
        return 0
    else:
        return x - 1 if x >= 0 else x + 1
    

def abs_diff_coefs(transform):
    return abs(transform[u1, v1]) - abs(transform[u2, v2])


def valid_coefficients(transform, bit, threshold):
    difference = abs_diff_coefs(transform)
    if (bit == 0) and (difference > threshold):
        return True
    elif (bit == 1) and (difference < -threshold):
        return True
    else:
        return False


def change_coefficients(transform, bit):
    coefs = transform.copy()
    if bit == 0:
        coefs[u1, v1] = increment_abs(coefs[u1, v1])
        coefs[u2, v2] = decrement_abs(coefs[u2, v2])
    elif bit == 1:
        coefs[u1, v1] = decrement_abs(coefs[u1, v1])
        coefs[u2, v2] = increment_abs(coefs[u2, v2])
    return coefs


def embed_bit(block, bit):
    patch = block.copy()
    coefs = dctn(patch) 
    while not valid_coefficients(coefs, bit, P) or (bit != retrieve_bit(patch)):
        coefs = change_coefficients(coefs, bit)
        patch = double_to_byte(idctn(coefs)/(2*n)**2)
    return patch


def embed_message(orig, msg):
    changed = orig.copy()
    blocks = view_as_blocks(changed, block_shape=(n, n))
    h = blocks.shape[1]        
    for index, bit in enumerate(msg):
        i = index // h
        j = index % h
        block = blocks[i, j]
        changed[i*n: (i+1)*n, j*n: (j+1)*n] = embed_bit(block, bit)
    return changed


def retrieve_bit(block):
    transform = dctn(block)
    return 0 if abs_diff_coefs(transform) > 0 else 1


def retrieve_message(img, length):
    blocks = view_as_blocks(img, block_shape=(n, n))
    h = blocks.shape[1]
    return [retrieve_bit(blocks[index//h, index%h]) for index in range(length)]


if __name__ == '__main__':
    hologram_path = sys.argv[1]
    container_path = sys.argv[2]
    img = read_img(hologram_path)
    container = read_img(container_path)

    hologram = eval_hologram(img)
    w, h = hologram.shape
    bits = img_to_bits(hologram)
    img_with_stego = embed_message(container, bits)
    restore_bits = retrieve_message(img_with_stego, len(bits))
    img = bits_to_img(restore_bits, w, h)
    restored_img = restore_img(img)
    
    scipy.misc.imsave('restored.bmp', restored_img) 
    scipy.misc.imsave('stego.bmp', img_with_stego) 
    scipy.misc.imsave('holo.bmp', hologram) 




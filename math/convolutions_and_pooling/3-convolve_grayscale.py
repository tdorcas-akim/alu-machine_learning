#!/usr/bin/env python3
"""
Module to perform a strided convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a strided convolution on grayscale images.

    images: numpy.ndarray with shape (m, h, w)
    kernel: numpy.ndarray with shape (kh, kw)
    padding: tuple (ph, pw), 'same', or 'valid'
    stride: tuple (sh, sw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding dimensions (ph, pw)
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images with zeros
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Calculate output dimensions considering stride
    output_h = (h + (2 * ph) - kh) // sh + 1
    output_w = (w + (2 * pw) - kw) // sw + 1

    # Initialize output array
    convolved = np.zeros((m, output_h, output_w))

    # Perform strided convolution using exactly two loops
    for i in range(output_h):
        for j in range(output_w):
            patch = images_padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved

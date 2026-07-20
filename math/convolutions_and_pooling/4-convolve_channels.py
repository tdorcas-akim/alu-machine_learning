#!/usr/bin/env python3
"""
Module to perform a convolution on images with channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    images: numpy.ndarray with shape (m, h, w, c)
    kernel: numpy.ndarray with shape (kh, kw, c)
    padding: tuple (ph, pw), 'same', or 'valid'
    stride: tuple (sh, sw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # Determine padding dimensions (ph, pw)
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images on height and width, but not on m or c dimensions
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Calculate output dimensions
    output_h = (h + (2 * ph) - kh) // sh + 1
    output_w = (w + (2 * pw) - kw) // sw + 1

    # Initialize output array
    convolved = np.zeros((m, output_h, output_w))

    # Perform convolution across spatial dimensions using two loops
    for i in range(output_h):
        for j in range(output_w):
            patch = images_padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]
            # Multiply patch and kernel element-wise, then sum over kh, kw, c
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))

    return convolved

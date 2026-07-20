#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    images: numpy.ndarray with shape (m, h, w, c)
    kernels: numpy.ndarray with shape (kh, kw, c, nc)
    padding: tuple (ph, pw), 'same', or 'valid'
    stride: tuple (sh, sw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Determine padding dimensions (ph, pw)
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images on height and width dimensions
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Calculate output spatial dimensions
    output_h = (h + (2 * ph) - kh) // sh + 1
    output_w = (w + (2 * pw) - kw) // sw + 1

    # Initialize output array with shape (m, output_h, output_w, nc)
    convolved = np.zeros((m, output_h, output_w, nc))

    # Perform convolution using three loops (i for height, j for width, k for kernel index)
    for i in range(output_h):
        for j in range(output_w):
            patch = images_padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]
            for k in range(nc):
                convolved[:, i, j, k] = np.sum(
                    patch * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved

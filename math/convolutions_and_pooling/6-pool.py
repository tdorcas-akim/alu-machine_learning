#!/usr/bin/env python3
"""
Module to perform pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    images: numpy.ndarray with shape (m, h, w, c)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: string, 'max' or 'avg'

    Returns: numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output spatial dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize output array with shape (m, output_h, output_w, c)
    pooled = np.zeros((m, output_h, output_w, c))

    # Perform pooling using two loops over spatial dimensions
    for i in range(output_h):
        for j in range(output_w):
            patch = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(patch, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(patch, axis=(1, 2))

    return pooled

#!/usr/bin/env python3
"""
Module to perform a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    images: numpy.ndarray with shape (m, h, w)
    kernel: numpy.ndarray with shape (kh, kw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding needed on each side so output size equals input size
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad images with zeros around height and width (axes 1 and 2)
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    # Initialize output array matching original height and width (h, w)
    convolved = np.zeros((m, h, w))

    # Perform convolution using exactly two loops
    for i in range(h):
        for j in range(w):
            patch = images_padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved

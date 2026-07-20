#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    images: numpy.ndarray with shape (m, h, w)
    kernel: numpy.ndarray with shape (kh, kw)
    padding: tuple of (ph, pw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros using custom ph (height) and pw (width)
    images_padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Calculate output height and width after custom padding
    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1

    # Initialize output array
    convolved = np.zeros((m, output_h, output_w))

    # Perform convolution using exactly two loops
    for i in range(output_h):
        for j in range(output_w):
            patch = images_padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolvedi

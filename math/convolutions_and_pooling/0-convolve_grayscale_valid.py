#!/usr/bin/env python3
"""
Module to perform a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    images: numpy.ndarray with shape (m, h, w)
    kernel: numpy.ndarray with shape (kh, kw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions for valid convolution
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output array with zeros
    convolved = np.zeros((m, output_h, output_w))

    # Perform convolution using exactly two loops
    for i in range(output_h):
        for j in range(output_w):
            # Extract image patch and multiply by kernel, then sum
            patch = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved

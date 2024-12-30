import numpy as np
import cupy as cp


class KernelWithSize:
    kernel: cp.RawKernel
    grid_size: tuple[int]
    block_size: tuple[int]

grid_sizes = {}

def get_grid_size_block_size(shape: cp.shape, reduced_thread_count: bool = False) -> (tuple[int], tuple[int]):
    """
    Heuristic calculation to obtain a whole number division of the provided shape that can be used as a valid CUDA kernel grid and block size.

    :param shape: The number of kernel threads in each direction; shape = grid_size * block_size
    :param reduced_thread_count: If true, then a limiter block size is allowed.
    :return: grid_size, block_size
    """
    global grid_sizes
    key = ()
    # Create cache key and use cache if available:
    if len(shape) == 3:
        key = (shape[0], shape[1], shape[2])
        if key in grid_sizes:  # Use cached value
            return tuple(grid_sizes[key]), (shape[0] // grid_sizes[key][0], shape[1] // grid_sizes[key][1], shape[2] // grid_sizes[key][2])
    elif len(shape) == 2:
        key = (shape[0], shape[1])
        if key in grid_sizes:  # Use cached value
            return tuple(grid_sizes[key]), (shape[0] // grid_sizes[key][0], shape[1] // grid_sizes[key][1])
    elif len(shape) == 1:
        key = (shape[0])
        if key in grid_sizes:  # Use cached value
            return tuple(grid_sizes[key]), (shape[0] // grid_sizes[key][0])

    allowed_thread_count_per_block = 256 if reduced_thread_count else 512
    initial_guess = 2
    while True:
        grid_size = []
        block_size = []
        thread_count = 1
        for i in range(len(shape)):
            grid_size.append(initial_guess)
            if shape[i] < initial_guess:
                grid_size[i] = shape[i]
            # Find divisor:
            while True:
                if shape[i] % grid_size[i] == 0:
                    break
                grid_size[i] += 1
            # Check total thread count in a single block:
            block_dim = shape[i] // grid_size[i]
            thread_count *= block_dim
            block_size.append(block_dim)
        if thread_count <= allowed_thread_count_per_block:
            break
        else:
            initial_guess *= 2  # Need to find a greater divisor

    grid_sizes[key] = grid_size  # Cache

    return tuple(grid_size), tuple(block_size)


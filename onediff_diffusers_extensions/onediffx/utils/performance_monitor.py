import time
from contextlib import contextmanager

import torch


@contextmanager
def track_inference_time(warmup=False, use_cuda=True):
    """
    A context manager to measure the execution time of models.
    Parameters:
        warmup (bool): If True, prints the time for warmup runs; otherwise, prints the time for normal runs.
        use_cuda (bool): If CUDA is available, uses torch.cuda.Event for timing; otherwise, uses time.time().
    """
    if use_cuda and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.time()

    try:
        yield
    finally:
        if use_cuda and torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000.0
        else:
            elapsed_time = time.time() - start_time

        if warmup:
            print(f"Warmup run - Execution time: {elapsed_time:.2f} seconds")
        else:
            print(f"Normal run - Execution time: {elapsed_time:.2f} seconds")

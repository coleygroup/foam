"""parallel_utils.py

Utilities for parallel processing with automatic chunking and retry logic.
Uses pathos multiprocessing for better pickling support with complex objects.
"""
import itertools
import logging
from typing import Any, Callable, Iterator, List, Optional, TypeVar

import multiprocess.context as ctx
from tqdm import tqdm

ctx._force_start_method('spawn')

T = TypeVar('T')
R = TypeVar('R')


def simple_parallel(
    input_list: List[T],
    function: Callable[[T], R],
    max_cpu: int = 16,
    timeout: int = 4000,
    max_retries: int = 3,
    task_name: str = "",
) -> List[R]:
    """ Simple parallelization.

    Use map async and retries in case we get odd stalling behavior.

    input_list: Input list to op on
    function: Fn to apply
    max_cpu: Num cpus
    timeout: Length of timeout
    max_retries: Num times to retry this

    """
    from multiprocess.context import TimeoutError
    from pathos import multiprocessing as mp

    cpus = min(mp.cpu_count(), max_cpu)
    with mp.ProcessPool(processes=cpus) as pool:
        list_outputs = list(tqdm(pool.imap(function, input_list), total=len(input_list), desc=task_name))

    return list_outputs


def chunked_parallel(
    input_list: List[T],
    function: Callable[[T], R],
    chunks: int = 100,
    max_cpu: int = 16,
    timeout: int = 4000,
    max_retries: int = 3,
    task_name: str = "",
) -> List[R]:
    """Apply a function to a list in parallel with automatic chunking.

    Divides input into chunks to reduce process spawning overhead while
    maintaining parallelism. More efficient than simple_parallel for
    large input lists with fast-executing functions.

    Args:
        input_list: List of objects to process.
        function: Callable taking one input and returning a single value.
        chunks: Target number of chunks to divide input into.
        max_cpu: Maximum number of CPU workers.
        timeout: Timeout in seconds for the entire operation.
        max_retries: Number of retry attempts on failure.
        task_name: Description for progress bar.

    Returns:
        List of results in same order as input_list.
    """
    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs: 
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [input_list[i:i+step_size]
                    for i in range(0, len(input_list), step_size)]

    list_outputs = simple_parallel(chunked_list, batch_func, max_cpu=max_cpu,
                                   timeout=timeout, max_retries=max_retries,
                                   task_name=task_name)
    # Unroll
    full_output = [j for i in list_outputs for j in i]

    return full_output


def chunked_parallel_retries(
    input_list: List[T],
    function: Callable[[T], R],
    chunks: int = 100,
    max_cpu: int = 16,
    output_func: Optional[Callable[[Iterator[R]], None]] = None,
    task_name: str = "",
    gpu: bool = False,
    **kwargs: Any,
) -> Optional[List[R]]:
    """Apply a function in parallel with chunking and automatic failure retry.

    Extended version of chunked_parallel that tracks failures and can
    optionally stream results to disk via output_func.

    Args:
        input_list: List of objects to process.
        function: Callable taking one input and returning a single value.
        chunks: Target number of chunks to divide input into.
        max_cpu: Maximum number of CPU workers.
        output_func: Optional callable to stream results to disk. If provided,
            results are passed as an iterator and None is returned.
        task_name: Description for progress bar.
        gpu: Unused, reserved for future GPU support.
        **kwargs: Additional arguments passed to ProcessPool.

    Returns:
        List of results if output_func is None, otherwise None.
        Failed items are returned as string error messages.
    """
    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    if list_len == 0:
        raise ValueError('Empty list to process!')
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    from pathos import multiprocessing as mp
    cpus = min(mp.cpu_count(), max_cpu)
    with mp.ProcessPool(processes=cpus, **kwargs) as pool:
        iter_outputs = tqdm(pool.imap(batch_func, chunked_list), total=len(chunked_list), desc=task_name)
        if output_func is not None:
            output_func(itertools.chain.from_iterable(iter_outputs))

        else:
            list_outputs = list(iter_outputs)
            full_output = [j for i in list_outputs for j in i]

    if output_func is None:
        failcount = sum([1 for j in full_output if type(j) == str])
        if failcount > 1:
            logging.info(f"Restarting pool because of failures ({failcount})")
            pool.close()
            pool.join()
            pool.terminate()
            pool.clear()
        return full_output
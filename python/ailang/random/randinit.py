import ailang as al
import numpy as np

from typing import Tuple


def randn(shape: Tuple[int], dtype: al.dtype, device: str = "cpu") -> al.array:
    r"""
    Generate a random array with the given shape, data type, and device.

    Parameters:
    - shape (Tuple[int]): The shape of the array.
    - dtype (al.dtype): The data type of the array.
    - device (str): The device to store the array on.

    Returns:
    - al.array: The generated random array.
    """
    np_array = np.random.randn(*shape).astype(dtype)
    al_array = al.from_numpy(np_array, device=device)
    return al_array

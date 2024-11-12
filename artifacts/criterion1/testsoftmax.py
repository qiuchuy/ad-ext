import argparse
import sys
import os
import numpy as np
from typing import List
import typing
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import Colors
from utils.utils import delimiter, center
import ailang as al

torch.set_printoptions(precision=6)


def gen_random_nparray(shape: typing.Tuple[int], dtype: np.dtype) -> np.ndarray:
    if len(shape):
        random_nparray = np.random.randn(*shape).astype(dtype)
        return random_nparray
    else:
        return dtype(np.random.randn())


def numeric_check(a: al.array, b: np.ndarray):
    return np.allclose(
        a.tolist(), b.tolist(), rtol=args.rtol, atol=args.atol, equal_nan=True
    )


data_type_map = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
}


def test_softmax(input_data: al.array):
    x = al.from_numpy(input_data)
    res = al.standard.softmax(x)
    return res


def torch_softmax(input_data: torch.tensor):
    x = torch.from_numpy(input_data)
    res = torch.softmax(x, -1)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for numpy")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    parser.add_argument(
        "--data_shape", type=int, nargs=2, default=[3, 4], help="Shape of the data"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        help='Data type (e.g., "float32", "float64")',
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.data_type = data_type_map.get(args.data_type, np.float32)  #
    input_data = gen_random_nparray(args.data_shape, args.data_type)
    delimiter("=", color=Colors.GREEN)
    center("[CRITERION 1.1] Softmax", Colors.GREEN)
    delimiter("=", color=Colors.GREEN)

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test op: Softmax", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    shape: {args.data_shape}", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILANG IR", Colors.GREEN)
    ailang_res = test_softmax(input_data)
    torch_res = torch_softmax(input_data).detach().numpy()
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILang's Results is :\n {ailang_res}", Colors.BOLD)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"Pytorch's Results is :\n {torch_res}", Colors.BOLD)
    if numeric_check(ailang_res, torch_res):
        delimiter("=", color=Colors.GREEN)
        center("Numeric Check Passed!", Colors.GREEN)
        delimiter("=", color=Colors.GREEN)
    else:
        delimiter("=", color=Colors.FAIL)
        center("Numeric Check Passed!", Colors.FAIL)
        delimiter("=", color=Colors.FAIL)

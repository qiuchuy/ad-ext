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


def ailang_matmul(lhs_data, rhs_data):
    lhs_data = al.from_numpy(lhs_data)
    rhs_data = al.from_numpy(rhs_data)
    res = al.matmul(lhs_data, rhs_data)

    return res

def ailang_matmul_gpu(lhs_data, rhs_data):
    lhs_data = al.from_numpy(lhs_data, device="gpu")
    rhs_data = al.from_numpy(rhs_data, device="gpu")
    res = al.matmul(lhs_data, rhs_data)

    return res


def torch_matmul(lhs_data, rhs_data):
    lhs_data = torch.from_numpy(lhs_data)
    rhs_data = torch.from_numpy(rhs_data)
    return torch.matmul(lhs_data, rhs_data)

def torch_matmul_gpu(lhs_data, rhs_data):
    lhs_data = torch.from_numpy(lhs_data).cuda()
    rhs_data = torch.from_numpy(rhs_data).cuda()
    return torch.matmul(lhs_data, rhs_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for numpy")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance")
    parser.add_argument(
        "--lhs_data_shape",
        type=int,
        nargs=2,
        default=[3, 4],
        help="Shape of the lhs data",
    )
    parser.add_argument(
        "--rhs_data_shape",
        type=int,
        nargs=2,
        default=[4, 5],
        help="Shape of the rhs data",
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
    lhs_data = gen_random_nparray(args.lhs_data_shape, args.data_type)
    rhs_data = gen_random_nparray(args.rhs_data_shape, args.data_type)
    delimiter("=", color=Colors.GREEN)
    center("[CRITERION 1.3] Matmul", Colors.GREEN)
    delimiter("=", color=Colors.GREEN)

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test op: Matmul", Colors.GREEN)
    Colors.print_color(f"    device : cpu", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    lhs_shape: {args.lhs_data_shape}", Colors.GREEN)
    Colors.print_color(f"    rhs_shape: {args.rhs_data_shape}", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    ailang_res = ailang_matmul(lhs_data, rhs_data)
    torch_res = torch_matmul(lhs_data, rhs_data).detach().numpy()
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

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test op: Matmul", Colors.GREEN)
    Colors.print_color(f"    device : gpu", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    lhs_shape: {args.lhs_data_shape}", Colors.GREEN)
    Colors.print_color(f"    rhs_shape: {args.rhs_data_shape}", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    ailang_res = ailang_matmul_gpu(lhs_data, rhs_data)
    torch_res = torch_matmul(lhs_data, rhs_data).detach().numpy()
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
        center("Numeric Check Failed!", Colors.FAIL)
        delimiter("=", color=Colors.FAIL)

import argparse
import sys
import os
import numpy as np
from typing import List
import typing
import torch
import ailang

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


def ailang_convolution(
    input_data: np.array,
    weight: np.array,
    kernel_size: int,
    stride: int,
    lhs_dilation: int,
    rhs_dilation: int,
    padding: int,
):
    input_data = al.from_numpy(input_data)
    weight = al.from_numpy(weight)
    conv_Layer = ailang.nn.Conv2d(
        3, 2, stride, kernel_size, lhs_dilation, 1, 0, padding
    )
    conv_Layer.weight = weight
    res = conv_Layer(input_data)
    return res


def torch_convolution(
    input_data: np.array,
    weight: np.array,
    kernel_size: int,
    stride: int,
    lhs_dilation: int,
    rhs_dilation: int,
    padding: int,
):
    input_data = torch.from_numpy(input_data)
    weight = torch.from_numpy(weight)
    conv_Layer = torch.nn.Conv2d(
        3,
        2,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=rhs_dilation,
        bias=False,
    )
    conv_Layer.weight.data = weight
    res = conv_Layer(input_data)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for numpy")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    parser.add_argument("--in_channels", type=int, default=3, help="Conv Kernel Size")
    parser.add_argument("--out_channels", type=int, default=2, help="Conv Kernel Size")
    parser.add_argument("--kernel_size", type=int, default=3, help="Conv Kernel Size")
    parser.add_argument("--stride", type=int, default=2, help="Conv kernel Stride")
    parser.add_argument("--padding", type=int, default=0, help="Conv kernel padding")
    parser.add_argument("--dilation", type=int, default=1, help="Conv kernel dilation")
    parser.add_argument(
        "--data_shape",
        type=int,
        nargs=4,
        default=[1, 3, 7, 7],
        help="Shape of the data",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        help='Data type (e.g., "float32", "float64")',
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.data_type = data_type_map.get((args.data_type), np.float32)  #
    input_data = gen_random_nparray((args.data_shape), args.data_type)
    weight = gen_random_nparray(
        (args.out_channels, args.in_channels, args.kernel_size, args.kernel_size),
        args.data_type,
    )
    delimiter("=", color=Colors.GREEN)
    center("[CRITERION 1.2] Convolution", Colors.GREEN)
    delimiter("=", color=Colors.GREEN)

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test op: Convolution", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    shape: {args.data_shape}", Colors.GREEN)
    Colors.print_color(f"    out_channels: {args.out_channels}", Colors.GREEN)
    Colors.print_color(f"    in_channels: {args.in_channels}", Colors.GREEN)
    Colors.print_color(f"    kernel_size: {args.kernel_size}", Colors.GREEN)
    Colors.print_color(f"    stride: {args.stride}", Colors.GREEN)
    Colors.print_color(f"    padding: {args.padding}", Colors.GREEN)
    Colors.print_color(f"    dilation: {args.dilation}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILANG IR", Colors.GREEN)
    ailang_res = ailang_convolution(
        input_data,
        weight,
        args.kernel_size,
        args.stride,
        args.dilation,
        args.dilation,
        args.padding,
    )
    torch_res = (
        torch_convolution(
            input_data,
            weight,
            args.kernel_size,
            args.stride,
            args.dilation,
            args.dilation,
            args.padding,
        )
        .detach()
        .numpy()
    )
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
        center("Numeric Check Not Passed!", Colors.FAIL)
        delimiter("=", color=Colors.FAIL)

#pragma once
#include "array.h"
#include "cblas.h"
#include "dtype.h"
#include <cstdlib>
#include <cstring>
namespace ainl::core {
namespace {

template <typename T>
void conv2d_gemm(const Array &input, const Array &weight, Array &output,
                 const std::vector<int> &stride,
                 const std::vector<int> &padding,
                 const std::vector<int> &dilation) {
    if (!(stride[0] == stride[1] && padding[0] == padding[1] &&
          dilation[0] == dilation[1])) {
        throw std::invalid_argument(
            "[conv2d_gemm] stride_h and stride_w must be equal.");
    }
    int stride_int = stride[0];
    int padding_int = padding[0];
    int dilation_int = dilation[0];
    const int N = input.shape()[0];
    const int in_channels = input.shape()[1];
    const int out_channels = weight.shape()[0];
    const int in_H = input.shape()[2];
    const int in_W = input.shape()[3];
    const int weight_H = weight.shape()[1];
    const int weight_W = weight.shape()[2];
    const int out_H = output.shape()[2];
    const int out_W = output.shape()[3];
    const T *in_ptr = input.data<T>();
    const T *wt_ptr = weight.data<T>();
    int col_H = N * out_H * out_W;
    int col_W = in_channels * weight_H * weight_W;
    int vec_col_size = col_H * col_W;
    T *out_ptr = output.data<T>();

    T *vec_col = (T *)malloc(vec_col_size * sizeof(T));
    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < out_H; ++i) {
            for (int j = 0; j < out_W; ++j) {
                int base_index = ((n * out_H + i) * out_W + j) * (col_W);
                for (int c = 0; c < in_channels; ++c) {
                    for (int ki = 0; ki < weight_H; ++ki) {
                        for (int kj = 0; kj < weight_W; ++kj) {
                            int row = i * stride_int + ki * dilation_int -
                                      padding_int;
                            int col = j * stride_int + kj * dilation_int -
                                      padding_int;
                            if (row >= 0 && row < in_H && col >= 0 &&
                                col < in_W) {
                                vec_col[base_index + c * weight_H * weight_W +
                                        ki * weight_H + kj] =
                                    in_ptr[((n * in_channels + c) * in_H +
                                            row) *
                                               in_W +
                                           col];
                            } else {
                                vec_col[base_index + c * weight_H * weight_W +
                                        ki * weight_H + kj] =
                                    0.0f; // Padding with zeros
                            }
                        }
                    }
                }
            }
        }
    }

    // flatten kernel
    // T *kernel_flat = (T *)malloc(in_channels * weight_H * weight_W *
    // sizeof(T)); for (int n = 0; n < N; ++n) {
    //     for (int c_out = 0; c_out < out_channels; ++c_out) {
    //         for (int i = 0; i < weight_H; ++i) {
    //             for (int j = 0; j < weight_W; ++j) {
    //                 kernel_flat[(n * out_channels * weight_H * weight_W) +
    //                             (c_out * weight_H * weight_W) + (i *
    //                             weight_H) + j] =
    //                     wt_ptr[(n * out_channels * weight_H * weight_W) +
    //                            (c_out * weight_H * weight_W) + (i * weight_H)
    //                            + j];
    //             }
    //         }
    //     }
    // }
    // once lauchn
    std::cout << "out" << out_ptr[0] << std::endl;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, col_H, 1, col_W, 1.0,
                vec_col, col_W, wt_ptr, 1, 0.0, out_ptr, 1);
    // std::cout << "out" << out_ptr[0] << std::endl;
}

template <typename T>
void conv2d_op(const Array &input, const Array &weight, Array &output,
               const std::vector<int> &stride, const std::vector<int> &padding,
               const std::vector<int> &dilation) {
    const int N = input.shape()[0];
    const int in_channels = input.shape()[1];
    const int out_channels = weight.shape()[0];
    const int in_H = input.shape()[2];
    const int in_W = input.shape()[3];
    const int weight_H = weight.shape()[1];
    const int weight_W = weight.shape()[2];
    const int out_H = output.shape()[2];
    const int out_W = output.shape()[3];
    const T *in_ptr = input.data<T>();
    const T *wt_ptr = weight.data<T>();
    T *out_ptr = output.data<T>();
    // auto [N, C_in, H_in, W_in] = in_shape;
    // auto [C_out, KernelSizeH, KernelSizeW, C_in_] =
    // weight_shape; out [N, C_out, H_out, W_out]
    const size_t in_stride_N = input.strides()[0];
    const size_t in_stride_C = input.strides()[1];
    const size_t in_stride_H = input.strides()[2];
    const size_t in_stride_W = input.strides()[3];

    const size_t wt_stride_O = weight.strides()[0];
    const size_t wt_stride_H = weight.strides()[1];
    const size_t wt_stride_W = weight.strides()[2];
    const size_t wt_stride_C = weight.strides()[3];

    const size_t out_stride_N = output.strides()[0];
    const size_t out_stride_O = output.strides()[1];
    const size_t out_stride_H = output.strides()[2];
    const size_t out_stride_W = output.strides()[3];

    auto point_conv = [&](const T *in_ptr, const T *wt_ptr, T *out_ptr, int oH,
                          int oW) {
        // out_ptr += out_stride_H * oH + out_stride_W * oW;

        int temp = 1;
        temp++;
        // in one kernel, base
        int ih_base = oH * wt_stride_O - padding[0];
        int iw_base = oW * wt_stride_H - padding[1];

        for (int out_channel = 0; out_channel < out_channels;
             ++out_channel) { // channel wise
            float res = 0.;
            // weight element wise
            for (int wh = 0; wh < weight_H; ++wh) {
                for (int ww = 0; ww < weight_W; ++ww) {

                    if (ih_base >= 0 && ih_base < in_H && iw_base >= 0 &&
                        iw_base < in_W) {
                        // not implemnet flip/dilation
                        // const T *weight_ptr_pt =
                        //     wt_ptr + wh * wt_stride_H + ww *
                        //     wt_stride_W;
                        // const T *in_ptr_pt = in_ptr + ih_base
                        // * in_stride_H +
                        //                      iw_base *
                        //                      in_stride_W;
                        int temp = 1;
                        temp++;
                        for (int in_channel = 0; in_channel < in_channels;
                             in_channel++) {
                            // res +=
                            // static_cast<float>(in_ptr_pt[0])
                            // *
                            //        static_cast<float>(weight_ptr_pt[0]);
                            // in_ptr_pt += in_stride_C;
                            // weight_ptr_pt += wt_stride_C;
                            int temp = 1;
                            temp++;
                        }
                    } // c
                } // ww
            } // wh
            // printf("[CONV2D DEBUG INFO] at line %d at file %s
            // and res %f\n",
            //        __LINE__, __FILE__, 1.0);

            // out_ptr[0] = static_cast<T>(res);
            // out_ptr += out_stride_O;
            // wt_ptr += wt_stride_O;
            int temp = 1;
            temp++;
        }
    };
}
void conv2d_dispatch(const Array &input, const Array &weight, Array &output,
                     const std::vector<int> &stride,
                     const std::vector<int> &padding,
                     const std::vector<int> &dilation) {
    switch (input.dtype().type) {
    case Dtype::DataType::Float32Type:
        conv2d_gemm<float>(input, weight, output, stride, padding, dilation);
        break;
    case Dtype::DataType::Float64Type:
        conv2d_gemm<double>(input, weight, output, stride, padding, dilation);
        break;
    default:
        throw std::invalid_argument(
            "[conv2d_dispatch] dont support other types "
            "besides float or double.");
        break;
    }
}

} // namespace
} // namespace ainl::core

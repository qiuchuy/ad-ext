#pragma once
#include "array.h"
#include "dtype.h"
namespace ainl::core {
namespace {
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
    // auto [C_out, KernelSizeH, KernelSizeW, C_in_] = weight_shape;
    // out [N, C_out, H_out, W_out]
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
                        //     wt_ptr + wh * wt_stride_H + ww * wt_stride_W;
                        // const T *in_ptr_pt = in_ptr + ih_base * in_stride_H +
                        //                      iw_base * in_stride_W;
                        int temp = 1;
                        temp++;
                        for (int in_channel = 0; in_channel < in_channels;
                             in_channel++) {
                            // res += static_cast<float>(in_ptr_pt[0]) *
                            //        static_cast<float>(weight_ptr_pt[0]);
                            // in_ptr_pt += in_stride_C;
                            // weight_ptr_pt += wt_stride_C;
                            int temp = 1;
                            temp++;
                        }
                    } // c
                }     // ww
            }         // wh
            // printf("[CONV2D DEBUG INFO] at line %d at file %s and res %f\n",
            //        __LINE__, __FILE__, 1.0);

            // out_ptr[0] = static_cast<T>(res);
            // out_ptr += out_stride_O;
            // wt_ptr += wt_stride_O;
            int temp = 1;
            temp++;
        }
    };
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < out_H; ++oh) {
            for (int ow = 0; ow < out_W; ++ow) {
                point_conv(in_ptr, wt_ptr, out_ptr, oh, ow);
            }
        }
    }
}

void conv2d_dispatch(const Array &input, const Array &weight, Array &output,
                     const std::vector<int> &stride,
                     const std::vector<int> &padding,
                     const std::vector<int> &dilation) {
    switch (input.dtype().type) {
    case Dtype::DataType::Float32Type:
        conv2d_op<float>(input, weight, output, stride, padding, dilation);
        break;
    case Dtype::DataType::Float64Type:
        conv2d_op<double>(input, weight, output, stride, padding, dilation);
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

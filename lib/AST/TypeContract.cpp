#include "ailang/IR/TypeContract.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/Type.h"
#include "ailang/IR/Value.h"

namespace ainl::ir {

TypePtr matmulTypeContract(const TypePtr &lhsType, const TypePtr &rhsType) {
  // Type Checking
  if (!lhsType->isTensorType() || !rhsType->isTensorType()) {
    throw ainl::core::AINLError("matmul operator only applies to two tensors.");
  }
  TensorTypePtr lhsTensorType = SAFE_TYPE_DOWNCAST(lhsType, TensorType);
  TensorTypePtr rhsTensorType = SAFE_TYPE_DOWNCAST(rhsType, TensorType);
  std::vector<ValuePtr> lhsShape = lhsTensorType->getShape();
  std::vector<ValuePtr> rhsShape = rhsTensorType->getShape();
  if (*lhsShape.back() != *rhsShape.front()) {
    throw ainl::core::AINLError("tensor shapes are not matched for matmul.");
  }

  // Construct the result type
  std::vector<ValuePtr> lhsShapeWithoutLastDim(lhsShape.begin(),
                                               lhsShape.end() - 1);
  std::vector<ValuePtr> rhsShapeWithoutFirstDim(rhsShape.begin() + 1,
                                                rhsShape.end());
  std::vector<ValuePtr> matmulShape(lhsShapeWithoutLastDim.begin(),
                                    lhsShapeWithoutLastDim.end());
  matmulShape.insert(matmulShape.end(), rhsShapeWithoutFirstDim.begin(),
                     rhsShapeWithoutFirstDim.end());
  TypePtr elementType = lhsTensorType->getElementType();
  return TensorType::create(elementType, matmulShape);
}

TypePtr transposeTypeContract(const TypePtr &inType) {
  // Type Checking
  if (!inType->isTensorType()) {
    throw ainl::core::AINLError("transpose operator only applies to tensors.");
  }
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();

  // Construct the result type
  std::reverse(inTensorShape.begin(), inTensorShape.end());
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, inTensorShape);
}

TypePtr addTypeContract(const TypePtr &lhsType, const TypePtr &rhsType) {
  if (!lhsType->isTensorType() || !rhsType->isTensorType()) {
    throw ainl::core::AINLError("add operator only applies to two tensors.");
  }
  TensorTypePtr lhsTensorType = SAFE_TYPE_DOWNCAST(lhsType, TensorType);
  TensorTypePtr rhsTensorType = SAFE_TYPE_DOWNCAST(rhsType, TensorType);

  std::vector<ValuePtr> lhsShape = lhsTensorType->getShape();
  std::vector<ValuePtr> rhsShape = rhsTensorType->getShape();
  // std::vector<ValuePtr> addShape = std::move(lhsShape);
  if (lhsShape.size() != rhsShape.size()) {
    throw ainl::core::AINLError(
        "two tensor dont not have the same dim for add.");
  }
  if (!std::equal(lhsShape.begin(), lhsShape.end(), rhsShape.begin(),
                  [](ValuePtr &lhs, ValuePtr &rhs) { return *lhs == *rhs; })) {
    throw ainl::core::AINLError("tensor shapes are not matched for add.");
  }
  // TypePtr elementType = lhsTensorType->getElementType();
  TypePtr elementType;
  TypePtr lhsBaseType = lhsTensorType->getElementType();
  TypePtr rhsBaseType = rhsTensorType->getElementType();
  if (lhsBaseType->compare((*rhsBaseType))) {
    elementType = rhsBaseType;
  } else {
    elementType = lhsBaseType;
  }
  return TensorType::create(elementType, lhsShape);
}

// TypePtr broadcastTypeContract(const TypePtr &inType) {
//     if (!inType->isTensorType()) {
//         throw ainl::core::AINLError(
//             "broadcastbroadcast operator only applies to tensors.");
//     }
//     TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
//     std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
//     TypePtr elementType = inTensorType->getElementType();
//     return TensorType::create(elementType, inTensorShape);
// }

TypePtr reluTypeContract(const TypePtr &inType) {
  if (!inType->isTensorType()) {
    throw ainl::core::AINLError("relu operator only applies to tensors.");
  }
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, inTensorShape);
}

TypePtr meanTypeContract(const TypePtr &inType) {
  if (!inType->isTensorType()) {
    throw ainl::core::AINLError("mean operator only applies to tensors.");
  }
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  std::vector<ValuePtr> outTensorshape;
  outTensorshape.emplace_back(Literal::create(1));
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, outTensorshape);
}

TypePtr maxpool2dTypeContract(const TypePtr &inType) {
  // IR写完再来补参数

  // Union[int, Tuple[int, int]]

  // input size (N,C,H,W) out (N,C,H_out,W_out)
  /*
    Shape:
      - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in},
    W_{in})`
      - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out},
    W_{out})`, where

        .. math::
            H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} -
    \text{dilation[0]} \times (\text{kernel\_size[0]} - 1) -
    1}{\text{stride[0]}} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} -
    \text{dilation[1]} \times (\text{kernel\_size[1]} - 1) -
    1}{\text{stride[1]}} + 1\right\rfloor
  */
  if (!inType->isTensorType()) {
    throw ainl::core::AINLError("maxpool2d operator only applies to tensors.");
  }
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  std::vector<int> inConcreateShape = inTensorType->getConcreteShape();
  if (inConcreateShape.size() != 4 && inConcreateShape.size() != 3) {
    throw ainl::core::AINLError(
        "expected 4d (N,C,H,W) or 3d (C,H,W), input dim is not matched.");
  }

  uint32_t padding_h = 0;
  uint32_t dilation_h = 1;
  uint32_t kernel_size_h = 3;
  uint32_t stride_h = 2;

  uint32_t padding_w = 0;
  uint32_t dilation_w = 1;
  uint32_t kernel_size_w = 3;
  uint32_t stride_w = 2;

  uint32_t shapeSize = inConcreateShape.size();
  uint32_t W = inConcreateShape[shapeSize - 2];
  uint32_t H = inConcreateShape[shapeSize - 1];
  uint32_t H_out =
      (H + 2 * padding_h - dilation_h * (kernel_size_h - 1)) / stride_h + 1;
  uint32_t W_out =
      (W + 2 * padding_w - dilation_w * (kernel_size_w - 1)) / stride_w + 1;

  inConcreateShape[shapeSize - 2] = H_out;
  inConcreateShape[shapeSize - 1] = W_out;

  std::vector<ValuePtr> outTensorShape;
  for (const auto &dim : inConcreateShape) {
    outTensorShape.push_back(Literal::create(dim));
  }
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, outTensorShape);
}

TypePtr convolutionTypeContract(const TypePtr &inputType,
                                const TypePtr &weightType) {
  if (!inputType->isTensorType() || !weightType->isTensorType()) {
    throw ainl::core::AINLError(
        "convolution operator only applies to tensors.");
  }
  TensorTypePtr inputTensorType = SAFE_TYPE_DOWNCAST(inputType, TensorType);
  TensorTypePtr weightTensorType = SAFE_TYPE_DOWNCAST(weightType, TensorType);
  std::vector<ValuePtr> inputTensorShape = inputTensorType->getShape();
  std::vector<ValuePtr> weightTensorShape = weightTensorType->getShape();
  std::vector<int> inputConcreateShape = inputTensorType->getConcreteShape();
  std::vector<int> weightConcreateShape = weightTensorType->getConcreteShape();

  if (inputConcreateShape.size() != 4 && weightConcreateShape.size() != 3) {
    throw ainl::core::AINLError(
        "expected input 4d (N,H,W,C) and weight(H,W,C,O), input or weight "
        "dim is not matched.");
  }

  /*
  in_channels (int) – Number of channels in the input image
  out_channels (int) – Number of channels produced by the convolution
  kernel_size (int or tuple) – Size of the convolving kernel
  stride (int or tuple, optional) – Stride of the convolution. Default: 1
  padding (int, tuple or str, optional) – Padding added to all four sides of
  the input. Default: 0 padding_mode (str, optional) – 'zeros', 'reflect',
  'replicate' or 'circular'. Default: 'zeros' dilation (int or tuple,
  optional) – Spacing between kernel elements. Default: 1 groups (int,
  optional) – Number of blocked connections from input channels to output
  channels. Default: 1 bias (bool, optional) – If True, adds a learnable bias
  to the output. Default: True
  */

  /*
   in stablehlo the lhs(input img) corresppponds to the NHWC layout.
   And the weights corresponds to HWIO.
   output corresponds to NHWC layout*/
  int padding_h = 0;
  int dilation_h = 1;
  int stride_h = 2;

  int padding_w = 0;
  int dilation_w = 1;
  int stride_w = 2;

  int N = inputConcreateShape[0];
  int H = inputConcreateShape[1];
  int W = inputConcreateShape[2];
  int C = inputConcreateShape[3];

  int kernel_size_h = weightConcreateShape[0];
  int kernel_size_w = weightConcreateShape[1];
  int I = weightConcreateShape[2];
  int O = weightConcreateShape[3];
  assert(C == I);
  int H_out =
      (H + 2 * padding_h - dilation_h * (kernel_size_h - 1)) / stride_h + 1;
  int W_out =
      (W + 2 * padding_w - dilation_w * (kernel_size_w - 1)) / stride_w + 1;
  std::vector<ValuePtr> outTensorShape = {
      Literal::create(N),
      Literal::create(H_out),
      Literal::create(W_out),
      Literal::create(O),
  };
  TypePtr elementType = inputTensorType->getElementType();
  return TensorType::create(elementType, outTensorShape);
}

TypePtr batchnorm2dTypeContract(const TypePtr &inType, const TypePtr &scaleType,
                                const TypePtr &offsetType,
                                const TypePtr &meanType,
                                const TypePtr &varianceType) {
  // same shape as input
  if (!inType->isTensorType() || !scaleType->isTensorType() ||
      !offsetType->isTensorType() || !meanType->isTensorType() ||
      !varianceType->isTensorType()) {
    throw ainl::core::AINLError(
        "batchnorm2d operator only applies to tensors.");
  }
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  std::vector<int> inConcreateShape = inTensorType->getConcreteShape();
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, inTensorShape); // with same shape
}

TypePtr compareTypeContract(const TypePtr &lhsType, const TypePtr &rhsType) {
  if (!lhsType->isTensorType() || !rhsType->isTensorType()) {
    throw ainl::core::AINLError("[typeinfer] compare operator type infer "
                                "only applies to two tensors.");
  }
  auto tensorType = asType<TensorType>(lhsType);
  return TensorType::create(SingletonTypePtr<BoolType>::get(),
                            tensorType->getShape());
}

TypePtr ifTypeContract(const TypePtr &condType, const TypePtr &trueType,
                       const TypePtr &falseType) {
  if (!trueType->equals(falseType)) {
    throw ainl::core::AINLError("[typeinfer] if operator type infer only "
                                "applies to two tensors with "
                                "the same type.");
  }
  return trueType;
}

TypeContract::TypeContract() {
  registerContract("matmul", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator matmul");
    }
    return matmulTypeContract((args[0]), (args[1]));
  });

  registerContract("transpose", [](std::vector<TypePtr> args) {
    if (args.size() != 1) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator transpose");
    }
    return transposeTypeContract((args[0]));
  });

  registerContract("add", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError("Invalid argument number for operator add");
    }
    return addTypeContract((args[0]), (args[1]));
  });

  // registerContract("broadcast", [](std::vector<TypePtr> args) {
  //     if (args.size() != 1) {
  //         throw ainl::core::AINLError(
  //             "Invalid argument number for operator broadcast");
  //     }
  //     return broadcastTypeContract((args[0]));
  // });

  registerContract("relu", [](std::vector<TypePtr> args) {
    if (args.size() != 1) {
      throw ainl::core::AINLError("Invalid argument number for operator relu");
    }
    return reluTypeContract((args[0]));
  });

  // kernel_size (Union[int, Tuple[int, int]]) – the size of the window to
  // take a max over stride (Union[int, Tuple[int, int]]) – the stride of the
  // window. Default value is kernel_size padding (Union[int, Tuple[int,
  // int]]) –都arameter that controls the
  // stride of elements in the window 暂时没加 return_indices (bool) – if
  // True, will return the max indices along with the outputs. Useful for
  // torch.nn.MaxUnpool2d later ceil_mode (bool) – when True, will use ceil
  // instead of floor to compute the output shape
  // 先假设我们的maxpool的参数是固定，动态后面再看看框架改动

  registerContract("maxpool2d", [](std::vector<TypePtr> args) {
    if (args.size() != 1) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator maxpool2d");
    }
    return maxpool2dTypeContract((args[0]));
  });
  registerContract("convolution", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator convolution");
    }
    return convolutionTypeContract((args[0]), (args[1]));
  });
  registerContract("batchnorm2d", [](std::vector<TypePtr> args) {
    if (args.size() != 5) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator bacthnorm2d");
    }
    return batchnorm2dTypeContract(args[0], args[1], args[2], args[3], args[4]);
  });
  registerContract("mean", [](std::vector<TypePtr> args) {
    if (args.size() != 1) {
      throw ainl::core::AINLError("Invalid argument number for operator mean");
    }
    return meanTypeContract(args[0]);
  });
  registerContract("eq", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError("Invalid argument number for operator eq");
    }
    return compareTypeContract((args[0]), (args[1]));
  });
  registerContract("ne", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError("Invalid argument number for operator ne");
    }
    return compareTypeContract((args[0]), (args[1]));
  });
  registerContract("gt", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError("Invalid argument number for operator gt");
    }
    return compareTypeContract((args[0]), (args[1]));
  });
  registerContract("ge", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError("Invalid argument number for operator ge");
    }
    return compareTypeContract((args[0]), (args[1]));
  });
  registerContract("le", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError("Invalid argument number for operator le");
    }
    return compareTypeContract((args[0]), (args[1]));
  });
  registerContract("lt", [](std::vector<TypePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError("Invalid argument number for operator lt");
    }
    return compareTypeContract((args[0]), (args[1]));
  });
  registerContract("ifop", [](std::vector<TypePtr> args) {
    if (args.size() != 3) {
      throw ainl::core::AINLError("Invalid argument number for operator ifop");
    }
    return ifTypeContract((args[0]), (args[1]), (args[2]));
  });
}

TypeContract &getTypeContract() {
  static TypeContract contract;
  return contract;
}

TypePtr resolveContract(const std::string &name, std::vector<TypePtr> args) {
  return getTypeContract().resolveContract(name, std::move(args));
}

} // namespace ainl::ir
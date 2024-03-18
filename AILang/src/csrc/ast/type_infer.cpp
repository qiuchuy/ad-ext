#include "ast/type_infer.h"
#include "ir/symbol.h"
#include "ir/type.h"
#include "utils/utils.h"

namespace ainl::ir {

TypePtr matmulTypeContract(const TypePtr &lhsType, const TypePtr &rhsType) {
    // Type Checking
    if (!lhsType->isTensorType() || !rhsType->isTensorType()) {
        throw ainl::core::AINLError(
            "matmul operator only applies to two tensors.");
    }
    TensorTypePtr lhsTensorType = SAFE_TYPE_DOWNCAST(lhsType, TensorType);
    TensorTypePtr rhsTensorType = SAFE_TYPE_DOWNCAST(rhsType, TensorType);
    std::vector<ValuePtr> lhsShape = lhsTensorType->getShape();
    std::vector<ValuePtr> rhsShape = rhsTensorType->getShape();
    if (*lhsShape.back() != *rhsShape.front()) {
        throw ainl::core::AINLError(
            "tensor shapes are not matched for matmul.");
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
        throw ainl::core::AINLError(
            "transpose operator only applies to tensors.");
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
        throw ainl::core::AINLError(
            "add operator only applies to two tensors.");
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
    if (!std::equal(
            lhsShape.begin(), lhsShape.end(), rhsShape.begin(),
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

TypePtr reluTypeContract(const TypePtr &inType) {
    if (!inType->isTensorType()) {
        throw ainl::core::AINLError("relu operator only applies to tensors.");
    }
    TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
    std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
    TypePtr elementType = inTensorType->getElementType();
    return TensorType::create(elementType, inTensorShape);
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
        throw ainl::core::AINLError(
            "maxpool2d operator only applies to tensors.");
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

TypePtr convolutionTypeContract(const TypePtr &inType) {
    if (!inType->isTensorType()) {
        throw ainl::core::AINLError(
            "convolution operator only applies to tensors.");
    }
    TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
    std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
    std::vector<int> inConcreateShape = inTensorType->getConcreteShape();
    if (inConcreateShape.size() != 4 && inConcreateShape.size() != 3) {
        throw ainl::core::AINLError(
            "expected 4d (N,C,H,W) or 3d (C,H,W), input dim is not matched.");
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

TypePtr batchnorm2dTypeContract(const TypePtr &inType) {
    if (!inType->isTensorType()) {
        throw ainl::core::AINLError(
            "batchnorm2d operator only applies to tensors.");
    }
    TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
    std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
    std::vector<int> inConcreateShape = inTensorType->getConcreteShape();
    if (inConcreateShape.size() != 4) {
        throw ainl::core::AINLError(
            "expected 4d (N,C,H,W) input dim is not matched.");
    }
    TypePtr elementType = inTensorType->getElementType();
    return TensorType::create(elementType, inTensorShape);
}

void TypeInfer::initLibraryOperatorTypeContract() {
    contract.registerContract("matmul", [](std::vector<TypePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator matmul");
        }
        return matmulTypeContract((args[0]), (args[1]));
    });

    contract.registerContract("transpose", [](std::vector<TypePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator transpose");
        }
        return transposeTypeContract((args[0]));
    });

    contract.registerContract("add", [](std::vector<TypePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator add");
        }
        return addTypeContract((args[0]), (args[1]));
    });

    contract.registerContract("relu", [](std::vector<TypePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator relu");
        }
        return reluTypeContract((args[0]));
    });

    // kernel_size (Union[int, Tuple[int, int]]) – the size of the window to
    // take a max over stride (Union[int, Tuple[int, int]]) – the stride of the
    // window. Default value is kernel_size padding (Union[int, Tuple[int,
    // int]]) – Implicit negative infinity padding to be added on both sides
    // dilation (Union[int, Tuple[int, int]]) – a parameter that controls the
    // stride of elements in the window 暂时没加 return_indices (bool) – if
    // True, will return the max indices along with the outputs. Useful for
    // torch.nn.MaxUnpool2d later ceil_mode (bool) – when True, will use ceil
    // instead of floor to compute the output shape
    // 先假设我们的maxpool的参数是固定，动态后面再看看框架改动

    contract.registerContract("maxpool2d", [](std::vector<TypePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator maxpool2d");
        }
        return maxpool2dTypeContract((args[0]));
    });
    contract.registerContract("convolution", [](std::vector<TypePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator convolution");
        }
        return convolutionTypeContract((args[0]));
    });
    contract.registerContract("batchnorm2d", [](std::vector<TypePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator bacthnorm2d");
        }
        return batchnorm2dTypeContract((args[0]));
    });
}
TypeInfer::TypeInfer(const std::vector<std::string> &args,
                     const std::vector<TypePtr> &types) {

    assert(args.size() == types.size());
    initLibraryOperatorTypeContract();
    size_t len = args.size();
    for (size_t i = 0; i < len; i++)
        typedParams[args[i]] = types[i];
}

void TypeInfer::visitModule(ModuleNode *node) {}

void TypeInfer::visitBinaryOp(BinaryOpNode *node) {
    TypePtr lhsType = node->getLHS()->getType();
    TypePtr rhsType = node->getRHS()->getType();
    if (lhsType->compare(*rhsType)) {
        node->setType(rhsType);
    } else {
        node->setType(lhsType);
    }
}

void TypeInfer::visitCall(CallNode *node) {
    std::string funcName = node->getCallFunction()->getName();
    // for now, we use "::" to specify namespace
    size_t lastNamespace = funcName.find_last_of("::");
    if (lastNamespace != std::string::npos) {
        // This is a library function call
        std::string libraryFunction = (funcName.substr(lastNamespace + 1));
        std::vector<Expr> callArgs = node->getCallArgs();
        std::vector<TypePtr> argTypes;
        for (const auto &arg : callArgs) {
            argTypes.push_back(arg->getType());
        }
        TypePtr returnType =
            contract.resolveContract(libraryFunction, argTypes);
        node->setType(returnType);
    } else {
        TypePtr returnType =
            SAFE_TYPE_DOWNCAST(
                (env->lookup(node->getCallFunction()->getName())->getType()),
                FunctionType)
                ->getReturnType();
        node->setType(returnType);
    }
}

void TypeInfer::visitConstant(ConstantNode *node) {
    std::string value = node->getValue();
    // [TODO] A naive method, improve this
    if (value.find('.') != std::string::npos) {
        node->setType(FloatTypePtr::get());
    } else if (value == "True" || value == "False") {
        node->setType(BoolTypePtr::get());
    } else {
        node->setType(IntTypePtr::get());
    }
}

void TypeInfer::visitExpr(ExprNode *node) {}

void TypeInfer::visitFunctionDef(FunctionDefNode *node) {
    env = new Environment();
    auto params = node->getParams();
    for (const auto &param : params) {
        env->insertSymbol(param, typedParams[param]);
    }
    curFunc = node->getName();
}

void TypeInfer::visitReturn(ReturnNode *node) {
    std::vector<TypePtr> types;
    for (const auto &param : typedParams) {
        types.push_back(param.second);
    }
    TypePtr returnType = node->getReturnValue()->getType();
    FunctionTypePtr type =
        FunctionType::create(TupleType::createUnnamedTuple(types), returnType);
    env->insertSymbol(curFunc, type);
}

void TypeInfer::visitCompare(CompareNode *node) {
    node->setType(BoolTypePtr::get());
}

void TypeInfer::visitStmt(StmtNode *node) {}

void TypeInfer::visitUnaryOp(UnaryOpNode *node) {
    Expr operand = node->getOperand();
    node->setType(operand->getType());
}

void TypeInfer::visitVar(VarNode *node) {
    TypePtr type = env->lookup(node->getName())->getType();
    node->setType(type);
}

void TypeInfer::visitVarDef(VarDefNode *node) {}

void TypeInfer::visitTuple(TupleNode *node) {
    std::vector<TypePtr> types;
    for (const auto &expr : node->getElems()) {
        types.push_back(expr->getType());
    }
    node->setType(TupleType::createUnnamedTuple(types));
}

void TypeInfer::visitBind(BindNode *node) {
    TypePtr sourceType = node->getSource()->getType();
    auto targets = node->getTargets();
    size_t targetSize = targets.size();
    if (targetSize == 0)
        throw ainl::core::AINLError("Illegal Bind statement.");
    if (targetSize > 1) {
        for (auto &target : targets) {
            target->setType(sourceType);
            env->insertSymbol(target->getName(), sourceType);
        }
        return;
    }
    auto target = targets.back();
    if (target->isTupleNode()) {
        TupleTypePtr sourceTupleType =
            SAFE_TYPE_DOWNCAST(sourceType, TupleType);
        Tuple targetTupleNode = SAFE_AST_DOWNCAST(target, TupleNode);
        assert(sourceTupleType->getTypes().size() ==
               targetTupleNode->getElems().size());
        size_t len = sourceTupleType->getTypes().size();
        for (size_t i = 0; i < len; i++) {
            targetTupleNode->getElems()[i]->setType(
                sourceTupleType->getTypes()[i]);
            env->insertSymbol(targetTupleNode->getElems()[i]->getName(),
                              sourceTupleType->getTypes()[i]);
        }
    } else {
        target->setType(sourceType);
        env->insertSymbol(target->getName(), sourceType);
    }
}

void TypeInfer::visitWhile(WhileNode *node) {}

void TypeInfer::visitIf(IfNode *node) {}
} // namespace ainl::ir
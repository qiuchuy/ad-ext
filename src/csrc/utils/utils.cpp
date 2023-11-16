#include "utils.h"

BinaryOpNode::BinaryOpKind BinaryOpASTHelper(std::string OpKind) {
    DISPATCH_BINARYOP(Add)
    DISPATCH_BINARYOP(Sub)
    DISPATCH_BINARYOP(Mul)
    DISPATCH_BINARYOP(Div)
    DISPATCH_BINARYOP(FloorDiv)
    DISPATCH_BINARYOP(Mod)
    DISPATCH_BINARYOP(Pow)
    DISPATCH_BINARYOP(LShift)
    DISPATCH_BINARYOP(RShift)
    DISPATCH_BINARYOP(BitOr)
    DISPATCH_BINARYOP(BitXor)
    DISPATCH_BINARYOP(BitAnd)
    DISPATCH_BINARYOP(MatMult)
    throw AINLError("Unsupported BinaryOpNode when transforming AST.");
}

UnaryOpNode::UnaryOpKind UnaryOpASTHelper(std::string OpKind) {
    DISPATCH_UNARYOP(UAdd)
    DISPATCH_UNARYOP(USub)
    DISPATCH_UNARYOP(Not)
    DISPATCH_UNARYOP(Invert)
    throw AINLError("Unsupported UnaryOpNode when transforming AST.");
}

CompareNode::CompareOpKind CompareOpASTHelper(std::string OpKind) {
    DISPATCH_COMPAREOP(Eq)
    DISPATCH_COMPAREOP(NotEq)
    DISPATCH_COMPAREOP(Lt)
    DISPATCH_COMPAREOP(LtE)
    DISPATCH_COMPAREOP(Gt)
    DISPATCH_COMPAREOP(GtE)
    DISPATCH_COMPAREOP(Is)
    DISPATCH_COMPAREOP(In)
    DISPATCH_COMPAREOP(NotIn)
    throw AINLError("Unsupported CompareOpNode when transforming AST.");
}

std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos)
        return "";

    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

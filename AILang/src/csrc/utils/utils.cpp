#include "utils/utils.h"

namespace ainl::core {

ainl::ir::BinaryOpNode::BinaryOpKind BinaryOpASTHelper(std::string OpKind) {
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

ainl::ir::UnaryOpNode::UnaryOpKind UnaryOpASTHelper(std::string OpKind) {
  DISPATCH_UNARYOP(UAdd)
  DISPATCH_UNARYOP(USub)
  DISPATCH_UNARYOP(Not)
  DISPATCH_UNARYOP(Invert)
  throw AINLError("Unsupported UnaryOpNode when transforming AST.");
}

ainl::ir::CompareNode::CompareOpKind CompareOpASTHelper(std::string OpKind) {
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

int caseInsensitiveStrcmp(const std::string &str1, const std::string &str2) {
  // Get the minimum length of the two strings
  size_t min_length = std::min(str1.length(), str2.length());

  for (size_t i = 0; i < min_length; ++i) {
    if (std::tolower(str1[i]) < std::tolower(str2[i])) {
      return -1;
    } else if (std::tolower(str1[i]) > std::tolower(str2[i])) {
      return 1;
    }
  }

  // If the common prefix is the same, compare the lengths
  if (str1.length() < str2.length()) {
    return -1;
  } else if (str1.length() > str2.length()) {
    return 1;
  }

  return 0; // Strings are equal
}
} // namespace ainl::core

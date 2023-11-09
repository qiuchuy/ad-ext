#include "ast_node.h"

std::array<std::string, 4> UnaryOpString = {"+", "-", "~", "!"};
std::array<std::string, 13> BinaryOpString = {
    "+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "@"};
std::array<std::string, 10> CompareOpString = {
    "==", "!=", "<", "<=", ">", ">=", "is", "is not", "in", "not in"};

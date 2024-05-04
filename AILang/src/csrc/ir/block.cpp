#include "ir/block.h"

namespace ainl::ir {
int Block::blockCount = 0;

Block::Block() {
  paramNode = nullptr;
  returnNode = nullptr;
  beginNode = new Node();
  endNode = new Node();
  beginNode->setNext(endNode);
  endNode->setPrev(beginNode);
  beginBlock = endBlock = nullptr;
}

Block::Block(int idx) {
  paramNode = nullptr;
  returnNode = nullptr;
  beginNode = new Node();
  endNode = new Node();
  beginNode->setNext(endNode);
  endNode->setPrev(beginNode);
  beginBlock = endBlock = nullptr;
  label = "b" + std::to_string(idx);
}

std::vector<ValuePtr> Block::getParams() { return paramNode->getParams(); }
void Block::insertNodeAtHead(NodePtr Node) { beginNode->insertAfter(Node); }
void Block::insertNodeAtEnd(NodePtr Node) { endNode->insertBefore(Node); }

Block::operator std::string() const {
  std::string str;
  str.append("{\n");
  // str.append(label + ":\n");
  for (auto node = (NodePtr)beginNode->next; node->next != nullptr;
       node = (NodePtr)node->next) {
    str.append("\t");
    str.append(std::string(*node));
    str.append("\n");
  }
  str.append("}\n");
  return str;
}

Block::BlockIterator Block::begin() {
  return BlockIterator(beginNode, paramNode, returnNode);
}

Block::BlockIterator Block::end() {
  return BlockIterator(endNode, paramNode, returnNode);
}

Block::BlockIterator::BlockIterator(NodePtr node, NodePtr paramNode,
                                    NodePtr returnNode)
    : node(node), paramNode(paramNode), returnNode(returnNode),
      beginNode(beginNode), endNode(endNode) {}

Block::BlockIterator::reference Block::BlockIterator::operator*() {
  return node;
}

Block::BlockIterator &Block::BlockIterator::operator++() {
  if (node == beginNode) {
    node = (NodePtr)(paramNode);
  } else if (node == paramNode) {
    node = (NodePtr)(beginNode->next);
  } else if (node == endNode) {
    node = (NodePtr)(returnNode);
  } else {
    node = (NodePtr)(node->next);
  }
  return *this;
}

Block::BlockIterator Block::BlockIterator::operator++(int) {
  BlockIterator tmp = *this;
  ++(*this);
  return tmp;
}

bool Block::BlockIterator::operator==(const BlockIterator &rhs) const {
  return node == rhs.node;
}

bool Block::BlockIterator::operator!=(const BlockIterator &rhs) const {
  return node != rhs.node;
}

} // namespace ainl::ir
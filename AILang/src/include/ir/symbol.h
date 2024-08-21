#pragma once

#include <algorithm>
#include <map>
#include <utility>

#include "ir/type.h"
#include "ir/value.h"

namespace ainl::ir {

class Symbol {
public:
  Symbol() = default;
  explicit Symbol(std::string name) : name(std::move(name)) {}
  Symbol(std::string name, TypePtr type)
      : name(std::move(name)), type(std::move(type)) {}
  Symbol(std::string name, TypePtr type, ValuePtr value)
      : name(std::move(name)), type(std::move(type)), value(value) {}
  TypePtr getType() { return type; }
  ValuePtr getValue() { return value; }
  std::string getName() { return name; }
  void setType(TypePtr inType) { this->type = std::move(inType); }
  void setValue(ValuePtr inValue) { this->value = inValue; }

private:
  std::string name;
  TypePtr type;
  ValuePtr value{};
};
using SymbolPtr = Symbol *;

class SymbolTable;
using SymbolTablePtr = SymbolTable *;
class SymbolTable {
public:
  SymbolTable() { parent = nullptr; }
  explicit SymbolTable(SymbolTable *parent) : parent(parent) {
    parent->childs.push_back(this);
  }

  SymbolPtr lookup(const std::string &symbol) {
    if (auto entry = lookup_(symbol)) {
      return entry;
    } else {
      // throw AINLError("symbol " + symbol +
      // " is not found in the symbol table.");
    }
  }
  void insertSymbol(const std::string &name) {
    auto symbol = new Symbol(name);
    symbols[name] = symbol;
  }
  void insertSymbol(const std::string &name, TypePtr type) {
    auto symbol = new Symbol(name, std::move(type));
    symbols[name] = symbol;
  }
  void insertSymbol(const std::string &name, TypePtr type, ValuePtr value) {
    auto symbol = new Symbol(name, std::move(type), value);
    symbols[name] = symbol;
  }
  friend class Environment;

protected:
  SymbolTablePtr getParent() { return parent; }
  std::vector<SymbolTablePtr> getChilds() { return childs; }

private:
  SymbolTablePtr parent;
  std::vector<SymbolTablePtr> childs;
  std::map<std::string, SymbolPtr> symbols;
  SymbolPtr lookup_(const std::string &name) {
    auto it = symbols.find(name);
    if (it != symbols.end()) {
      return it->second;
    }
    if (parent != nullptr) {
      return parent->lookup_(name);
    }
    return nullptr;
  }
};

class Environment {
public:
  Environment() { currentEnv = new SymbolTable(); }
  SymbolPtr lookup(const std::string &symbol) {
    return currentEnv->lookup(symbol);
  }
  void addScope() {
    auto newEnv = new SymbolTable(currentEnv);
    currentEnv = newEnv;
  }
  void insertSymbol(const std::string &name) { currentEnv->insertSymbol(name); }
  void insertSymbol(const std::string &name, TypePtr type) {
    currentEnv->insertSymbol(name, std::move(type));
  }
  void insertSymbol(const std::string &name, TypePtr type, ValuePtr value) {
    currentEnv->insertSymbol(name, std::move(type), value);
  }
  void resolveScope() { currentEnv = resolveScope_(); }

private:
  SymbolTablePtr currentEnv;
  SymbolTablePtr resolveScope_() {
    if (!currentEnv->getParent()) {
      auto upperScope = currentEnv->getParent();
      auto fields = upperScope->getChilds();
      auto iter = std::find(fields.begin(), fields.end(), currentEnv);
      if (iter != upperScope->getChilds().end()) {
        currentEnv = *(iter++);
      } else {
        currentEnv = currentEnv->getParent();
        return resolveScope_();
      }
      return currentEnv;
    } else {
      return currentEnv;
    }
  }
};

extern Environment *env;

} // namespace ainl::ir
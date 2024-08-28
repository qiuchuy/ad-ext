#include "ailang/Core/Array.h"
#include "ailang/Core/Primitive.h"
#include <cstring>
#include <map>

namespace ainl::core {

template <typename T, typename... TArgs> class PrimShapeInferContract {
  public:
    explicit PrimShapeInferContract();

    using FactoryKeyTYPE = std::string;
    using PrimShapeInferFunction = std::function<T(TArgs...)>;
    void registerPrimShapeInferContract(const FactoryKeyTYPE &name,
                                        PrimShapeInferFunction func) {
        function_map[name] = std::move(func);
    }
    // function with attr
    using PrimShapeInferFunctionWithAttribute =
        std::function<T(TArgs..., std::vector<int>)>;
    // overload
    void
    registerPrimShapeInferContract(const FactoryKeyTYPE &name,
                                   PrimShapeInferFunctionWithAttribute func) {
        attr_function_map[name] = std::move(func);
    }

    T resolvePrimShapeInferContract(const FactoryKeyTYPE &name, TArgs... args) {
        if (function_map.find(name) == function_map.end()) {
            throw std::runtime_error("The PrimShape contract of operator [" +
                                     name + "] has not been registered yet.");
        }
        function_map[name](std::forward<TArgs>(args)...);
    }
    T resolvePrimShapeInferContract(const FactoryKeyTYPE &name, TArgs... args,
                                    std::vector<int> attr) {
        if (attr_function_map.find(name) == attr_function_map.end()) {
            throw std::runtime_error("The PrimShape contract of operator [" +
                                     name + "] has not been registered yet.");
        }
        attr_function_map[name](std::forward<TArgs>(args)..., attr);
    }

  private:
    std::map<FactoryKeyTYPE, PrimShapeInferFunction> function_map;
    std::map<FactoryKeyTYPE, PrimShapeInferFunctionWithAttribute>
        attr_function_map;
};

PrimShapeInferContract<void, const std::vector<Array> &, Array &> &
getPrimShapeInferContract();

void resolvePrimShapeInferContract(const std::string &name,
                                   const std::vector<Array> &inputs,
                                   Array &output);

} // namespace ainl::core
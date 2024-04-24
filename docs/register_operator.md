# Register a New Operator
- Implement a new `Primitive` operator by subclassing `Primitive` class.
    - For JIT, also implement the type contract and node contract for the operator.
- Implement `op` level interface for `Array`
- Implement generic `op` level interface for different types of tracers using macro.
- Bind the operator to python with pybind11 with operator overloading for different tracers.
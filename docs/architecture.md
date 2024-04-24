# Overview
## Array
`Array` serves as a abstraction for constructing high-dimensional arrays within the AINL framework, providing a versatile structure for data representation and manipulation. 
Below is a succinct outline of the `Array` structure:
``` 
class Array {
    ...
    private:
        ... data_;
        ... shape_;
        ... info_;
        ... size_;
        ... dtype_;
};
```
- `data_` represents a pointer to the memory location where the array is stored, forming the backbone of data handling and manipulation within the computational graph. It is important to known that `Array` is underlying immutable. When using `y=x` at Python toplevel, AINL is creating a new `Array` instance, and the two `Array` instances are sharing the same memory buffer.
- `shape_` encapsulates essential information regarding the array's dimensions.
- `info_` acts as a repository for metadata essential for traceability and composing computational graph. It encompasses some computational data, including references to Primitive and a vector of Array instances as input. The tracer_ field enables traceability, it can be `pack`ed or `unpack` ed by `Trace`.
- `size_` denotes the memory allocation size upon creating an Array
- `dtype_` specifies the data type of the elements stored within the Array.
## Operator(Op)
`Op` serves as a pivotal interface for Python users, providing mechanisms for constructing complex operations and computations within the AINL framework.
These operators function as decorators for `Array`, integrating `Primitive` information within `Array` fields.
Operators can be composable, making the evaluation of an `Array` lazily. The evaluation of an `Array` is running recursvly in reversed topo order under the control of `Scheduler` ad `Trace`.

## Primitive
Primitive embodies functional semantics, encapsulating the essence of computations within the framework.
```c++
class Primitive:
    public:
        ... eval()
        ... evalCPU()
        ... typeRelation()
```
- `eval` serves as the entrypoint during Array evaluation, orchestrating the execution of computations within the computational graph. It intelligently dispatches operations to various implementations based on trace, data type, and device information.
- `evalCPU`, `typeRelation`, and similar methods detail the underlying implementation for a Primitive. 

## Trace
Trace provides the mechanism for transformations occurring within the computational graph, offering a view of program execution.
```C++
class Trace
        public:
            ... pack()
            ... unpack()
            ... process()
```
- `pack` and `unpack` facilitate the integration and disintegration of information necessary for Array operations before and after a trace concludes, ensuring composable transformations throughout the computational process.
- `process` handles the execution of computations within the framework. It is redispatched to concrete implementations of `Primitive` during execution.

## Integration with IREE
- [IREE C API Doc]("https://iree.dev/reference/bindings/c-api/")
- [IREE Runtime API Template Repo]("https://github.com/iree-org/iree-template-runtime-cmake/")

# Other Resources
- [Pybind11 doc]("https://pybind11.readthedocs.io/en/stable/")
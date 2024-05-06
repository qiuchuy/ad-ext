# Workflow
When `jit` a python function, the following steps are taken:
- a `JITTrace` instance is created and pushed onto the trace stack, holding a reference a `ALModule`
- input `Array`s are wrapped into corresponding `JITTracer` instances
- the function is called with the `JITTracer` instances
- for every primitives executed during the `jit` function evaluation, construct a new `JITTracer` instance with the primitive
- evaluate the function with the `JITTracer` instances:
    - create an/or more IR expression for single primitive 
    - when all `JITTracer` instances are evaluated, start standard evaluation (if standard evaluation are partially done because of some side-effects, just continue)
- the `JITTrace` instance is popped from the trace stack

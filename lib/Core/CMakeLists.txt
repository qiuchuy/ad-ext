add_ailang_library(AILangCore
  Allocator.cpp
  Array.cpp
  Device.cpp
  Dtype.cpp
  Graph.cpp
  Ops.cpp
  Primitive.cpp
  Trace.cpp
  Transformation.cpp

  DEPENDS AILangIR AILangTransform
)
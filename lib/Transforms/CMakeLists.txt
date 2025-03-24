add_ailang_library(AILangTransform
    Autodiff.cpp
    Pass.cpp
    StablehloConversion.cpp
    utils.cpp
    Visualize.cpp

    DEPENDS AILangIR

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRFuncDialect
    MLIRIR 
    MLIRPass
    StablehloOps 
    StablehloReferenceApi
)
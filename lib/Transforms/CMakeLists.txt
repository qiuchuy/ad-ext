add_ailang_library(AILangTransform
    Autodiff.cpp
    Pass.cpp
    StablehloConversion.cpp
    utils.cpp

    DEPENDS AILangIR

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRFuncDialect
    MLIRIR 
    StablehloOps 
    StablehloReferenceApi
)
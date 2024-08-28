add_ailang_library(AILangTransform
    Pass.cpp
    StablehloConversion.cpp

    DEPENDS AILangIR

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRFuncDialect
    MLIRIR 
    StablehloOps 
    StablehloReferenceApi
)
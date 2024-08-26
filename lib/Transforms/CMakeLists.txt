add_ailang_library(AILangTransform
    Autodiff.cpp
    Pass.cpp
    StablehloConversion.cpp

    DEPENDS AILangIR

    LINK_LIBS PUBLIC
    MLIRIR 
    StablehloOps 
    StablehloReferenceApi
)
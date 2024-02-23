.PHONY: lib, pybind, clean, format, all


SRC_DIR := ./AILang
SRCS := $(shell find $(SRC_DIR) -type f -name '*.cpp')
HDRS := $(shell find $(SRC_DIR) -type f -name '*.h')

all:
	./build_tools/build.sh

format:
	python3 -m black ./AILang
	clang-format -i --style=llvm $(SRCS) $(HDRS)


clean:
	rm -rf build AILang/python/ailang/ffi/*.so



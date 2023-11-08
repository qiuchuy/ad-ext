.PHONY: lib, pybind, clean, format, all


SRC_DIR := .
SRCS := $(shell find $(SRC_DIR) -type f -name '*.cpp')
HDRS := $(shell find $(SRC_DIR) -type f -name '*.h')

all: lib

lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i $(SRCS) $(HDRS)

clean:
	rm -rf build python/ailang/ffi/*.so



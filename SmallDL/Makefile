#SmallDL Makefile

CXX = dpclang++
CXXFLAGS = -O3 -fsycl -I./core -I./nn -I./optim
LIB_FLAGS = -fPIC -shared

LIB_SRCS = ./nn/arena_alloc.cpp ./nn/matrices.cpp ./nn/tensor.cpp ./nn/layer.cpp ./nn/m_operations.cpp
LIB_HDRS = ./core/arena_alloc.hpp ./core/matrices.hpp ./core/tensor.hpp ./core/layer.hpp ./core/m_operations.hpp ./core/constants.hpp 

.PHONY: all lib main clean

all: lib main

libSmallDL.so: $(LIB_SRCS) $(LIB_HDRS)
	$(CXX) $(LIB_SRCS) $(CXXFLAGS) $(LIB_FLAGS) -o $@

lib: libSmallDL.so

test.out: ./examples/test.cpp libSmallDL.so ./optim/optimizer.hpp
	$(CXX) $< $(CXXFLAGS) -L. -lSmallDL -Wl,-rpath,. -o $@

main: test.out

clean:
	rm -f libSmallDL.so test.out

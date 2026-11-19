#SmallDL Makefile

CXX = dpclang++
CXXFLAGS = -O3 -std=c++20 -fsycl -I./core -I./nn -I./optim
LIB_FLAGS = -fPIC -shared

LIB_SRCS = ./nn/arena_alloc.cpp ./nn/matrices.cpp ./nn/graph.cpp ./nn/t_operations.cpp ./nn/layer.cpp ./nn/m_operations.cpp
LIB_HDRS = ./core/arena_alloc.hpp ./core/matrices.hpp ./core/graph.hpp ./core/t_operations.hpp ./core/layer.hpp ./core/m_operations.hpp ./core/constants.hpp ./core/tensor.hpp ./core/reflection.hpp

.PHONY: all lib main clean

all: lib main

libSmallDL.so: $(LIB_SRCS) $(LIB_HDRS)
	$(CXX) $(LIB_SRCS) $(CXXFLAGS) $(LIB_FLAGS) -o $@

lib: libSmallDL.so

test.out: ./examples/test.cpp libSmallDL.so ./optim/optimizer.hpp
	$(CXX) $< $(CXXFLAGS) -L. -lSmallDL -Wl,-rpath,. -o $@

main: test.out

dbg_test.out: ./examples/test.cpp libSmallDL.so ./optim/optimizer.hpp
	$(CXX) $< -rdynamic $(CXXFLAGS) -L. -lSmallDL -Wl,-rpath,. -o $@
	gdb -batch -ex "run" -ex "bt" ./dbg_test.out	

debug: dbg_test.out

clean:
	rm -f libSmallDL.so test.out dbg_test.out

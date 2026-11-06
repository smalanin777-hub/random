all: lib main
lib: matrices.hpp matrices.cpp tensor.hpp tensor.cpp m_operations.cpp m_operations.hpp
	dpclang++ matrices.cpp tensor.cpp m_operations.cpp -fPIC -shared -O3 -fsycl -o libSmallDL.so
main: libSmallDL.so optimizer.hpp test.cpp 
	dpclang++ test.cpp -O3 -fsycl -L. -lSmallDL -Wl,-rpath,. -o test.out
clean:
	rm libSmallDL.so test.out
	

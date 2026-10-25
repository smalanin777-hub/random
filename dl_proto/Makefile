all: lib main
lib: matrices.hpp matrices.cpp tensor.hpp tensor.cpp
	dpclang++ matrices.cpp tensor.cpp -fPIC -shared -O3 -fsycl -o libmatrices.so
main: matrices.hpp matrices.cpp optimizer.hpp tensor.hpp tensor.cpp test.cpp 
	dpclang++ test.cpp -O3 -fsycl -L. -lmatrices -Wl,-rpath,. -o test.out
clean:
	rm libmatrices.so test.out
	

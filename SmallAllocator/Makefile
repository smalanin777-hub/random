all: lib main
lib: SmallAllocator.cpp SmallAllocator.hpp
	g++ SmallAllocator.cpp -shared -fPIC -o libSmallAllocator.so 
main: lib tests.cpp
	g++ tests.cpp -L. -lSmallAllocator -o main 
clean: 
	rm main libSmallAllocator.so
copy_files:
	cat Makefile SmallAllocator.hpp SmallAllocator.cpp tests.cpp | xclip -selection c

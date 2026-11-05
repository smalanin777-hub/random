all: lib main
lib: SmallAllocator.cpp SmallAllocator.hpp
	g++ SmallAllocator.cpp -shared -fPIC -o libSmallAllocator.so 
main: lib tests.cpp
	g++ tests.cpp -L. -lSmallAllocator -o test.out 
clean: 
	rm main libSmallAllocator.so
copy_files:
	cat Makefile SmallAllocator.hpp SmallAllocator.cpp tests.cpp | xclip -selection c
test: test.out 
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./main

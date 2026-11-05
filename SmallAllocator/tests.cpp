#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cassert>
#include <set>
#include <algorithm>
#include "SmallAllocator.hpp"

void test_basic_allocation() {
    std::cout << "=== Тест 1: Базовое выделение и освобождение ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    void* ptr1 = allocator.Alloc(100);
    void* ptr2 = allocator.Alloc(200);
    void* ptr3 = allocator.Alloc(300);
    
    assert(ptr1 != nullptr);
    assert(ptr2 != nullptr);
    assert(ptr3 != nullptr);
    assert(ptr1 != ptr2);
    assert(ptr2 != ptr3);
    
    allocator.Free(ptr2);
    allocator.Free(ptr1);
    allocator.Free(ptr3);
    
    std::cout << "✓ Пройден" << std::endl;
}

void test_realloc_grow_shrink() {
    std::cout << "=== Тест 2: ReAlloc с увеличением и уменьшением ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    void* ptr = allocator.Alloc(100);
    assert(ptr != nullptr);
    
    std::memset(ptr, 0xAB, 100);
    
    ptr = allocator.ReAlloc(ptr, 50);
    assert(ptr != nullptr);
    
    uint8_t* bytes = reinterpret_cast<uint8_t*>(ptr);
    for (int i = 0; i < 50; ++i) {
        assert(bytes[i] == 0xAB);
    }
    
    ptr = allocator.ReAlloc(ptr, 500);
    assert(ptr != nullptr);
    
    bytes = reinterpret_cast<uint8_t*>(ptr);
    for (int i = 0; i < 50; ++i) {
        assert(bytes[i] == 0xAB);
    }
    
    allocator.Free(ptr);
    std::cout << "✓ Пройден" << std::endl;
}

void test_coalescing() {
    std::cout << "=== Тест 3: Слияние свободных блоков ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    void* ptr1 = allocator.Alloc(100);
    void* ptr2 = allocator.Alloc(100);
    void* ptr3 = allocator.Alloc(100);
    
    allocator.Free(ptr1);
    allocator.Free(ptr3);
    allocator.Free(ptr2);
    
    void* big_ptr = allocator.Alloc(300);
    assert(big_ptr != nullptr);
    
    allocator.Free(big_ptr);
    std::cout << "✓ Пройден" << std::endl;
}

void test_fragmentation() {
    std::cout << "=== Тест 4: Фрагментация и дефрагментация ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i) {
        void* ptr = allocator.Alloc(100);
        if (ptr) ptrs.push_back(ptr);
    }
    
    for (int i = 0; i < 100; i += 2) {
        allocator.Free(ptrs[i]);
    }
    
    for (int i = 1; i < 100; i += 2) {
        allocator.Free(ptrs[i]);
    }
    
    void* final_ptr = allocator.Alloc(50000);
    assert(final_ptr != nullptr);
    
    allocator.Free(final_ptr);
    std::cout << "✓ Пройден" << std::endl;
}

void test_edge_cases() {
    std::cout << "=== Тест 5: Граничные случаи ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    void* ptr1 = allocator.Alloc(1);
    assert(ptr1 != nullptr);
    allocator.Free(ptr1);
    
    void* ptr2 = allocator.Alloc(64);
    assert(ptr2 != nullptr);
    allocator.Free(ptr2);
    
    void* ptr3 = allocator.ReAlloc(nullptr, 100);
    assert(ptr3 != nullptr);
    allocator.Free(ptr3);
    
    allocator.Free(nullptr);
    
    std::cout << "✓ Пройден" << std::endl;
}

void test_memory_reuse() {
    std::cout << "=== Тест 6: Переиспользование памяти ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    void* ptr1 = allocator.Alloc(100);
    allocator.Free(ptr1);
    
    void* ptr2 = allocator.Alloc(100);
    assert(ptr2 == ptr1);
    
    allocator.Free(ptr2);
    std::cout << "✓ Пройден" << std::endl;
}

void test_alignment() {
    std::cout << "=== Тест 7: Выравнивание ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    for (int size = 1; size <= 1000; size += 17) {
        void* ptr = allocator.Alloc(size);
        assert(ptr != nullptr);
        assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
        allocator.Free(ptr);
    }
    
    std::cout << "✓ Пройден" << std::endl;
}

void test_full_pool() {
    std::cout << "=== Тест 8: Полное заполнение пула ===" << std::endl;
    SmallAllocator allocator(64 * 1024);
    
    std::vector<void*> ptrs;
    int count = 0;
    while (true) {
        void* ptr = allocator.Alloc(100);
        if (!ptr) break;
        ptrs.push_back(ptr);
        count++;
    }
    
    assert(count > 0);
    
    void* extra = allocator.Alloc(100);
    assert(extra == nullptr);
    
    for (void* ptr : ptrs) {
        allocator.Free(ptr);
    }
    
    std::cout << "✓ Пройден (выделено " << count << " блоков)" << std::endl;
}

void test_stress_random() {
    std::cout << "=== Тест 9: Агрессивный случайный стресс-тест ===" << std::endl;
    SmallAllocator allocator(2 * 1024 * 1024);
    
    const int NUM_PTRS = 500;
    void* ptrs[NUM_PTRS] = {nullptr};
    unsigned int sizes[NUM_PTRS] = {0};
    uint8_t magic_bytes[NUM_PTRS] = {0};
    
    std::mt19937 gen(12345);
    std::uniform_int_distribution<unsigned int> dist_size(1, 8000);
    std::uniform_int_distribution<int> dist_op(0, 100);
    std::uniform_int_distribution<uint8_t> dist_byte(1, 254);
    
    int success_ops = 0;
    
    for (int i = 0; i < 50000; ++i) {
        int idx = gen() % NUM_PTRS;
        int op = dist_op(gen);
        
        if (op < 40) {
            if (ptrs[idx] == nullptr) {
                sizes[idx] = dist_size(gen);
                magic_bytes[idx] = dist_byte(gen);
                ptrs[idx] = allocator.Alloc(sizes[idx]);
                if (ptrs[idx]) {
                    std::memset(ptrs[idx], magic_bytes[idx], sizes[idx]);
                    success_ops++;
                }
            }
        } else if (op < 75) {
            if (ptrs[idx] != nullptr) {
                uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(ptrs[idx]);
                for (unsigned int j = 0; j < sizes[idx]; ++j) {
                    if (byte_ptr[j] != magic_bytes[idx]) {
                        std::cerr << "Данные затерты по индексу " << idx << std::endl;
                        assert(false);
                    }
                }
                allocator.Free(ptrs[idx]);
                ptrs[idx] = nullptr;
                sizes[idx] = 0;
                magic_bytes[idx] = 0;
                success_ops++;
            }
        } else {
            if (ptrs[idx] != nullptr) {
                uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(ptrs[idx]);
                for (unsigned int j = 0; j < sizes[idx]; ++j) {
                    assert(byte_ptr[j] == magic_bytes[idx]);
                }
                unsigned int new_size = dist_size(gen);
                uint8_t new_magic = dist_byte(gen);
                void* new_ptr = allocator.ReAlloc(ptrs[idx], new_size);
                if (new_ptr) {
                    ptrs[idx] = new_ptr;
                    unsigned int check_len = std::min(sizes[idx], new_size);
                    uint8_t* new_byte_ptr = reinterpret_cast<uint8_t*>(new_ptr);
                    for (unsigned int j = 0; j < check_len; ++j) {
                        assert(new_byte_ptr[j] == magic_bytes[idx]);
                    }
                    sizes[idx] = new_size;
                    magic_bytes[idx] = new_magic;
                    std::memset(ptrs[idx], magic_bytes[idx], sizes[idx]);
                    success_ops++;
                }
            }
        }
    }
    
    for (int i = 0; i < NUM_PTRS; ++i) {
        if (ptrs[i] != nullptr) {
            allocator.Free(ptrs[i]);
        }
    }
    
    std::cout << "✓ Пройден (" << success_ops << " операций)" << std::endl;
}

void test_realloc_chain() {
    std::cout << "=== Тест 10: Цепочка ReAlloc ===" << std::endl;
    SmallAllocator allocator(1024 * 1024);
    
    void* ptr = allocator.Alloc(50);
    assert(ptr != nullptr);
    
    for (int i = 0; i < 50; ++i) {
        std::memset(ptr, i, 50);
        ptr = allocator.ReAlloc(ptr, 50 + i * 10);
        assert(ptr != nullptr);
        
        uint8_t* bytes = reinterpret_cast<uint8_t*>(ptr);
        for (int j = 0; j < 50; ++j) {
            assert(bytes[j] == i);
        }
    }
    
    allocator.Free(ptr);
    std::cout << "✓ Пройден" << std::endl;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Комплексный стресс-тест SmallAllocator             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    test_basic_allocation();
    test_realloc_grow_shrink();
    test_coalescing();
    test_fragmentation();
    test_edge_cases();
    test_memory_reuse();
    test_alignment();
    test_full_pool();
    test_stress_random();
    test_realloc_chain();
    
    std::cout << std::endl;
    std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        ✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! ✓                 ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    return 0;
}

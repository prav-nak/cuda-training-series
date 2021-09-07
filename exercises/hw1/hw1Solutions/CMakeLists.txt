cmake_minimum_required(VERSION 3.10)
project (hw1 LANGUAGES CUDA C)

# Example hello
add_executable(hello hello.cu)

# Vector addition
add_executable(vector_add vector_add.cu)

# Matrix multiplication addition
add_executable(mat_mul mat_mul.cu)
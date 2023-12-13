cmake_minimum_required (VERSION 3.11)

project (helm-eqMPI)
set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

add_executable(helm-eqMPI main.cpp)

target_link_libraries(helm-eqMPI MPI::MPI_CXX)
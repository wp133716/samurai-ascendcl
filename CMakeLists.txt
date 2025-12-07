cmake_minimum_required(VERSION 3.10)
project(sam2_tracker)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -O2") #-ggdb
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include
                    /usr/local/Ascend/ascend-toolkit/latest/acllib/include)

link_directories(/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64)

add_executable(${PROJECT_NAME}_acl main_acl.cpp
                        src/sam2_tracker_acl.cpp
                        src/kalman_filter.cpp)
target_link_libraries(${PROJECT_NAME}
                    ${OpenCV_LIBS}
                    ascendcl
                    )
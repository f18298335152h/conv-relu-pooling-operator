include_directories(${CMAKE_CURRENT_BINARY_DIR}/../include)


add_executable(test test.cc)
target_link_libraries(test intel)



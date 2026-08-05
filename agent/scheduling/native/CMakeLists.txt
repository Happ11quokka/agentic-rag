cmake_minimum_required(VERSION 3.20)
project(parallel_scheduled_engine LANGUAGES CXX)

find_package(llama CONFIG REQUIRED)

add_executable(parallel-scheduled-engine engine.cpp)
target_link_libraries(parallel-scheduled-engine PRIVATE llama)
target_compile_features(parallel-scheduled-engine PRIVATE cxx_std_20)
target_compile_options(parallel-scheduled-engine PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(
  parallel-scheduled-engine PRIVATE
  PS_LLAMA_BUILD=${PS_LLAMA_BUILD}
  PS_LLAMA_COMMIT="${PS_LLAMA_COMMIT}"
  PS_SOURCE_SHA256="${PS_SOURCE_SHA256}"
)

get_target_property(LLAMA_LIBRARY_PATH llama IMPORTED_LOCATION)
get_filename_component(LLAMA_LIBRARY_DIR "${LLAMA_LIBRARY_PATH}" DIRECTORY)
set_target_properties(
  parallel-scheduled-engine PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "${LLAMA_LIBRARY_DIR}"
)

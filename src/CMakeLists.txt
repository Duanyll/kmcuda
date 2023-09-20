cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(KMCUDA CXX C CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 14)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
if (MSVC)
  set (OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -openmp:experimental")
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(CUDAToolkit REQUIRED)
if (NOT CUDA_ARCH)
  set(CUDA_ARCH 86)
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

if (NOT DISABLE_PYTORCH)
  if (MSVC)
    # suppress C4624
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4624")
  endif()
  find_package(PythonInterp 3 REQUIRED)
  find_package(PythonLibs 3 REQUIRED)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
  message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
  message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
  find_package(Torch REQUIRED)
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
elseif (NOT DISABLE_PYTHON)
  find_package(PythonInterp 3 REQUIRED)
  find_package(PythonLibs 3 REQUIRED)
  if (NOT NUMPY_INCLUDES)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDES)
  endif()
endif()

if (NOT DISABLE_R)
  find_package(R)
endif()

set(SOURCE_FILES kmcuda.cc kmcuda.h wrappers.h private.h fp_abstraction.h tricks.cuh
                 metric_abstraction.h kmeans.cu knn.cu transpose.cu)
if (PYTHONLIBS_FOUND)
  if (NOT DISABLE_PYTORCH)
    list(APPEND SOURCE_FILES torch.cc)
  else()
    list(APPEND SOURCE_FILES python.cc)
  endif()
endif()
if (R_FOUND)
  list(APPEND SOURCE_FILES r.cc)
endif()

add_library(KMCUDA SHARED ${SOURCE_FILES})

add_compile_definitions(_MWAITXINTRIN_H_INCLUDED _FORCE_INLINES)
add_compile_definitions(CUDA_ARCH=${CUDA_ARCH})

target_link_libraries(KMCUDA CUDA::curand)
target_link_libraries(KMCUDA OpenMP::OpenMP_CXX)

if (PYTHONLIBS_FOUND)
  if (NOT DISABLE_PYTORCH)
    include_directories(${PYTHON_INCLUDE_DIRS})
    target_link_libraries(KMCUDA 
      ${PYTHON_LIBRARIES}
      ${TORCH_LIBRARIES}
      ${TORCH_PYTHON_LIBRARY}
    )
    if (WIN32)
      target_compile_definitions(KMCUDA PRIVATE TORCH_EXTENSION_NAME=KMCUDA)
    else()
      target_compile_definitions(KMCUDA PRIVATE TORCH_EXTENSION_NAME=libKMCUDA)
    endif()
  else()
    include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES})
    target_link_libraries(KMCUDA ${PYTHON_LIBRARIES})
  endif()
endif()

if (R_FOUND)
  include_directories(${R_INCLUDE_DIRS})
  target_link_libraries(KMCUDA ${R_LIBRARIES})
endif()

if (SUFFIX)
  set_target_properties(KMCUDA PROPERTIES SUFFIX ${SUFFIX})
endif()

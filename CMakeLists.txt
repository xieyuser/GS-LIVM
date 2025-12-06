cmake_minimum_required(VERSION 3.2)

option(BUILD_GS "option for build 3DGS" ON)

# need add this option
# issue #30
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-static-libstdc++")

if(BUILD_GS)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;90")
  endif()

  project(gslivm LANGUAGES CUDA CXX)
  add_definitions(-DBUILD_GS)
  message(STATUS "will build gaussian")
else()
  project(gslivm LANGUAGES CXX)
endif()

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_definitions(-O3)

if(BUILD_GS)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

option(USE_LIVOX "option for livox" ON)
if(USE_LIVOX)
  set(PACKAGE_XML_PATH ${CMAKE_CURRENT_SOURCE_DIR}/package_livox.xml)
  add_definitions(-DUSE_LIVOX)
  find_package(catkin REQUIRED COMPONENTS
    eigen_conversions
    geometry_msgs
    nav_msgs
    pcl_ros
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    livox_ros_driver2
    cv_bridge
  )
  else()
  set(PACKAGE_XML_PATH ${CMAKE_CURRENT_SOURCE_DIR}/package.xml)
  find_package(catkin REQUIRED COMPONENTS
    eigen_conversions
    geometry_msgs
    nav_msgs
    pcl_ros
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    cv_bridge
  )
endif(USE_LIVOX)

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PythonLibs REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Sophus REQUIRED)
find_package(tsl-robin-map REQUIRED)

if(BUILD_GS)
  find_package(CUDAToolkit REQUIRED)
endif()

if(${OpenMP_CXX_FOUND})
    message(WARNING "OpenMP found and will be used.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3
)

if(BUILD_GS)
  include_directories(include/gs)
endif(BUILD_GS)

include_directories(  
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${SOPHUS_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${tsl-robin-map_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/external/tinyply/source
  ${PYTHON_INCLUDE_DIRS}
)

include_directories(
  ${CUDAToolkit_INCLUDE_DIRS}
)

# common
add_library(${PROJECT_NAME}.common 
  src/common/timer/timer.cc
)

if(BUILD_GS)
  include("${PROJECT_SOURCE_DIR}/cmake/torch.cmake")

  list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}
    $ENV{CMAKE_PREFIX_PATH})
  find_package(Torch REQUIRED)
  add_subdirectory(external)

  set(gs_libs
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    tinyply
    glm
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusolver
    Threads::Threads
    args
    cnpy
  )

  include_directories(${TORCH_INCLUDE_DIRS})

  # gs
  set(SOURCES
    src/gs/gaussian.cu
    src/gs/camera.cu
    src/gs/loss_monitor.cu
    src/gs/parameters.cu
    src/gs/rasterizer.cu
    src/gs/rasterize_points.cu
    src/cuda_rasterizer/backward.cu
    src/cuda_rasterizer/forward.cu
    src/cuda_rasterizer/rasterizer_impl.cu)

  add_library(${PROJECT_NAME}.gs ${SOURCES})
  target_link_libraries(${PROJECT_NAME}.gs ${gs_libs} ${PCL_LIBRARIES} ${sophus_LIBRARIES})
  target_compile_features(${PROJECT_NAME}.gs PRIVATE cxx_std_17)

  # gp3d
  set(
    gp_files
    src/gp3d/cell.cpp
    src/gp3d/map.cpp
    src/gp3d/gpprocess.cu
  )

  add_library(${PROJECT_NAME}.gp3d ${gp_files})
  target_link_libraries(
    ${PROJECT_NAME}.gp3d 
    ${gs_libs}
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES}
    ${PROJECT_NAME}.common
    OpenMP::OpenMP_CXX
  )
  target_compile_features(${PROJECT_NAME}.gp3d PRIVATE cxx_std_17)
endif(BUILD_GS)

add_executable(livo_node
		src/liw/lioOptimization.cpp
		src/liw/imageProcessing.cpp
		src/liw/opticalFlowTracker.cpp
		src/liw/rgbMapTracker.cpp
		src/liw/lkpyramid.cpp
		src/liw/optimize.cpp
		src/liw/cloudMap.cpp
		src/liw/cloudProcessing.cpp 
		src/liw/state.cpp
		src/liw/eskfEstimator.cpp
		src/liw/utility.cpp
		src/liw/parameters.cpp
    src/liw/cudaMatrixMultiply.cpp)

target_link_libraries(livo_node ${catkin_LIBRARIES} ${PROJECT_NAME}.common ${PROJECT_NAME}.gs ${PROJECT_NAME}.gp3d ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} tsl::robin_map)

# gmm
# find_package(Open3D REQUIRED)  
# include_directories(${Open3D_INCLUDE_DIR} src/test/gmm)

# add_library(${PROJECT_NAME}.gmm src/test/gmm/GMMFit.cpp)
# target_link_libraries(${PROJECT_NAME}.gmm ${gs_libs} Open3D::Open3D)

# add_executable(
#   ${PROJECT_NAME}_test_gmm
#   src/test/test_gmm.cpp
# )
# target_link_libraries(${PROJECT_NAME}_test_gmm  ${PROJECT_NAME}.gmm )

# eclipse
# find_package(Open3D REQUIRED)  
# include_directories(${Open3D_INCLUDE_DIR})

# add_executable(
#   ${PROJECT_NAME}_test_ecli
#   src/test/test_ecli.cpp
# )
# target_link_libraries(${PROJECT_NAME}_test_ecli ${gs_libs} Open3D::Open3D )

# hash
# add_executable(
#   ${PROJECT_NAME}_test_hash
#   src/test/test_hash.cpp
# )

# target_link_libraries(${PROJECT_NAME}_test_hash  pthread)

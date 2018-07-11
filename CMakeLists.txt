cmake_minimum_required(VERSION 2.8)
project(gadit LANGUAGES CXX CUDA)


FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_64_BIT_DEVICE_CODE ON)
#INCLUDE(FindCUDA)

include_directories("${CUDA_INCLUDE_DIRS}")

#add_library(memory_unit SHARED include/memory_unit.h )
#add_library(memory_manager SHARED include/memory_manager.h )

cuda_add_executable (gadit 
	include/memory_unit.h
	include/memory_manager.h 
	src/main.cpp
#	include/memory_unit.h
#	src/memory_unit.cpp
	)				   

#target_link_libraries( gadit memory_unit )		
#target_link_libraries( gadit memory_manager )		

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(gadit PUBLIC cxx_std_11)	
			 
install(TARGETS gadit DESTINATION bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
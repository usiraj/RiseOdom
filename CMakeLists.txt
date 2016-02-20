project(RiseOdom)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Release)
####################### FUNCTIONS #####################
function(optionprint msg var)
	if ( ${var} )
		message(STATUS ${msg} " : ON " )
	else ( ${var} )
		message(STATUS ${msg} " : OFF " )
	endif ( ${var} )
endfunction(optionprint)
#################### CMAKE OPTIONS #####################
option(USE_PENTIUM4 "For Use with Pentium 4 PC's" OFF)
option(BUILD_BINARIES "Build binaries" OFF)

optionprint("Using Pentium 4 instructions" USE_PENTIUM4)
optionprint("Building Binaries" BUILD_BINARIES)
message(STATUS "CMAKE_INSTALL_PREFIX : ${CMAKE_INSTALL_PREFIX}" )
################# GENERATE OPTIONS HEADER ##############
configure_file("${PROJECT_SOURCE_DIR}/riseodomsettings.h.in" "${PROJECT_SOURCE_DIR}/include/riseodomsettings.h" @ONLY)
################## INCLUDE OPENCV ######################
SET(CMAKE_MODULE_PATH /usr/local/lib/cmake/ )
find_package(OpenCV REQUIRED)
################## NATIVE INSTRUCTIONS SET #############
add_compile_options(-std=gnu++11) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Ofast")
############### include subdirectories #################
include_directories(${PROJECT_SOURCE_DIR}/include)
############### BUILD LIBRARY RISEODOM #################
add_library(riseodom SHARED
				src/image_database.cpp src/utils.cpp
				src/transformation_matrix.cpp 
				src/stereopreprocess.cpp
				src/utils_optim.cpp  src/spkf.cpp src/ispkf_motionmodels.cpp
				src/stereoodom_base.cpp src/stereovodom.cpp)
target_link_libraries(riseodom ${OpenCV_LIBS})
add_library(riseodom_s
            	src/image_database.cpp src/utils.cpp
				src/transformation_matrix.cpp 
				src/stereopreprocess.cpp
				src/utils_optim.cpp  src/spkf.cpp src/ispkf_motionmodels.cpp
				src/stereoodom_base.cpp src/stereovodom.cpp)
target_link_libraries(riseodom ${OpenCV_LIBS})
################## BUILD BINARIES ######################
if ( BUILD_BINARIES )
	# Build Binaries that must not use G2O SLAM
	add_executable(processodomdb test/processodomdb.cpp)
	target_link_libraries(processodomdb riseodom)
	# Build Binaries that are not dependent on G2O SLAM	
endif ( BUILD_BINARIES )
################### INSTALL TARGETS #####################
set(libdir                  "${CMAKE_INSTALL_PREFIX}/lib")
set(prefix                  "${CMAKE_INSTALL_PREFIX}")
set(exec_prefix             "\${prefix}")
set(includedir              "\${prefix}/include/riseodom")
################ GENERATE PKG-CONFIG ####################
configure_file("${PROJECT_SOURCE_DIR}/riseodom.pc.in" "${CMAKE_BINARY_DIR}/riseodom.pc" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/riseodom.pc" DESTINATION lib/pkgconfig)
################## INSTALL HEADERS ######################
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" 
	DESTINATION include/riseodom
	FILES_MATCHING PATTERN "*.h")
################## INSTALL LIBRARIES ####################
install(TARGETS riseodom DESTINATION lib)
install(TARGETS riseodom_s DESTINATION lib)
################# INSTALL EXECUTABLES ###################
if ( BUILD_BINARIES )
	install(TARGETS processodomdb DESTINATION bin)
endif( BUILD_BINARIES )

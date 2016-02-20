# find sgslam
FIND_PATH(RISEODOM_INCLUDE_DIR NAMES riseodom/stereoodom_base.h
  PATHS
  ${RISEODOM_ROOT}/include
  ${RISEODOM_ROOT}
  /usr/include
  /opt/local/include
  /usr/local/include
  /sw/include
  NO_DEFAULT_PATH
  )
  
FIND_LIBRARY(RISEODOM_LIBRARY NAMES riseodom
  PATHS
  ${RISEODOM_ROOT}/lib
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  )
  
find_package(OpenCV REQUIRED)
set(RISEODOM_LIBRARY ${OpenCV_LIBS} ${RISEODOM_LIBRARY} )
add_compile_options(-std=gnu++11)

# Earlier sd targets request C++11, a subset of the configured C++17 dialect.
include("${CMAKE_CURRENT_LIST_DIR}/nec-ve.cmake")
list(APPEND CMAKE_CXX_COMPILE_FEATURES cxx_std_11 cxx_std_14)

INCLUDE_DIRECTORIES(
    ../
)

ADD_LIBRARY(rectify rectify.cc)
ADD_LIBRARY(reproject reproject.cc)
TARGET_LINK_LIBRARIES(rectify
    reproject
)


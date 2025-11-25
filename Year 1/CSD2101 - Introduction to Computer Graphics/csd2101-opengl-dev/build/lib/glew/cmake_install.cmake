# Install script for directory: D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/lib/glew

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/opengl-dev")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/Debug/glewd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/Release/glew.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/MinSizeRel/glew.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/RelWithDebInfo/glew.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/Debug/glew-sharedd.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/Release/glew-shared.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/MinSizeRel/glew-shared.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY OPTIONAL FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/lib/RelWithDebInfo/glew-shared.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glew/glewConfig.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glew/glewConfig.cmake"
         "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/CMakeFiles/Export/7a894a12241bfddc41ca6be6d0e647bd/glewConfig.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glew/glewConfig-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/glew/glewConfig.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glew" TYPE FILE FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/CMakeFiles/Export/7a894a12241bfddc41ca6be6d0e647bd/glewConfig.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glew" TYPE FILE FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/CMakeFiles/Export/7a894a12241bfddc41ca6be6d0e647bd/glewConfig-debug.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glew" TYPE FILE FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/CMakeFiles/Export/7a894a12241bfddc41ca6be6d0e647bd/glewConfig-minsizerel.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glew" TYPE FILE FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/CMakeFiles/Export/7a894a12241bfddc41ca6be6d0e647bd/glewConfig-relwithdebinfo.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/glew" TYPE FILE FILES "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/build/lib/glew/CMakeFiles/Export/7a894a12241bfddc41ca6be6d0e647bd/glewConfig-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GL" TYPE FILE FILES
    "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/lib/glew/include/GL/eglew.h"
    "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/lib/glew/include/GL/glew.h"
    "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/lib/glew/include/GL/glxew.h"
    "D:/SIT-Digipen/Year 1/Trimester 3/CSD2101 - Introduction to Computer Graphics/csd2101-opengl-dev/lib/glew/include/GL/wglew.h"
    )
endif()


# Projects Settings

cmake_minimum_required(VERSION 2.8...3.13)

project (opengl-dev)

add_definitions(-DUSE_CSD2101_AUTOMATION=0)
add_definitions(-DGLEW_STATIC)
add_subdirectory(lib/glfw EXCLUDE_FROM_ALL) 
add_subdirectory(lib/glew EXCLUDE_FROM_ALL) 

set(EXTERNAL_LIBRARY_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/lib/lodepng/lodepng.h
    ${CMAKE_CURRENT_LIST_DIR}/lib/lodepng/lodepng.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lib/glfw-event-recorder/recorder.h
    ${CMAKE_CURRENT_LIST_DIR}/lib/glfw-event-recorder/recorder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lib/glfw-event-recorder/hook_csd2101.h
)

# All libraries - GLFW, glew
set(ALL_LIBS
  PRIVATE glfw
  PRIVATE libglew_static
)
ADD_LIBRARY(csd2101 STATIC
    ${EXTERNAL_LIBRARY_SOURCES}
)
target_link_libraries(csd2101
    ${ALL_LIBS}
) 
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/imgui)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/imgui/backends)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/gleq)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/lodepng)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/glfw-event-recorder)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/glm)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/tinyobjloader)
include_directories(${CMAKE_CURRENT_LIST_DIR}/lib/dpml/include)


file(GLOB_RECURSE assignment-1_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/assignment-1/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/assignment-1/include/*.h
)

add_executable(assignment-1 
    ${assignment-1_source_files} 
)

target_link_directories(assignment-1
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(assignment-1
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET assignment-1 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(assignment-1 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(assignment-1 PRIVATE /W3 /WX-)
endif()
target_include_directories(assignment-1 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/assignment-1/include)

file(GLOB_RECURSE assignment-2_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/assignment-2/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/assignment-2/include/*.h
)

add_executable(assignment-2 
    ${assignment-2_source_files} 
)

target_link_directories(assignment-2
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(assignment-2
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET assignment-2 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(assignment-2 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(assignment-2 PRIVATE /W3 /WX-)
endif()
target_include_directories(assignment-2 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/assignment-2/include)

file(GLOB_RECURSE tutorial-0_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-0/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-0/include/*.h
)

add_executable(tutorial-0 
    ${tutorial-0_source_files} 
)

target_link_directories(tutorial-0
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(tutorial-0
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET tutorial-0 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tutorial-0 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(tutorial-0 PRIVATE /W3 /WX-)
endif()
target_include_directories(tutorial-0 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-0/include)

file(GLOB_RECURSE tutorial-1_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-1/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-1/include/*.h
)

add_executable(tutorial-1 
    ${tutorial-1_source_files} 
)

target_link_directories(tutorial-1
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(tutorial-1
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET tutorial-1 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tutorial-1 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(tutorial-1 PRIVATE /W3 /WX-)
endif()
target_include_directories(tutorial-1 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-1/include)

file(GLOB_RECURSE tutorial-2_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-2/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-2/include/*.h
)

add_executable(tutorial-2 
    ${tutorial-2_source_files} 
)

target_link_directories(tutorial-2
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(tutorial-2
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET tutorial-2 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tutorial-2 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(tutorial-2 PRIVATE /W3 /WX-)
endif()
target_include_directories(tutorial-2 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-2/include)

file(GLOB_RECURSE tutorial-3_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-3/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-3/include/*.h
)

add_executable(tutorial-3 
    ${tutorial-3_source_files} 
)

target_link_directories(tutorial-3
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(tutorial-3
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET tutorial-3 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tutorial-3 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(tutorial-3 PRIVATE /W3 /WX-)
endif()
target_include_directories(tutorial-3 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-3/include)

file(GLOB_RECURSE tutorial-4_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-4/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-4/include/*.h
)

add_executable(tutorial-4 
    ${tutorial-4_source_files} 
)

target_link_directories(tutorial-4
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(tutorial-4
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET tutorial-4 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tutorial-4 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(tutorial-4 PRIVATE /W3 /WX-)
endif()
target_include_directories(tutorial-4 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-4/include)

file(GLOB_RECURSE tutorial-5_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-5/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-5/include/*.h
)

add_executable(tutorial-5 
    ${tutorial-5_source_files} 
)

target_link_directories(tutorial-5
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(tutorial-5
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET tutorial-5 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tutorial-5 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(tutorial-5 PRIVATE /W3 /WX-)
endif()
target_include_directories(tutorial-5 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-5/include)

file(GLOB_RECURSE tutorial-6_source_files 
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-6/src/*.[ch]pp
    ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-6/include/*.h
)

add_executable(tutorial-6 
    ${tutorial-6_source_files} 
)

target_link_directories(tutorial-6
    PRIVATE
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Debug>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/lib/dpml/lib/Release>
)
target_link_libraries(tutorial-6
    ${ALL_LIBS}
    csd2101
    dpml
) 

set_property(TARGET tutorial-6 PROPERTY CXX_STANDARD 20) 
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tutorial-6 PRIVATE -Wall ${DisableWarnings})
elseif (MSVC)
    target_compile_options(tutorial-6 PRIVATE /W3 /WX-)
endif()
target_include_directories(tutorial-6 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/projects/tutorial-6/include)


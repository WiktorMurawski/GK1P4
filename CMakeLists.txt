cmake_minimum_required(VERSION 3.20)
project(GK1P4 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ────────────────────────────────────────────────
# GLFW (still fetched from git)
# ────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ────────────────────────────────────────────────
# GLM (still fetched from git)
# ────────────────────────────────────────────────
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
)
FetchContent_MakeAvailable(glm)

# ────────────────────────────────────────────────
# GLAD – local copy
# ────────────────────────────────────────────────
set(GLAD_DIR "${CMAKE_SOURCE_DIR}/external/glad")

add_library(glad STATIC
    ${GLAD_DIR}/src/glad.c
)

target_include_directories(glad
    PUBLIC
        ${GLAD_DIR}/include
)

# Optional: reduce noise from glad.c
if(MSVC)
    target_compile_options(glad PRIVATE /wd4201)
else()
    target_compile_options(glad PRIVATE -Wno-unused-parameter)
endif()

# ────────────────────────────────────────────────
# Main executable
# ────────────────────────────────────────────────
add_executable(gk1p4
    src/main.cpp
)

target_link_libraries(gk1p4 PRIVATE
    glfw
    glad
    glm::glm
)

# Linux / Unix system libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(gk1p4 PRIVATE X11 pthread dl)
endif()

# Usually not needed anymore (GLAD sets it via PUBLIC), but kept for safety
target_include_directories(gk1p4 PRIVATE
    ${GLAD_DIR}/include
)

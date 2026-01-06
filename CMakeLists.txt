cmake_minimum_required(VERSION 3.14)
project(Czworoscian LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ────────────────────────────────────────────────
# GLFW
# ────────────────────────────────────────────────
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ────────────────────────────────────────────────
# GLM
# ────────────────────────────────────────────────
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# ────────────────────────────────────────────────
# GLAD v2 – poprawna integracja z SOURCE_SUBDIR cmake
# ────────────────────────────────────────────────
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v2.0.8          # stabilna wersja z 2024, możesz dać v2.0.9 / main jeśli chcesz nowszą
    SOURCE_SUBDIR  cmake           # ← to jest kluczowe – mówi CMake, że projekt zaczyna się w podkatalogu cmake/
)
FetchContent_MakeAvailable(glad)

# Generujemy bibliotekę glad (STATIC, REPRODUCIBLE = powtarzalne generowanie)
glad_add_library(glad_gl33 STATIC REPRODUCIBLE
    API gl:core=3.3
    # opcjonalnie: LOADER (jeśli chcesz ręczne wczytywanie funkcji zamiast auto)
    # EXTENSIONS ... (dodaj rozszerzenia jeśli potrzebujesz)
)

# ────────────────────────────────────────────────
# Główny program
# ────────────────────────────────────────────────
add_executable(czworoscian main.cpp)

target_link_libraries(czworoscian PRIVATE
    glfw
    glad_gl33
    glm::glm
)

# Linux / Unix – systemowe zależności
if(UNIX AND NOT APPLE)
    target_link_libraries(czworoscian PRIVATE X11 pthread dl)
endif()

# GLAD sam ustawia include'y, ale na wszelki wypadek:
target_include_directories(czworoscian PRIVATE ${glad_BINARY_DIR}/include)

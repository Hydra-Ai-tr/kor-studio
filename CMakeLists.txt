cmake_minimum_required(VERSION 3.20)
project(KorStudio VERSION 0.3.0 LANGUAGES C CXX)

option(KOR_BUILD_GUI "Build the Kor Studio desktop application" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
  add_compile_options(/utf-8 /W3)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------- engine
add_library(korcore STATIC
  src/core/parallel.cpp
  src/core/kernels.cpp
  src/core/kernels_base.cpp
  src/core/layers.cpp
  src/core/model.cpp
  src/core/data.cpp
  src/core/trainer.cpp
  src/core/sysinfo.cpp)
target_include_directories(korcore PUBLIC src)
target_compile_definitions(korcore PUBLIC KOR_VERSION="${PROJECT_VERSION}")
if(NOT MSVC)
  # Warnings for our own code only (PUBLIC reaches kor-cli and kor-studio, not third-party libraries).
  target_compile_options(korcore PUBLIC -Wall -Wextra)
endif()

# The AVX2 kernels live in their own translation unit so the rest of the
# program still runs on CPUs without AVX2 (selected at runtime).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86|x86)$")
  target_sources(korcore PRIVATE src/core/kernels_avx2.cpp)
  target_compile_definitions(korcore PRIVATE KOR_HAVE_AVX2_TU)
  if(MSVC)
    set_source_files_properties(src/core/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  else()
    set_source_files_properties(src/core/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  endif()
endif()
if(NOT MSVC)
  # Let the compiler vectorize the hot loops in the kernel files.
  set_property(SOURCE src/core/kernels_base.cpp src/core/kernels_avx2.cpp
               APPEND PROPERTY COMPILE_OPTIONS "-O3;-fno-math-errno;-fno-trapping-math")
endif()

find_package(Threads REQUIRED)
target_link_libraries(korcore PUBLIC Threads::Threads)

add_executable(kor-cli src/cli/main.cpp)
target_link_libraries(kor-cli PRIVATE korcore)
if(WIN32)
  target_link_libraries(kor-cli PRIVATE shell32)
endif()

if(MINGW)
  # Ship self-contained executables (no libstdc++/winpthread DLLs needed).
  target_link_options(kor-cli PRIVATE -static)
endif()

enable_testing()
add_test(NAME selftest COMMAND kor-cli selftest)

# ------------------------------------------------------------------- GUI
if(KOR_BUILD_GUI)
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(glfw imgui)
  find_package(OpenGL REQUIRED)

  add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
  if(NOT MSVC)
    target_compile_options(imgui PRIVATE -w)
  endif()

  set(KOR_APP_SOURCES
    src/app/main.cpp
    src/app/ui.cpp
    src/app/kit.cpp
    src/app/i18n.cpp
    src/app/platform.cpp
    src/app/plugins.cpp)
  if(WIN32)
    list(APPEND KOR_APP_SOURCES assets/kor.rc)
  endif()
  add_executable(kor-studio WIN32 ${KOR_APP_SOURCES})
  target_link_libraries(kor-studio PRIVATE korcore imgui)
  if(WIN32)
    target_link_libraries(kor-studio PRIVATE comdlg32)
  endif()
  if(MSVC)
    target_link_options(kor-studio PRIVATE /ENTRY:mainCRTStartup)
  endif()
  if(MINGW)
    target_link_options(kor-studio PRIVATE -static)
  endif()
  # Plugins call kor-cli, so build it together with the app.
  add_dependencies(kor-studio kor-cli)
  # Put the example datasets and the bundled plugins next to the executable.
  add_custom_command(TARGET kor-studio POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/examples $<TARGET_FILE_DIR:kor-studio>/examples
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/plugins $<TARGET_FILE_DIR:kor-studio>/plugins)
endif()

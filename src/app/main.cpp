// Kor Studio - desktop application entry point (GLFW + OpenGL + Dear ImGui).
#include "i18n.h"
#include "ui.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Latin (incl. Turkish), Cyrillic and common punctuation.
static const ImWchar kGlyphRanges[] = {0x0020, 0x00FF, 0x0100, 0x017F, 0x0400, 0x052F, 0x2010, 0x205E, 0};

static const char* find_font() {
    static const char* const candidates[] = {
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    };
    for (const char* c : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(c, ec)) return c;
    }
    return nullptr;
}

// Saves the framebuffer as a 24-bit BMP (used by --shot for documentation).
static bool save_bmp(const char* path, int w, int h) {
    std::vector<unsigned char> rgb((size_t)w * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    int stride = (w * 3 + 3) & ~3;
    unsigned size = 54 + (unsigned)stride * h;
    unsigned char hdr[54] = {'B', 'M'};
    auto put = [&](int off, unsigned v) { for (int i = 0; i < 4; ++i) hdr[off + i] = (unsigned char)(v >> (8 * i)); };
    put(2, size); put(10, 54); put(14, 40); put(18, (unsigned)w); put(22, (unsigned)h);
    hdr[26] = 1; hdr[28] = 24;
    put(34, (unsigned)stride * h);
    std::ofstream o(std::filesystem::u8path(path), std::ios::binary);
    if (!o) return false;
    o.write((const char*)hdr, 54);
    std::vector<unsigned char> row(stride, 0);
    for (int y = 0; y < h; ++y) {  // OpenGL rows are already bottom-up, like BMP
        const unsigned char* src = rgb.data() + (size_t)y * w * 3;
        for (int x = 0; x < w; ++x) {
            row[x * 3] = src[x * 3 + 2];
            row[x * 3 + 1] = src[x * 3 + 1];
            row[x * 3 + 2] = src[x * 3];
        }
        o.write((const char*)row.data(), stride);
    }
    return (bool)o;
}

int main(int argc, char** argv) {
    // Optional scripting flags, mainly for producing screenshots:
    //   --lang en|tr|ru  --page N  --data <file>  --train  --shot <out.bmp>  --delay <seconds>
    std::string shot, data;
    int page = -1, lang = -1;
    bool train = false;
    double delay = 1.0;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (!std::strcmp(argv[i], "--shot")) shot = next();
        else if (!std::strcmp(argv[i], "--data")) data = next();
        else if (!std::strcmp(argv[i], "--page")) page = std::atoi(next());
        else if (!std::strcmp(argv[i], "--delay")) delay = std::atof(next());
        else if (!std::strcmp(argv[i], "--train")) train = true;
        else if (!std::strcmp(argv[i], "--lang")) {
            std::string l = next();
            lang = l == "tr" ? LANG_TR : l == "ru" ? LANG_RU : LANG_EN;
        }
    }

    if (!glfwInit()) {
        std::fprintf(stderr, "Kor Studio: cannot initialize the window system\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(1320, 820, "Kor Studio", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Kor Studio: cannot create an OpenGL 3.0 window\n");
        glfwTerminate();
        return 1;
    }
    glfwSetWindowSizeLimits(window, 900, 600, GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    float xs = 1.f, ys = 1.f;
    glfwGetWindowContentScale(window, &xs, &ys);
    float scale = xs < 1.f ? 1.f : xs;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImFont* title = nullptr;
    if (const char* font = find_font()) {
        io.Fonts->AddFontFromFileTTF(font, 18.f * scale, nullptr, kGlyphRanges);
        title = io.Fonts->AddFontFromFileTTF(font, 30.f * scale, nullptr, kGlyphRanges);
    } else {
        io.Fonts->AddFontDefault();  // ASCII only: Turkish/Russian text needs a system font
    }
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    app::init(scale, title);
    if (lang >= 0) set_language(lang);
    if (!data.empty()) app::load_data(data.c_str());
    if (train) app::start_training();
    if (page >= 0) app::set_page(page);

    auto t0 = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        app::frame();
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.086f, 0.074f, 0.066f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (!shot.empty() && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > delay) {
            save_bmp(shot.c_str(), w, h);
            break;
        }
        glfwSwapBuffers(window);
    }

    app::shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

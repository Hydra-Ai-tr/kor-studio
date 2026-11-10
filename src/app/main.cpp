// Kor Studio - desktop application entry point (GLFW + OpenGL + Dear ImGui).
#include "i18n.h"
#include "kit.h"
#include "ui.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

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
    //   --lang en|tr|ru  --page N  --tab N  --data <file>  --train  --shot <out.bmp>  --delay <seconds>
    //   --record <folder> --every <seconds>   saves frame_0000.bmp, ... until --delay
    //   --size <width>x<height>  --skill <plugin file> (repeatable; only for this session)
    //   --bottom (scroll to the end of the page)  --community (load the community plugin list)
    // Screenshot runs leave the settings file unchanged.
    std::string shot, data, record;
    std::vector<const char*> skills;
    int page = -1, lang = -1, tab = -1, width = 1320, height = 820;
    bool train = false, use_skills = false, bottom = false, community = false;
    double delay = 1.0, every = 0.5;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (!std::strcmp(argv[i], "--shot")) shot = next();
        else if (!std::strcmp(argv[i], "--data")) data = next();
        else if (!std::strcmp(argv[i], "--page")) page = std::atoi(next());
        else if (!std::strcmp(argv[i], "--tab")) tab = std::atoi(next());
        else if (!std::strcmp(argv[i], "--delay")) delay = std::atof(next());
        else if (!std::strcmp(argv[i], "--record")) record = next();
        else if (!std::strcmp(argv[i], "--every")) every = std::max(0.05, std::atof(next()));
        else if (!std::strcmp(argv[i], "--size")) std::sscanf(next(), "%dx%d", &width, &height);
        else if (!std::strcmp(argv[i], "--train")) train = true;
        else if (!std::strcmp(argv[i], "--skill")) { skills.push_back(next()); use_skills = true; }
        else if (!std::strcmp(argv[i], "--bottom")) bottom = true;
        else if (!std::strcmp(argv[i], "--community")) community = true;
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
    GLFWwindow* window = glfwCreateWindow(std::max(900, width), std::max(600, height), "Kor Studio", nullptr, nullptr);
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
    if (!shot.empty() || !record.empty()) io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    app::init(scale);
    if (!shot.empty() || !record.empty()) app::keep_settings();
    if (use_skills) app::use_skills_for_session(skills.data(), (int)skills.size());
    if (bottom) app::scroll_to_end();
    if (community) app::load_community_list();
    if (lang >= 0) set_language(lang);
    if (!data.empty()) app::load_data(data.c_str());
    if (train) app::start_training();
    if (page >= 0) app::set_page(page);
    if (tab >= 0) app::set_train_tab(tab);

    auto t0 = std::chrono::steady_clock::now();
    int frame_no = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        for (int pass = 0; pass < 2; ++pass) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            if (io.DeltaTime <= 0.f) io.DeltaTime = 1e-4f;
            ImGui::NewFrame();
            app::frame();
            ImGui::Render();
            if (!kit::take_relayout()) break;
        }
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(kit::C::bg.x, kit::C::bg.y, kit::C::bg.z, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!record.empty() && t >= frame_no * every) {
            char name[32];
            std::snprintf(name, sizeof name, "/frame_%04d.bmp", frame_no++);
            save_bmp((record + name).c_str(), w, h);
        }
        if ((!shot.empty() || !record.empty()) && t > delay) {
            if (!shot.empty()) save_bmp(shot.c_str(), w, h);
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

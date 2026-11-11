// kor-cli serve: a small local HTTP server that answers like the OpenAI API,
// so other programs (chat front-ends, bots, scripts) can use a .kor text model.
//
//   GET  /v1/models
//   POST /v1/chat/completions   {"messages": [...], "max_tokens", "temperature", "top_p", "stream"}
//   POST /v1/completions        {"prompt": "...", "max_tokens", "temperature", "top_p", "stream"}
//   (also "top_k" and "repetition_penalty", as in other local servers)
//
// One request is handled at a time; every response closes the connection.
#include "serve.h"

#include "../core/gguf.h"
#include "../core/model.h"
#include "../core/parallel.h"
#include "../core/sysinfo.h"
#include "../skills/json.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
static const socket_t kNoSocket = INVALID_SOCKET;
static void close_socket(socket_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
using socket_t = int;
static const socket_t kNoSocket = -1;
static void close_socket(socket_t s) { close(s); }
#endif

namespace kor {
namespace {

std::atomic<bool> g_stop{false};
void on_stop(int) { g_stop = true; }

std::string arg_value(const std::vector<std::string>& a, const char* name, const char* def) {
    for (size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == name) return a[i + 1];
    return def;
}

bool arg_flag(const std::vector<std::string>& a, const char* name) { return std::find(a.begin(), a.end(), name) != a.end(); }

bool send_all(socket_t s, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        int n = send(s, data.data() + off, (int)std::min<size_t>(data.size() - off, 1 << 16), 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

struct Request {
    std::string method, path, body;
    std::string auth;  // the Authorization header
};

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Reads one request. Bodies are limited to 4 MB.
bool read_request(socket_t s, Request& r) {
    std::string buf;
    char tmp[8192];
    size_t head_end;
    while ((head_end = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > 65536) return false;
        int n = recv(s, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
    }
    std::string head = buf.substr(0, head_end);
    size_t line_end = head.find("\r\n");
    std::string first = head.substr(0, line_end);
    size_t sp1 = first.find(' '), sp2 = first.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    r.method = first.substr(0, sp1);
    r.path = first.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t q = r.path.find('?');
    if (q != std::string::npos) r.path.resize(q);
    size_t length = 0;
    size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
    while (pos < head.size()) {
        size_t e = head.find("\r\n", pos);
        if (e == std::string::npos) e = head.size();
        std::string line = head.substr(pos, e - pos);
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = lower(line.substr(0, colon)), value = line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') value.erase(value.begin());
            if (key == "content-length") length = (size_t)std::strtoull(value.c_str(), nullptr, 10);
            if (key == "authorization") r.auth = value;
        }
        pos = e + 2;
    }
    if (length > (4u << 20)) return false;
    r.body = buf.substr(head_end + 4);
    while (r.body.size() < length) {
        int n = recv(s, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        r.body.append(tmp, (size_t)n);
    }
    r.body.resize(std::min(r.body.size(), length));
    return true;
}

const char* status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        default: return "Internal Server Error";
    }
}

struct Server {
    std::shared_ptr<Model> model;
    std::shared_ptr<GgufModel> llm;  // a GGUF model instead of a .kor one
    std::string model_id, q_tag, a_tag, api_key;
    bool cors = false;
    int newline = -1;
    Rng rng{(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()};

    std::string headers(int code, const char* type, long long length) const {
        std::string h = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) + "\r\n";
        h += std::string("Content-Type: ") + type + "\r\n";
        if (length >= 0) h += "Content-Length: " + std::to_string(length) + "\r\n";
        if (cors)
            h += "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\n"
                 "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        h += "Connection: close\r\n\r\n";
        return h;
    }

    void reply(socket_t s, int code, const std::string& body, const char* type = "application/json") const {
        send_all(s, headers(code, type, (long long)body.size()) + body);
    }

    void error(socket_t s, int code, const std::string& msg) const {
        json::Writer w;
        w.begin_object().key("error").begin_object().field("message", msg).field("type", "invalid_request_error").end_object().end_object();
        reply(s, code, w.str());
    }

    // The conversation in the format of the training text, newest last, cut
    // from the front so that it and the answer fit in the model's context.
    std::vector<int> chat_prompt(const json::Value& messages, int max_tokens) const {
        std::vector<std::vector<int>> parts;
        for (const json::Value& m : messages.a) {
            const std::string& role = m["role"].str();
            std::string content = m["content"].str();
            for (char& c : content)
                if (c == '\n' || c == '\r') c = ' ';
            if (role == "user" || role == "system") parts.push_back(model->encode(q_tag + content + "\n"));
            else if (role == "assistant") parts.push_back(model->encode(a_tag + content + "\n"));
        }
        std::vector<int> tail = model->encode(a_tag);
        int budget = std::max(8, model->spec.ctx - std::min(max_tokens, model->spec.ctx / 2)) - (int)tail.size();
        std::vector<int> ids;
        for (int i = (int)parts.size() - 1; i >= 0; --i) {
            if ((int)parts[i].size() > budget) {
                if (ids.empty()) ids.assign(parts[i].end() - std::max(0, budget), parts[i].end());
                break;
            }
            budget -= (int)parts[i].size();
            ids.insert(ids.begin(), parts[i].begin(), parts[i].end());
        }
        ids.insert(ids.end(), tail.begin(), tail.end());
        return ids;
    }

    std::string chunk_json(const char* object, const std::string& id, bool chat, const std::string& text, bool first, const char* finish) const {
        json::Writer w;
        w.begin_object().field("id", id).field("object", object).field("created", (long long)std::time(nullptr)).field("model", model_id);
        w.key("choices").begin_array().begin_object().field("index", 0);
        if (chat) {
            w.key("delta").begin_object();
            if (first) w.field("role", "assistant");
            if (!text.empty()) w.field("content", text);
            w.end_object();
        } else {
            w.field("text", text);
        }
        w.key("finish_reason");
        if (finish) w.value(finish);
        else w.null();
        w.end_object().end_array().end_object();
        return w.str();
    }

    void complete(socket_t s, const json::Value& body, bool chat) {
        int max_tokens = (int)body["max_tokens"].num(body["max_completion_tokens"].num(200));
        max_tokens = std::max(1, std::min(max_tokens, 4096));
        Sampling smp;
        smp.temperature = (float)std::max(0.0, body["temperature"].num(0.8));
        smp.top_p = (float)std::max(0.0, std::min(1.0, body["top_p"].num(1.0)));
        smp.top_k = std::max(0, (int)body["top_k"].num(0));
        smp.repeat_penalty = (float)std::max(0.1, std::min(10.0, body["repetition_penalty"].num(1.0)));
        bool stream = body["stream"].flag();
        if (llm) return complete_llm(s, body, chat, max_tokens, smp, stream);
        std::vector<int> ids;
        if (chat) {
            if (body["messages"].type != json::Value::ARR || body["messages"].a.empty()) return error(s, 400, "messages must be a non-empty array");
            ids = chat_prompt(body["messages"], max_tokens);
        } else {
            const json::Value& p = body["prompt"];
            std::string text = p.type == json::Value::ARR && !p.a.empty() ? p.a[0].str() : p.str();
            ids = model->encode(text);
            int keep = std::max(1, model->spec.ctx - std::min(max_tokens, model->spec.ctx / 2));
            if ((int)ids.size() > keep) ids.erase(ids.begin(), ids.end() - keep);
        }
        if (ids.empty()) ids.push_back(newline >= 0 ? newline : 0);
        std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(rng.next() % 1000000000ull);
        const char* object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";
        bool one_line = chat && newline >= 0;
        std::string text;
        int produced = 0;
        bool ended = false, alive = true, first = true;
        if (stream) alive = send_all(s, headers(200, "text/event-stream", -1));
        model->generate(ids, max_tokens, smp, rng, one_line ? newline : -1, [&](int t) {
            ++produced;
            if (one_line && t == newline) {
                ended = true;
                return false;
            }
            std::string piece = t >= 0 && t < (int)model->vocab.size() ? model->vocab[t] : "";
            text += piece;
            if (stream && alive) {
                alive = send_all(s, "data: " + chunk_json(object, id, chat, piece, first, nullptr) + "\n\n");
                first = false;
            }
            return alive && !g_stop.load();
        });
        const char* finish = ended || produced < max_tokens ? "stop" : "length";
        if (stream) {
            if (alive) send_all(s, "data: " + chunk_json(object, id, chat, "", first, finish) + "\n\ndata: [DONE]\n\n");
            return;
        }
        json::Writer w;
        w.begin_object().field("id", id).field("object", object).field("created", (long long)std::time(nullptr)).field("model", model_id);
        w.key("choices").begin_array().begin_object().field("index", 0);
        if (chat) w.key("message").begin_object().field("role", "assistant").field("content", text).end_object();
        else w.field("text", text);
        w.field("finish_reason", finish).end_object().end_array();
        int completion = produced - (ended ? 1 : 0);
        w.key("usage").begin_object().field("prompt_tokens", (int)ids.size()).field("completion_tokens", completion)
            .field("total_tokens", (int)ids.size() + completion).end_object();
        w.end_object();
        reply(s, 200, w.str());
    }

    void complete_llm(socket_t s, const json::Value& body, bool chat, int max_tokens, const Sampling& smp, bool stream) {
        std::vector<int> ids;
        if (chat) {
            if (body["messages"].type != json::Value::ARR || body["messages"].a.empty()) return error(s, 400, "messages must be a non-empty array");
            std::vector<LlmMsg> msgs;
            for (const json::Value& m : body["messages"].a) {
                const json::Value& c = m["content"];
                std::string text;
                if (c.type == json::Value::ARR) {
                    for (const json::Value& part : c.a) text += part["text"].str();
                } else {
                    text = c.str();
                }
                msgs.push_back({m["role"].str(), text});
            }
            ids = llm->chat_prompt(msgs, max_tokens);
        } else {
            const json::Value& p = body["prompt"];
            ids = llm->encode(p.type == json::Value::ARR && !p.a.empty() ? p.a[0].str() : p.str(), true, false);
            int keep = std::max(1, llm->ctx - std::min(max_tokens, llm->ctx / 2));
            if ((int)ids.size() > keep) ids.erase(ids.begin(), ids.end() - keep);
        }
        if (ids.empty()) return error(s, 400, "empty prompt");
        std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(rng.next() % 1000000000ull);
        const char* object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";
        bool alive = true, first = true;
        if (stream) alive = send_all(s, headers(200, "text/event-stream", -1));
        int produced = 0;
        std::string text = llm->reply(ids, max_tokens, smp, rng, [&](const std::string& piece) {
            if (stream && alive) {
                alive = send_all(s, "data: " + chunk_json(object, id, chat, piece, first, nullptr) + "\n\n");
                first = false;
            }
            return alive && !g_stop.load();
        }, &produced);
        const char* finish = produced < max_tokens ? "stop" : "length";
        if (stream) {
            if (alive) send_all(s, "data: " + chunk_json(object, id, chat, "", first, finish) + "\n\ndata: [DONE]\n\n");
            return;
        }
        json::Writer w;
        w.begin_object().field("id", id).field("object", object).field("created", (long long)std::time(nullptr)).field("model", model_id);
        w.key("choices").begin_array().begin_object().field("index", 0);
        if (chat) w.key("message").begin_object().field("role", "assistant").field("content", text).end_object();
        else w.field("text", text);
        w.field("finish_reason", finish).end_object().end_array();
        w.key("usage").begin_object().field("prompt_tokens", (int)ids.size()).field("completion_tokens", produced)
            .field("total_tokens", (int)ids.size() + produced).end_object();
        w.end_object();
        reply(s, 200, w.str());
    }

    void handle(socket_t s) {
        Request r;
        if (!read_request(s, r)) return;
        std::time_t now = std::time(nullptr);
        char when[16];
        std::strftime(when, sizeof when, "%H:%M:%S", std::localtime(&now));
        std::printf("%s  %s %s\n", when, r.method.c_str(), r.path.c_str());
        std::fflush(stdout);
        if (r.method == "OPTIONS") return reply(s, 204, "");
        if (!api_key.empty() && r.auth != "Bearer " + api_key) return error(s, 401, "missing or wrong API key");
        if (r.path == "/" || r.path == "/health") {
            return reply(s, 200, "Kor Studio model server: " + model_id + "\nPOST /v1/chat/completions, POST /v1/completions, GET /v1/models\n",
                         "text/plain; charset=utf-8");
        }
        if (r.path == "/v1/models") {
            json::Writer w;
            w.begin_object().field("object", "list").key("data").begin_array();
            w.begin_object().field("id", model_id).field("object", "model").field("created", (long long)now).field("owned_by", "kor-studio").end_object();
            w.end_array().end_object();
            return reply(s, 200, w.str());
        }
        bool chat = r.path == "/v1/chat/completions", plain = r.path == "/v1/completions";
        if (!chat && !plain) return error(s, 404, "unknown path " + r.path);
        if (r.method != "POST") return error(s, 405, "use POST");
        json::Value body;
        std::string err;
        if (!json::parse(r.body, body, &err) || body.type != json::Value::OBJ) return error(s, 400, "invalid JSON: " + err);
        complete(s, body, chat);
    }
};

}  // namespace

int serve(const std::vector<std::string>& a) {
    std::string path = a.size() > 2 ? a[2] : "", err;
    Server sv;
    std::string ext = std::filesystem::u8path(path).extension().u8string();
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    std::string params;
    if (ext == ".gguf") {
        int threads = std::atoi(arg_value(a, "--threads", "0").c_str());
        set_threads(threads > 0 ? threads : recommended_threads(detect_system()));
        sv.llm = GgufModel::load(path, &err, std::max(64, std::atoi(arg_value(a, "--ctx", "4096").c_str())));
        if (sv.llm)
            for (size_t i = 0; i + 1 < a.size(); ++i)
                if (a[i] == "--lora" && !sv.llm->lora_load(a[i + 1], &err)) {
                    sv.llm = nullptr;
                    break;
                }
        if (!sv.llm) {
            std::printf("error: %s\n", err.c_str());
            return 1;
        }
        sv.model_id = arg_value(a, "--name", sv.llm->name.c_str());
        params = std::to_string((long long)sv.llm->n_params);
    } else {
        auto m = Model::load(path, &err);
        if (!m) {
            std::printf("error: %s\n", err.c_str());
            return 1;
        }
        if (m->spec.kind != MODEL_GPT || m->spec.media != MEDIA_TEXT) {
            std::printf("error: %s is not a text model\n", path.c_str());
            return 1;
        }
        sv.model = m;
        auto ni = m->meta.find("name");
        std::string stem = ni != m->meta.end() && !ni->second.empty() ? ni->second : std::filesystem::u8path(path).stem().u8string();
        sv.model_id = arg_value(a, "--name", stem.c_str());
        auto qi = m->meta.find("q_tag"), ai = m->meta.find("a_tag");
        bool stored = qi != m->meta.end() && ai != m->meta.end();
        sv.q_tag = arg_value(a, "--q-tag", stored ? qi->second.c_str() : "Kullanıcı: ");
        sv.a_tag = arg_value(a, "--a-tag", stored ? ai->second.c_str() : "Asistan: ");
        m->gen_on_gpu = arg_flag(a, "--gpu");
        for (size_t i = 0; i < m->vocab.size(); ++i)
            if (m->vocab[i] == "\n") sv.newline = (int)i;
        params = std::to_string((long long)m->param_count());
    }
    sv.api_key = arg_value(a, "--api-key", "");
    sv.cors = arg_flag(a, "--cors");
    std::string host = arg_value(a, "--host", "127.0.0.1");
    int port = std::atoi(arg_value(a, "--port", "8080").c_str());

#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::printf("error: cannot start networking\n");
        return 1;
    }
#endif
    socket_t ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == kNoSocket) {
        std::printf("error: cannot create a socket\n");
        return 1;
    }
    int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof yes);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        std::printf("error: --host must be an IPv4 address such as 127.0.0.1\n");
        return 1;
    }
    if (bind(ls, (sockaddr*)&addr, sizeof addr) != 0 || listen(ls, 16) != 0) {
        std::printf("error: cannot listen on %s:%d (is the port in use?)\n", host.c_str(), port);
        close_socket(ls);
        return 1;
    }
    std::signal(SIGINT, on_stop);
#if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN);  // a client that hangs up must not end the server
#endif
    std::printf("Serving %s (%s parameters) at http://%s:%d/v1\n", sv.model_id.c_str(), params.c_str(), host.c_str(), port);
    if (host != "127.0.0.1") std::printf("warning: reachable from other computers on the network%s\n", sv.api_key.empty() ? " without an API key" : "");
    if (sv.llm) std::printf("Chat format: %s   Ctrl+C stops.\n", sv.llm->chat_format().c_str());
    else std::printf("Chat format: \"%s...\" / \"%s...\"   Ctrl+C stops.\n", sv.q_tag.c_str(), sv.a_tag.c_str());
    std::fflush(stdout);
    while (!g_stop) {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(ls, &set);
        timeval tv{0, 250000};
        if (select((int)ls + 1, &set, nullptr, nullptr, &tv) <= 0) continue;
        socket_t c = accept(ls, nullptr, nullptr);
        if (c == kNoSocket) continue;
#if defined(_WIN32)
        DWORD timeout = 15000;
#else
        timeval timeout{15, 0};
#endif
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout);
        sv.handle(c);
        close_socket(c);
    }
    close_socket(ls);
#if defined(_WIN32)
    WSACleanup();
#endif
    std::printf("stopped\n");
    return 0;
}

}  // namespace kor

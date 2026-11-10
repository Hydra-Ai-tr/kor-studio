#include "skill.h"

#include "json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <climits>
#include <unistd.h>
#endif

#ifndef KOR_VERSION
#define KOR_VERSION "dev"
#endif

namespace kor {

using json::Value;
using json::Writer;

static const char* const kHookNames[] = {"start", "step", "eval", "tests", "end"};
static const char kPrefix[] = "@kor ";

static int tone_of(const std::string& t) {
    if (t == "good") return TONE_GOOD;
    if (t == "warn") return TONE_WARN;
    if (t == "bad" || t == "error") return TONE_BAD;
    if (t == "muted") return TONE_MUTED;
    return TONE_INFO;
}

void PythonSkill::fail(TrainControl& ctl, const std::string& why) {
    if (!alive_) return;
    alive_ = false;
    ctl.log(why, TONE_BAD);
    proc_.kill();
}

bool PythonSkill::open(TrainControl& ctl, std::string* err) {
    if (spec_.python.empty()) {
        if (err) *err = "Python 3 was not found";
        return false;
    }
    std::vector<std::string> cmd = python_launch(spec_.python, spec_.script);
    cmd.insert(cmd.end(), spec_.args.begin(), spec_.args.end());
    EnvList env = spec_.env;
    env.push_back({"PYTHONIOENCODING", "utf-8"});
    env.push_back({"PYTHONUNBUFFERED", "1"});
    env.push_back({"KOR_SKILL", "1"});
    env.push_back({"KOR_LANG", spec_.lang});
    if (!proc_.start(cmd, env, err)) return false;
    alive_ = true;

    const Model& m = ctl.model();
    const Dataset& d = ctl.data();
    const TrainConfig& c = ctl.config();
    Model& mm = const_cast<Model&>(m);  // parameter counting is not const
    bool text = m.spec.kind == MODEL_GPT;
    Writer w;
    w.begin_object().field("type", "hello").field("protocol", 1).field("kor_version", KOR_VERSION)
        .field("skill", spec_.name).field("lang", spec_.lang);
    w.key("args").begin_array();
    for (auto& a : spec_.args) w.value(a);
    w.end_array();
    w.key("model").begin_object()
        .field("kind", text ? "text" : "table")
        .field("parameters", mm.param_count(false))
        .field("trainable", mm.param_count(true))
        .field("lora_rank", m.spec.lora_rank);
    if (text) {
        w.field("vocab_size", m.spec.vocab).field("context", m.spec.ctx).field("width", m.spec.dim)
            .field("layers", m.spec.nlayer).field("heads", m.spec.heads);
    } else {
        w.field("inputs", m.spec.in_dim).field("outputs", m.spec.out_dim).field("regression", m.spec.regression);
        w.key("classes").begin_array();
        if (!m.spec.regression) for (auto& v : m.vocab) w.value(v);
        w.end_array();
    }
    w.end_object();
    w.key("data").begin_object()
        .field("path", d.path)
        .field("kind", text ? "text" : "table")
        .field("train", d.train_items())
        .field("validation", d.val_items())
        .field("question_tag", spec_.q_tag)
        .field("answer_tag", spec_.a_tag);
    if (text) {
        std::string chars;
        for (auto& v : m.vocab) chars += v;
        w.field("characters", chars);
    }
    w.end_object();
    w.key("config").begin_object()
        .field("steps", c.steps).field("batch", c.batch).field("accum", c.accum)
        .field("lr", (double)c.lr).field("weight_decay", (double)c.weight_decay)
        .field("eval_every", c.eval_every).field("test_every", c.probe_every)
        .field("optimizer", c.optimizer == OPT_SGD ? "sgd" : "adamw")
        .field("keep_best", c.keep_best).field("patience", c.patience)
        .end_object();
    w.key("tests").begin_array();
    for (auto& p : c.probes) w.begin_object().field("question", p.question).field("expected", p.expected).end_object();
    w.end_array();
    w.end_object();
    if (!proc_.write_line(w.str())) {
        alive_ = false;
        proc_.kill();
        if (err) *err = "the skill did not start";
        return false;
    }
    std::string why;
    if (!converse(ctl, spec_.start_timeout_ms, true, &why)) {
        alive_ = false;
        proc_.kill();
        if (err) *err = why;
        return false;
    }
    return true;
}

void PythonSkill::on_event(const HookInfo& in, TrainControl& ctl) {
    if (!alive_) return;
    Writer w;
    w.begin_object().field("type", "event").field("event", kHookNames[in.kind])
        .field("step", in.step).field("total", in.total).field("elapsed", in.elapsed)
        .field("loss", (double)in.loss).field("lr", (double)in.lr);
    w.key("val_loss");
    if (in.val_loss >= 0) w.value((double)in.val_loss); else w.null();
    w.key("val_acc");
    if (in.val_acc >= 0) w.value((double)in.val_acc); else w.null();
    w.key("best_val");
    if (in.best_val >= 0) w.value((double)in.best_val); else w.null();
    w.field("best_step", in.best_step).field("improved", in.improved);
    if (in.kind == HOOK_TESTS && in.tests && in.results) {
        w.field("passed", in.passed).field("total_tests", in.tests->size());
        w.key("results").begin_array();
        size_t n = std::min(in.tests->size(), in.results->size());
        for (size_t i = 0; i < n; ++i)
            w.begin_object()
                .field("question", (*in.tests)[i].question)
                .field("expected", (*in.tests)[i].expected)
                .field("answer", (*in.results)[i].answer)
                .field("passed", (*in.results)[i].pass)
                .end_object();
        w.end_array();
    }
    if (in.kind == HOOK_END) w.field("state", in.state);
    w.end_object();
    if (!proc_.write_line(w.str())) {
        fail(ctl, "the skill has exited");
        return;
    }
    std::string why;
    if (!converse(ctl, spec_.event_timeout_ms, false, &why)) fail(ctl, why);
}

bool PythonSkill::converse(TrainControl& ctl, int timeout_ms, bool opening, std::string* err) {
    auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::string line;
    for (;;) {
        int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(until - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            *err = "no answer for " + std::to_string(timeout_ms / 1000) + " s - the skill was turned off";
            return false;
        }
        int r = proc_.read_line(line, std::min(left, 500));
        if (r == 0) continue;
        if (r < 0) {
            int code = proc_.wait(1000);
            *err = "the skill has exited" + (code > -1000 ? " (code " + std::to_string(code) + ")" : std::string());
            return false;
        }
        if (line.compare(0, sizeof kPrefix - 1, kPrefix) != 0) {
            if (line.size() > 2000) line.resize(2000);
            ctl.log(line, TONE_MUTED);
            continue;
        }
        Value msg;
        std::string perr;
        if (!json::parse(line.substr(sizeof kPrefix - 1), msg, &perr)) {
            ctl.log("bad message from the skill (" + perr + ")", TONE_WARN);
            continue;
        }
        const std::string& op = msg["op"].str();
        bool has_id = msg.has("id");
        Writer reply;
        reply.begin_object().field("type", "reply").field("id", (long long)msg["id"].num()).field("ok", true);
        bool replied = false;

        if (op == "ready") {
            for (auto& e : msg["events"].a)
                for (int k = 0; k < 5; ++k)
                    if (e.str() == kHookNames[k]) want_[k] = true;
            every_ = std::max(0, (int)msg["every"].num());
            if (every_ > 0) want_[HOOK_STEP] = true;
            if (opening) return true;
        } else if (op == "done") {
            if (!opening) return true;
        } else if (op == "log") {
            ctl.log(msg["text"].str(), tone_of(msg["tone"].str()));
        } else if (op == "metric") {
            ctl.metric(msg["name"].str(), (float)msg["value"].num(NAN));
        } else if (op == "lr_factor") {
            ctl.set_lr_factor((float)msg["value"].num(1));
        } else if (op == "lr") {
            ctl.set_lr((float)msg["value"].num(0));
        } else if (op == "stop") {
            ctl.stop(msg["reason"].str());
        } else if (op == "save_best") {
            ctl.save_best();
        } else if (op == "restore_best") {
            reply.key("value").value(ctl.restore_best());
            replied = true;
        } else if (op == "add_text") {
            reply.key("value").value(ctl.add_text(msg["text"].str()));
            replied = true;
        } else if (op == "clear_added") {
            ctl.clear_added();
        } else if (op == "mix") {
            ctl.set_mix((float)msg["value"].num(-1));
        } else if (op == "generate") {
            std::string out = ctl.generate(msg["prompt"].str(), (int)msg["length"].num(200),
                                           (float)msg["temperature"].num(0.8), msg["line"].flag());
            reply.key("value").value(out);
            replied = true;
        } else if (op == "ask") {
            Probe p = make_probe(msg["question"].str(), msg["expected"].str(), spec_.q_tag, spec_.a_tag);
            ProbeResult res = ctl.ask(p);
            reply.key("value").begin_object().field("answer", res.answer).field("passed", res.pass).end_object();
            replied = true;
        } else {
            ctl.log("unknown command \"" + op + "\"", TONE_WARN);
            if (has_id) {
                Writer bad;
                bad.begin_object().field("type", "reply").field("id", (long long)msg["id"].num()).field("ok", false)
                    .field("error", "unknown command: " + op).end_object();
                proc_.write_line(bad.str());
            }
            continue;
        }
        if (has_id) {
            if (!replied) reply.key("value").null();
            reply.end_object();
            if (!proc_.write_line(reply.str())) {
                *err = "the skill has exited";
                return false;
            }
        }
    }
}

void PythonSkill::close() {
    if (!alive_) return;
    alive_ = false;
    proc_.write_line("{\"type\":\"bye\"}");
    proc_.close_input();
    if (proc_.wait(3000) == -1000) proc_.kill();
}

// ---------------------------------------------------------------- helpers

std::vector<std::string> python_launch(const std::vector<std::string>& python, const std::string& script) {
    static const char* const boot =
        "import os, runpy, sys\n"
        "s = sys.argv[1]\n"
        "sys.argv = sys.argv[1:]\n"
        "lib = os.environ.get('KOR_PYLIB')\n"
        "sys.path[0:1] = ([lib] if lib else []) + [os.path.dirname(os.path.abspath(s))]\n"
        "runpy.run_path(s, run_name='__main__')\n";
    std::vector<std::string> cmd = python;
    cmd.insert(cmd.end(), {"-X", "utf8", "-u", "-c", boot, script});
    return cmd;
}

std::vector<std::string> split_command(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false, any = false;
    for (char c : s) {
        if (c == '"') { quoted = !quoted; any = true; }
        else if ((c == ' ' || c == '\t') && !quoted) {
            if (any) out.push_back(cur);
            cur.clear();
            any = false;
        } else {
            cur += c;
            any = true;
        }
    }
    if (any) out.push_back(cur);
    return out;
}

std::vector<std::string> find_python3(std::string* version) {
    std::vector<std::vector<std::string>> candidates;
    if (const char* env = std::getenv("KOR_PYTHON"))
        if (*env) candidates.push_back(split_command(env));
#if defined(_WIN32)
    candidates.push_back({"py", "-3"});
    candidates.push_back({"python"});
    candidates.push_back({"python3"});
#else
    candidates.push_back({"python3"});
    candidates.push_back({"python"});
#endif
    for (auto& c : candidates) {
        std::vector<std::string> cmd = c;
        cmd.push_back("--version");
        std::string out;
        if (run_capture(cmd, {}, out, 15000) == 0 && out.compare(0, 8, "Python 3") == 0) {
            if (version) *version = out.substr(0, out.find_first_of("\r\n"));
            return c;
        }
    }
    return {};
}

std::string program_dir() {
#if defined(_WIN32)
    wchar_t buf[4096];
    DWORD n = GetModuleFileNameW(nullptr, buf, 4096);
    if (n > 0 && n < 4096) return std::filesystem::path(buf).parent_path().u8string();
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t size = sizeof buf;
    if (_NSGetExecutablePath(buf, &size) == 0) return std::filesystem::path(buf).parent_path().u8string();
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::filesystem::path(buf).parent_path().u8string();
    }
#endif
    return ".";
}

}  // namespace kor

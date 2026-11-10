// Kor Studio - Python training skills.
//
// A skill is a Python script that runs next to the trainer for the whole run.
// Kor writes one JSON object per line to its standard input ("hello", then an
// "event" for everything the skill subscribed to, then "bye"); the skill
// answers with lines that start with "@kor " followed by a JSON command, and
// ends every event with {"op": "done"}. Any other output becomes a log line.
// docs/plugins.md describes the protocol; python/kor/skill.py implements the
// Python side.
#pragma once
#include "../core/trainer.h"
#include "process.h"

#include <string>
#include <vector>

namespace kor {

struct SkillSpec {
    std::string name;    // shown in the log
    std::string script;  // the .py file
    std::vector<std::string> args;
    std::vector<std::string> python;  // interpreter command, e.g. {"py", "-3"}
    EnvList env;
    std::string lang = "en";
    std::string q_tag, a_tag;  // dialogue format used by "ask"
    int start_timeout_ms = 60000;
    int event_timeout_ms = 300000;
};

class PythonSkill : public TrainHook {
public:
    explicit PythonSkill(SkillSpec spec) : spec_(std::move(spec)) {}
    std::string name() const override { return spec_.name; }
    bool open(TrainControl& ctl, std::string* err) override;
    bool wants(int kind) const override { return kind >= 0 && kind < 5 && want_[kind]; }
    int step_every() const override { return every_; }
    void on_event(const HookInfo& info, TrainControl& ctl) override;
    void close() override;
    bool alive() const override { return alive_; }

private:
    // Handles the skill's messages until it says "done" (or "ready" during open).
    bool converse(TrainControl& ctl, int timeout_ms, bool opening, std::string* err);
    void fail(TrainControl& ctl, const std::string& why);

    SkillSpec spec_;
    Process proc_;
    bool alive_ = false;
    bool want_[5] = {};
    int every_ = 0;
};

// The command that runs a plugin script with $KOR_PYLIB first on the module
// path, so the `kor` package wins over stray kor.py files next to the script.
std::vector<std::string> python_launch(const std::vector<std::string>& python, const std::string& script);

// Splits a command line into arguments; double quotes group words.
std::vector<std::string> split_command(const std::string& s);
// Finds a Python 3 interpreter ($KOR_PYTHON, then py -3 / python3 / python).
std::vector<std::string> find_python3(std::string* version);
// Folder of the running program (UTF-8).
std::string program_dir();

}  // namespace kor

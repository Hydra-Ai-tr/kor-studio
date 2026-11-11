// kor-cli serve <model.kor>: an OpenAI-compatible HTTP server for a text model.
#pragma once
#include <string>
#include <vector>

namespace kor {
// a = the full command line ("kor-cli", "serve", "<model.kor>", flags...)
int serve(const std::vector<std::string>& a);
}

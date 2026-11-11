// Kor Studio - dataset tools: cleaning training text and importing JSON Lines
// chat datasets (OpenAI messages, ShareGPT, Alpaca, prompt/completion).
#pragma once
#include <string>

namespace kor {

struct CleanOptions {
    bool trim = true;            // spaces at line ends, tabs and odd whitespace
    bool collapse_spaces = true; // runs of spaces become one
    bool drop_empty = true;      // empty lines (more than one in a row for plain text)
    bool dedupe = true;          // repeated question-answer pairs, or repeated lines of plain text
    bool drop_broken = true;     // dialogue: questions without an answer and answers without a question
};

struct CleanReport {
    size_t lines_in = 0, lines_out = 0;
    size_t duplicates = 0, empty = 0, broken = 0, changed = 0;
    size_t bytes_in = 0, bytes_out = 0;
};

// With both tags set (e.g. "Kullanıcı: " and "Asistan: ") the text is treated
// as dialogue: a question line followed by its answer line.
std::string clean_text(const std::string& text, const std::string& q_tag, const std::string& a_tag,
                       const CleanOptions& o, CleanReport& r);

struct ImportReport {
    size_t records = 0, dialogues = 0, turns = 0, skipped = 0;
    std::string format;  // "messages", "sharegpt", "alpaca", "prompt", "text" or "mixed"
};

// Converts JSON Lines (or one JSON array) into training text: every user turn
// becomes a q_tag line and every assistant turn an a_tag line. Records with
// only a "text" field become paragraphs of plain text.
bool import_jsonl(const std::string& json_text, const std::string& q_tag, const std::string& a_tag, std::string& out,
                  ImportReport& r, std::string* err);

}  // namespace kor

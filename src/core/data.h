// Kor Studio - dataset loading: plain text (character-level) and CSV tables.
#pragma once
#include "model.h"

#include <string>
#include <vector>

namespace kor {

enum DataKind { DATA_NONE = 0, DATA_TEXT = 1, DATA_TABLE = 2 };

// How text batches are drawn.
struct SampleOpts {
    // Share of rows that start at the beginning of an example (a question turn
    // of dialogue data) instead of a random position.
    float align = 0;
    // Loss weight of tokens outside the answers of dialogue data (1 = all
    // tokens count the same). Answers keep weight 1.
    float prompt_weight = 1;
};

// The example structure of dialogue tokens: where question turns start and
// which tokens belong to an answer (the answer text after the answer tag and
// the newline that ends it).
struct TextLayout {
    std::vector<size_t> starts;
    std::vector<uint8_t> answer;
    bool empty() const { return starts.empty(); }
};
// Subword vocabulary by byte-pair merges: all characters plus the most frequent
// merged pieces, up to vocab_size entries. Pieces never contain a line break
// next to other characters.
std::vector<std::string> learn_subwords(const std::vector<std::string>& chars, int vocab_size);

TextLayout layout_dialogue(const std::vector<int>& ids, const std::vector<int>& q_tag, const std::vector<int>& a_tag, int newline);
// Start of a T-token window inside ids[lo, hi): an example start with
// probability `align` (when one fits), otherwise a random position.
size_t pick_window(const TextLayout& lay, size_t lo, size_t hi, int T, float align, Rng& rng);
// Copies the window starting at s into row `row` of a text batch. When b.w is
// in use, targets outside the answers get `prompt_weight`.
void fill_window(Batch& b, int row, const std::vector<int>& ids, size_t s, const TextLayout& lay, float prompt_weight);

struct Dataset {
    int kind = DATA_NONE;
    std::string path;

    // ---- token sequences: text, images or sound (see MediaKind)
    int media = MEDIA_TEXT;
    std::string raw;     // text only
    std::vector<int> ids;
    std::vector<std::string> vocab;
    size_t n_train = 0;  // ids[0, n_train) train, rest validation
    TextLayout layout;   // dialogue data, images and sound
    int img_size = 0;    // images: width = height
    int rate = 0;        // sound: samples per second
    int sep = -1;        // images and sound: token that starts each example
    int items = 0;       // images and sound: files read
    int skipped = 0;     // images and sound: files that could not be read

    // ---- table
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> preview;  // first raw rows
    int n_cols = 0, n_feat = 0, target_col = -1;
    Tensor X;                          // standardized features
    std::vector<int> cls;              // class index per row (classification)
    std::vector<float> yv;             // standardized target (regression)
    std::vector<std::string> labels;   // class names
    bool regression = false;
    std::vector<float> mean, stdv;     // n_feat entries + 1 for the target
    std::vector<int> train_idx, val_idx;
    int bad_cells = 0;                 // non-numeric feature cells replaced by 0

    // subword > 0 learns that many subword pieces (byte-pair merges) instead
    // of one token per character.
    bool load_text(const std::string& path, std::string* err, int subword = 0);
    // A folder of images (PNG, BMP, PPM, PGM): each one is cropped to a
    // square, scaled to size x size and reduced to a palette of `colors`.
    // mirror adds a left-right flipped copy of every training image. With a
    // palette (the vocabulary of an image model) its colours are used as they
    // are, so the images fit that model for fine-tuning.
    bool load_images(const std::string& path, int size, int colors, bool mirror, std::string* err,
                     const std::vector<std::string>* palette = nullptr);
    // A WAV file or a folder of them, resampled to `rate` and mu-law encoded.
    bool load_audio(const std::string& path, int rate, std::string* err);
    // target_col < 0 selects the last column.
    bool load_table(const std::string& path, int target_col, std::string* err);
    // Re-encodes raw text with a fixed vocabulary (for fine-tuning).
    void encode_with(const std::vector<std::string>& fixed_vocab);

    // Detects "Tag: text" dialogue data where two speakers alternate, e.g.
    // q_tag = "User: ", a_tag = "Bot: ". Returns false for ordinary text.
    bool dialogue_tags(std::string& q_tag, std::string& a_tag) const;
    // Up to n distinct question/answer pairs from dialogue data (tags removed),
    // in a fixed shuffled order. Questions with conflicting answers are skipped.
    std::vector<std::pair<std::string, std::string>> dialogue_pairs(int n) const;
    // dialogue_pairs() turned into question tests.
    std::vector<Probe> auto_probes(int n) const;
    // Removes every line that asks one of the test questions (with dialogue data: the
    // question turn and the answer after it) from the tokens the model trains and
    // validates on, so the tests measure answers to questions it has never seen.
    // raw is left unchanged. Returns how many question lines were removed.
    int hold_out(const std::vector<Probe>& tests);

    size_t train_items() const;
    size_t val_items() const;
    // Fills a random mini-batch. T and opts are only used for text.
    void sample(Batch& b, int B, int T, Rng& rng, bool val, const SampleOpts& opts = SampleOpts()) const;
    // Dialogue tags as token ids (empty when the data is not dialogue).
    bool tag_ids(std::vector<int>& q, std::vector<int>& a, int& newline) const;

private:
    void split_text();
};

// Builds a question test. With dialogue tags the prompt is
// q_tag + question + "\n" + a_tag, otherwise the question itself.
Probe make_probe(const std::string& question, const std::string& expected, const std::string& q_tag, const std::string& a_tag);
// Parses question tests written one per line as "question => expected answer".
// Empty lines and lines starting with '#' are ignored.
std::vector<Probe> parse_tests(const std::string& text, const std::string& q_tag, const std::string& a_tag);
// The inverse of parse_tests() for a list of question/answer pairs.
std::string format_tests(const std::vector<std::pair<std::string, std::string>>& pairs);

// Writes a small synthetic 3-class spiral dataset (CSV) for demos.
bool write_spiral_csv(const std::string& path, int per_class = 300);

}  // namespace kor

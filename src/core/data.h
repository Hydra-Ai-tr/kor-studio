// Kor Studio - dataset loading: plain text (character-level) and CSV tables.
#pragma once
#include "model.h"

#include <string>
#include <vector>

namespace kor {

enum DataKind { DATA_NONE = 0, DATA_TEXT = 1, DATA_TABLE = 2 };

struct Dataset {
    int kind = DATA_NONE;
    std::string path;

    // ---- text
    std::string raw;
    std::vector<int> ids;
    std::vector<std::string> vocab;
    size_t n_train = 0;  // ids[0, n_train) train, rest validation

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

    bool load_text(const std::string& path, std::string* err);
    // target_col < 0 selects the last column.
    bool load_table(const std::string& path, int target_col, std::string* err);
    // Re-encodes raw text with a fixed vocabulary (for fine-tuning).
    void encode_with(const std::vector<std::string>& fixed_vocab);

    size_t train_items() const;
    size_t val_items() const;
    // Fills a random mini-batch. T is only used for text.
    void sample(Batch& b, int B, int T, Rng& rng, bool val) const;

private:
    void split_text();
};

// Writes a small synthetic 3-class spiral dataset (CSV) for demos.
bool write_spiral_csv(const std::string& path, int per_class = 300);

}  // namespace kor

# Plugins

A Kor Studio plugin is a Python 3 script. Plugins prepare or generate training
data, write question tests and quiz trained models. They appear on the
**Plugins** page, where each one has an *Arguments* field and a *Run* button;
their output is shown live.

Kor Studio looks for Python in the standard install folders, then for the `py`
launcher and `python` on `PATH`. Plugins are read from two folders:

- `plugins/` next to the program (the bundled plugins and `kor.py`)
- your own folder: `%APPDATA%\KorStudio\plugins` on Windows,
  `~/.config/korstudio/plugins` elsewhere (*Open my plugins folder* creates it)

## Writing a plugin

The file starts with a comment block. The first line must be `# kor-plugin`:

```python
# kor-plugin
# name: Reverse lines
# name.tr: Satırları ters çevir
# description: Writes the loaded text with its lines in reverse order.
# description.tr: Yüklü metni satırları ters sırada yazar.
# args: --keep-empty

import kor

text = kor.read_data()
kor.write_data("\n".join(reversed(text.splitlines())))
```

| Header key | Meaning |
|---|---|
| `name`, `name.tr`, `name.ru` | Title shown in the list (English is the fallback) |
| `description`, `description.tr`, `description.ru` | One-paragraph explanation |
| `args` | Default contents of the *Arguments* field (passed to `sys.argv`) |

## What a plugin receives

Every run gets a fresh output folder and these environment variables:

| Variable | Contents |
|---|---|
| `KOR_OUT` | Folder for the plugin's results |
| `KOR_DATA` | Path of the loaded dataset (empty if none) |
| `KOR_MODEL` | A copy of the current model as a `.kor` file (empty if none) |
| `KOR_TESTS` | The current question tests, one `question => answer` per line |
| `KOR_CLI` | Path of `kor-cli`, used to run the model |
| `KOR_LANG` | Interface language: `en`, `tr` or `ru` |

`PYTHONPATH` contains the plugin folders, so `import kor` works everywhere.
Python runs in UTF-8 mode.

## What Kor Studio picks up

When the plugin exits with code 0:

| File in `KOR_OUT` | Effect |
|---|---|
| `data.txt` | Loaded as the new text dataset |
| `data.csv` / `data.tsv` | Loaded as the new table dataset |
| `tests.txt` | Becomes the question tests (Training > Question tests) |

Anything else the plugin writes (reports, plots) stays in the output folder.

## The `kor` helper module

| Function | Does |
|---|---|
| `read_data()` / `write_data(text, ext="txt")` | Read the loaded dataset / write a new one |
| `read_tests()` / `write_tests(pairs)` | Read / write question tests as `[(question, answer)]` |
| `generate(prompt, length, temperature, line=False)` | Continue a prompt with the current model |
| `run_tests(pairs)` | Ask the model every question; returns `TestResult(passed, question, expected, answer)` |
| `ask(question)` | The model's answer to one question |
| `tr(en, tr, ru)` | Pick a message in the interface language |
| `out_dir()`, `data_path()`, `model_path()`, `lang()` | The values above |

The model functions call `kor-cli`, which can also be used directly:

```bash
kor-cli generate model.kor "User: Hello" --len 200 --temp 0.8
kor-cli test model.kor tests.txt --data dialogue.txt
```

## Bundled plugins

| Plugin | What it does |
|---|---|
| `clean_text.py` | Removes invisible characters, odd spaces and typographic quotes; `--dedupe` drops repeated question/answer pairs |
| `math_dialogs.py` | Generates arithmetic dialogue in the interface language plus question tests that are *not* in the data, which measure real generalisation (`--append` adds to the loaded text) |
| `quiz.py` | Asks the current model every question test and writes `report.txt` |

## Question tests

Question tests are how Kor Studio checks whether a text model learned what it
was taught: train, then ask. Write one test per line:

```
What is the capital of Italy? => The capital of Italy is Rome.
# comments and empty lines are ignored
```

For dialogue data (lines such as `User: ...` followed by `Bot: ...`) each
question is asked in that format and the model's reply up to the end of the
line must equal the expected answer exactly. For dialogue data Kor fills the
tests with questions from the data automatically, quizzes the model while it
trains, keeps the weights that answered the most questions and can stop as
soon as all of them pass.

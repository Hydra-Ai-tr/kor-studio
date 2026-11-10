<p align="center"><img src="assets/logo.png" width="120" alt="Kor Studio logo"></p>

<h1 align="center">Kor Studio</h1>

<p align="center">
  <b>Design, train and test your own neural networks on the hardware you already have.</b><br>
  A desktop AI workshop with a training engine written from scratch in C++17. No Python stack, no GPU required.
</p>

<p align="center">
  <a href="https://github.com/Hydra-Ai-tr/kor-studio/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/Hydra-Ai-tr/kor-studio?color=e8711a"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus">
  <img alt="Platforms" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-555">
  <img alt="Dependencies" src="https://img.shields.io/badge/ML%20framework-none-success">
</p>

<p align="center">
  English · <a href="docs/README.tr.md">Türkçe</a> · <a href="docs/README.ru.md">Русский</a>
</p>

<p align="center"><img src="docs/screenshots/demo.gif" width="860" alt="Kor Studio training a small dialogue model in real time"></p>

---

## Overview

Kor Studio is a free, open-source desktop application for building small neural networks end to
end: load a dataset, let the app size a model for your machine, watch it train, quiz it, and save
it. Everything runs locally on an ordinary CPU.

The engine (`korcore`) is a self-contained C++17 library with its own tensors, layers, automatic
gradients, optimizers and a GPT-style transformer. It does not depend on PyTorch, TensorFlow or
any other ML framework, which keeps the whole program a single small executable and makes the
code a readable reference for how training actually works.

> *Kor* is Turkish for *ember*: a small glow that is enough to start a fire.

## Highlights

- **Two kinds of models.** Plain text trains a character-level transformer language model; a CSV
  table trains a classifier or a regressor. The task is detected from the data.
- **Auto-configuration.** Model size, batch size and thread count are picked from your free RAM,
  CPU cores and dataset size, with an estimate of the memory needed before you build.
- **Live training dashboard.** Loss and validation curves, accuracy, throughput, ETA,
  pause / resume / stop, early stopping, and keeping the best weights.
- **Train-then-ask question tests on unseen questions.** Write `question => expected answer`
  pairs (generated automatically for dialogue data). Test questions are removed from the
  training data, so they measure generalization, not memorization. The model is quizzed during
  training, keeps the checkpoint that answered the most questions, and can stop as soon as every
  answer is correct.
- **Training log.** Every event is shown in the app and saved to a timestamped log file.
- **Python plugins.** Drop a script into the plugins folder to clean data, generate datasets,
  write tests or examine a model. Three plugins are bundled.
- **Playground.** Generate text or run predictions, even while training continues.
- **Open model format.** Models are saved as [`.kor` files](docs/kor-format.md), optionally in
  half precision.
- **Command-line tool.** `kor-cli` trains, tests and samples models for scripting and CI.
- **Three interface languages.** English, Turkish and Russian, switchable at runtime.

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/home.png" alt="Home page with detected hardware"><br><sub><b>Home</b>: hardware detection and quick start</sub></td>
    <td width="50%"><img src="docs/screenshots/model.png" alt="Model designer"><br><sub><b>Model</b>: visual architecture and memory estimate</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/train.png" alt="Training dashboard with loss curves"><br><sub><b>Training</b>: live loss and validation curves</sub></td>
    <td><img src="docs/screenshots/tests.png" alt="Question test results"><br><sub><b>Question tests</b>: held-out questions the model never saw</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/log.png" alt="Training log"><br><sub><b>Training log</b>: shown live and saved to a file</sub></td>
    <td><img src="docs/screenshots/plugins.png" alt="Plugins page"><br><sub><b>Plugins</b>: Python scripts with live output</sub></td>
  </tr>
</table>

## Download

**Windows:** download `kor-studio-<version>-windows-x64.zip` from the
[latest release](https://github.com/Hydra-Ai-tr/kor-studio/releases/latest), extract it, and run
`kor-studio.exe`. It needs no installer or runtime; Windows 10 or newer is required. Python 3 is
only needed for plugins.

**Linux and macOS:** build from source (see [Building from source](#building-from-source)).

## Quick start

1. **Data:** choose *Use example text* (or load your own `.txt` / `.csv`).
2. **Model:** press *Auto-configure for my system*, then *Build model*.
3. **Training:** press *Start training*. Watch the curves, the log and the question tests.
4. **Playground:** type a prompt and generate text from the model.
5. **Save / Load:** store the model as a `.kor` file.

The bundled `examples/diyalog_tr.txt` is a short Turkish question-and-answer dialogue. On a
laptop CPU (Intel i5-12500H), the auto-configured 66K-parameter model trains on it in about three
minutes.

## Question tests

Loss tells you the model is learning; question tests tell you *what* it learned. A test file has
one question per line:

```text
# lines starting with # are comments
What is the capital of Italy? => The capital of Italy is Rome.
7 times 6 => 42
```

For dialogue data (pairs of lines such as `User:` / `Assistant:`), Kor Studio detects the speaker
tags, builds the prompt `User: <question>\nAssistant:` and compares the model's greedy answer with
the expected one. Tests run every few hundred steps during training; the trainer keeps the weights
that passed the most tests (or the lowest validation loss while none pass) and can stop when the
target pass rate is reached.

**Tests are always held out.** Before training starts, every line that asks a test question (and
the answer that follows it) is removed from the training and validation text, and the log reports
how many lines were removed. A passing test therefore means the model answered a question it has
never seen. On the bundled dialogue example, the 66K-parameter model reaches 92% next-character
accuracy but passes 0 of 12 held-out questions. It has learned the style of the conversation, not
arithmetic or facts it was never shown, and the test makes that visible.

## Plugins

A plugin is a Python 3 script whose first line is `# kor-plugin`:

```python
# kor-plugin
# name: Reverse lines
# description: Writes the loaded text with its lines in reverse order.
# args: --keep-empty

import kor

text = kor.read_data()
kor.write_data("\n".join(reversed(text.splitlines())))
```

Plugins receive the current dataset, model and tests through environment variables, and the
bundled `kor` helper module wraps them (`read_data`, `write_tests`, `generate`, `run_tests`, and
more). A `data.txt` or `tests.txt` written by the plugin is loaded back into the app
automatically. The full reference is in [docs/plugins.md](docs/plugins.md).

| Bundled plugin | What it does |
|---|---|
| `clean_text.py` | Normalizes whitespace and quotes and removes duplicate Q/A pairs |
| `math_dialogs.py` | Generates arithmetic dialogues plus held-out test questions |
| `quiz.py` | Asks the model every test question and writes a report |

## Command line

```bash
kor-cli sysinfo                                  # detected hardware and chosen kernels
kor-cli selftest                                 # gradient checks for every layer
kor-cli train examples/spiral.csv                # auto-configured table training
kor-cli train notes.txt --steps 2000 --out notes.kor
kor-cli train examples/diyalog_tr.txt --target 1 --log train.log --out chat.kor
kor-cli test chat.kor tests.txt --data examples/diyalog_tr.txt
kor-cli generate chat.kor "Kullanıcı: Merhaba" --len 200
```

Training options: `--steps N`, `--patience N` (early stopping), `--tests file`,
`--target 0..1` (stop at this pass rate), `--log file`, `--out file`, `--text` / `--table`.

## Building from source

Requirements: CMake 3.20+, a C++17 compiler (GCC, Clang or MSVC), Git and OpenGL 3.0.
GLFW and Dear ImGui are fetched automatically by CMake.

```bash
git clone https://github.com/Hydra-Ai-tr/kor-studio.git
cd kor-studio
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/kor-studio
```

- **Linux:** install the GLFW build dependencies first, e.g. `sudo apt install xorg-dev libgl1-mesa-dev`.
- **Engine and CLI only:** add `-DKOR_BUILD_GUI=OFF`.
- **Verify the engine:** `./build/bin/kor-cli selftest` compares every analytic gradient with
  finite differences.

## Under the hood

| Technique | Benefit |
|---|---|
| Hardware auto-tuning | Model, batch and thread settings sized to free RAM, cores and data |
| Gradient accumulation | Large effective batches with only one small batch in memory |
| LoRA fine-tuning | Adapt a trained model by training a small fraction of its weights |
| AdamW with warm-up and cosine decay | Stable training without hand-tuning |
| Runtime SIMD dispatch | AVX2/FMA kernels when available, portable code otherwise |
| Half-precision checkpoints | Saved models take half the disk space |
| Zero-dependency engine | The trainer is one small static library |

## Scope and limitations

Kor Studio trains on the **CPU** and is designed for small models: tabular networks and
character-level transformers up to a few million parameters, which train in seconds to minutes on
a laptop. It is a tool for learning, experimentation and small, focused models. It will not
train a large language model. A GPU backend is on the roadmap.

## Project layout

```text
src/core/    training engine: tensors, layers, transformer, optimizers, trainer, auto-tuner
src/app/     desktop application (Dear ImGui + GLFW), plugin runner, translations (i18n.h)
src/cli/     kor-cli command-line tool
plugins/     bundled Python plugins and the kor.py helper module
examples/    small demo datasets
docs/        file format, plugin guide, translated READMEs, screenshots
assets/      logo and application icon
```

## Roadmap

- GPU backend (Vulkan compute) and int8 quantized export
- Sub-word (BPE) tokenizer and larger text models
- Image datasets and convolutional layers
- More dataset tools: splitting, augmentation, format conversion
- More interface languages: add a column to `src/app/i18n.h`

## Contributing

Bug reports, ideas and pull requests are welcome; please open an
[issue](https://github.com/Hydra-Ai-tr/kor-studio/issues) to discuss larger changes first. Before
submitting a change to the engine, run `kor-cli selftest` and make sure it passes. New plugins and
translations are especially appreciated.

## License

Kor Studio is released under the [MIT License](LICENSE). It uses
[Dear ImGui](https://github.com/ocornut/imgui) (MIT) and [GLFW](https://www.glfw.org) (zlib).

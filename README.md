<p align="center"><img src="assets/logo.png" width="128" alt="Kor Studio logo"></p>
<h1 align="center">Kor Studio</h1>
<p align="center"><b>Train your own AI model on the hardware you already have.</b><br>
English · <a href="docs/README.tr.md">Türkçe</a> · <a href="docs/README.ru.md">Русский</a></p>

Kor Studio is a free, open-source desktop workshop for **designing, training and testing your own
neural networks**. Its training engine is written from scratch in C++17 with no ML framework
underneath, and it tunes itself to your computer, so small models train on an ordinary CPU.

*Kor* is Turkish for *ember*: a small glow that is enough to start a fire.

![Training screen](docs/screenshot-train.png)

## What it does

- **Data** – load a plain text file (trains a character-level language model) or a CSV table
  (trains a classifier or a regressor). The task is detected automatically.
- **Model** – design a network visually (layer list for tables, transformer sliders for text) or
  press *Auto-configure* and let Kor pick a size that fits your RAM, CPU and dataset.
- **Training** – live loss curves, validation, accuracy, speed, ETA, pause/resume/stop.
- **Playground** – generate text or run predictions, even while training is in progress.
- **Save / Load** – models are stored in the open [`.kor` format](docs/kor-format.md).
- **Three languages** – English, Turkish and Russian, switchable at runtime.

## Why it is light

| Technique | What it gives you |
|---|---|
| Auto-tuning | Model size, batch size and thread count are chosen from free RAM, core count and dataset size |
| Gradient accumulation | Large effective batches while only one small batch is in memory |
| LoRA fine-tuning | Adapt a trained model by training a tiny fraction of its weights |
| Half-precision checkpoints | Saved models take half the disk space |
| Runtime SIMD dispatch | AVX2/FMA kernels when the CPU has them, a portable baseline otherwise |
| Zero dependencies in the engine | The whole trainer is a small static library; the app is one executable |

### What to expect (honestly)

Kor Studio v0.1 trains on the **CPU**. It is meant for small models: tabular networks and
character-level transformers up to a few million parameters, which train in seconds to minutes on
a laptop. It will not train a ChatGPT-sized model on a laptop; nothing can. A GPU backend is on the
roadmap, and the light-training techniques above apply there too.

## Build

Requirements: CMake 3.20+, a C++17 compiler (GCC, Clang or MSVC), Git, and OpenGL 3.0.
GLFW and Dear ImGui are downloaded automatically by CMake.

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/bin/kor-studio
```

On Linux install the usual GLFW build packages first (e.g. `xorg-dev libgl1-mesa-dev` on Debian/Ubuntu).
To build only the engine and the command-line tool, add `-DKOR_BUILD_GUI=OFF`.

### Command line

```bash
kor-cli sysinfo                          # detected hardware
kor-cli selftest                         # gradient checks for every layer
kor-cli train examples/spiral.csv        # auto-configured training
kor-cli train my_text.txt --steps 2000 --out model.kor
```

## Project layout

```
src/core/   training engine: tensors, layers, transformer, optimizer, trainer, auto-tuner
src/app/    desktop application (Dear ImGui) and translations (i18n.h)
src/cli/    kor-cli
examples/   small demo datasets
assets/     logo and icon      docs/  file format, translated READMEs
```

## Roadmap

- GPU backend (Vulkan compute) and int8 quantized export
- Sub-word (BPE) tokenizer and larger text models
- Image datasets and convolutional layers
- Dataset tools: cleaning, splitting, augmentation
- More languages – add a column to `src/app/i18n.h`

## Contributing

Issues and pull requests are welcome. Run `kor-cli selftest` before sending a change to the engine:
it compares every analytic gradient against finite differences.

## License

[MIT](LICENSE). Kor Studio uses [Dear ImGui](https://github.com/ocornut/imgui) (MIT) and
[GLFW](https://www.glfw.org) (zlib).

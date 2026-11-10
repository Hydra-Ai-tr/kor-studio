# Plugins

Kor Studio plugins are Python 3 scripts built on the `kor` library. There are
two kinds:

- **Skills** run alongside training and steer it while it runs. A skill can
  lower or raise the learning rate, add training examples, keep or restore the
  best weights, stop early, sample the model, ask it questions and draw its own
  charts. Turn skills on in the **Plugins** page and they join every training
  run, or pass them to `kor-cli train --skill`.
- **Tools** run once when you press *Run*. They prepare or generate training
  data, write question tests and quiz trained models.

Plugins are read from two folders:

- `plugins/` next to the program (the bundled plugins)
- your own folder: `%APPDATA%\KorStudio\plugins` on Windows,
  `~/.config/korstudio/plugins` elsewhere (*Open my plugins folder* creates it)

Kor Studio looks for Python in the standard install folders, then for the `py`
launcher and `python` on `PATH`. The `kor` library ships with Kor Studio in the
`python/` folder; to use it in your own editor or scripts, install it with
`pip install ./python` from the source tree.

## Your first skill

```python
# kor-plugin
# type: skill
# name: Plateau breaker
# description: Halves the learning rate when the validation loss stops improving.
# args: --patience 2

import kor

skill = kor.Skill()


@skill.on_eval
def check(run, ev):
    if ev.improved:
        run.state["stale"] = 0
        return
    run.state["stale"] = run.state.get("stale", 0) + 1
    if run.state["stale"] >= run.opt("patience", 2):
        run.scale_lr(0.5)
        run.log(f"no progress, learning rate x{run.lr_factor:.2f}", "warn")
        run.state["stale"] = 0


if __name__ == "__main__":
    skill.run()
```

Save it in your plugins folder, press *Refresh list* and turn it on, or try it
from the command line:

```bash
kor-cli train examples/diyalog_tr.txt --skill "plateau.py --patience 3"
```

`python -m kor new "My skill"` writes a ready-to-edit template (the *New skill*
button does the same).

## Header

Every plugin starts with a comment block whose first line is `# kor-plugin`:

| Key | Meaning |
|---|---|
| `type` | `skill` or `tool` (default) |
| `name`, `name.tr`, `name.ru` | Title shown in the list (English is the fallback) |
| `description`, `description.tr`, `description.ru` | One-paragraph explanation |
| `args` | Default arguments (editable on the card, saved between sessions) |
| `version`, `author` | Shown on the card |
| `requires` | Python packages the plugin needs, e.g. `numpy, scikit-learn`; the installer offers to `pip install` the missing ones |
| `url` | Where the plugin lives (for updates and sharing) |

## Skills

### Events

| Decorator | Called | Useful fields of `ev` |
|---|---|---|
| `@skill.on_start` | before the first step | |
| `@skill.on_step(every=N)` | every N steps | `step`, `loss`, `lr` |
| `@skill.on_eval` | after each validation | `val_loss`, `val_acc`, `improved`, `best_val`, `best_step` |
| `@skill.on_tests` | after the question tests | `passed`, `total_tests`, `results`, `failed` |
| `@skill.on_end` | when training finishes | `state`: `done`, `stopped`, `early_stop`, `target`, `hook_stop` or `error` |

Every event also has `step`, `total`, `progress` (0..1), `elapsed` (seconds),
`loss` (smoothed training loss) and `lr`. Fields that are not known yet are
`None`. `ev.results` is a list of `(question, expected, answer, passed)`;
`ev.failed` keeps only the failures.

The trainer waits while a handler runs, so its changes apply to the very next
step. Keep handlers quick: a few milliseconds per step event, up to a few
seconds for test events. A handler that raises is reported in the log and the
skill keeps running; a skill that exits or stops answering for 5 minutes is
turned off and training continues without it.

### What a skill can do

| `run.` | Does |
|---|---|
| `log(text, tone="info")` | Writes to the training log; tones: `info`, `good`, `warn`, `bad`, `muted`. `print()` also works (muted) |
| `metric(name, value)` | Records a value; the **Skills** tab of the Training page charts it |
| `scale_lr(mult)`, `set_lr_factor(f)`, `lr_factor` | Learning rate = schedule x factor |
| `set_lr(rate)` | Fixed learning rate from now on (`0` returns to the schedule) |
| `add_text(text)`, `add_pairs(pairs)` | Text models: mix extra training text into the batches; returns the characters added |
| `mix(share)`, `clear_added()` | Share of each batch taken from the added text (`None` = automatic, at most half); forget it |
| `generate(prompt, length=200, temperature=0.8, line=False)` | Text models: continue a prompt with the current weights |
| `ask(question, expected="")` | Text models: the model's answer in the dialogue format, as `Answer(answer, passed)` |
| `save_best()`, `restore_best()` | Keep the current weights as the best ones / go back to the best ones |
| `stop(reason)` | End training after this event (the best weights are kept as usual) |

And what it can read:

| `run.` | Contents |
|---|---|
| `args`, `opt(name, default)` | Arguments; `opt` reads `--name value`, `--name=value` or `name=value` and converts to the type of `default` |
| `model` | `kind` (`text`/`table`), `parameters`, `trainable`, `vocab_size`, `context`, `width`, `layers`, `heads`, `classes` ... |
| `data` | `path`, `kind`, `train`, `validation`, `characters`, `question_tag`, `answer_tag` |
| `config` | `steps`, `batch`, `lr`, `eval_every`, `test_every`, `optimizer`, `keep_best`, `patience` |
| `tests` | The question tests as `(question, expected)` |
| `q_tag`, `a_tag`, `format_pair(q, a)`, `can_write(text)` | Dialogue format of the data; whether the model knows every character of a text |
| `lang`, `tr(en, tr, ru)` | Interface language and a helper to pick a message |
| `state` | A dict for your own values between events |

### Keep the tests honest

The question tests are removed from the training text before training starts,
so they measure answers to questions the model has never seen. A skill that
adds text must keep it that way: never add `run.tests` (or their answers) to
the training data. The bundled skills filter them out explicitly.

## Tools

A tool is a script without `type: skill`. Every run gets a fresh output folder
and these environment variables:

| Variable | Contents |
|---|---|
| `KOR_OUT` | Folder for the tool's results |
| `KOR_DATA` | Path of the loaded dataset (empty if none) |
| `KOR_MODEL` | A copy of the current model as a `.kor` file (empty if none) |
| `KOR_TESTS` | The current question tests, one `question => answer` per line |
| `KOR_CLI` | Path of `kor-cli`, used to run the model |
| `KOR_LANG` | Interface language: `en`, `tr` or `ru` |

When the tool exits with code 0, Kor Studio picks up `data.txt` (new text
dataset), `data.csv` / `data.tsv` (new table) and `tests.txt` (new question
tests) from `KOR_OUT`. Anything else stays in the output folder.

| `kor.` | Does |
|---|---|
| `read_data()` / `write_data(text, ext="txt")` | Read the loaded dataset / write a new one |
| `read_tests()` / `write_tests(pairs)` | Read / write question tests as `[(question, answer)]` |
| `generate(prompt, length, temperature, line=False)` | Continue a prompt with the current model |
| `run_tests(pairs)` | Ask the model every question; returns `TestResult(passed, question, expected, answer)` |
| `ask(question)` | The model's answer to one question |
| `tr(en, tr, ru)`, `lang()` | Pick a message in the interface language |
| `out_dir()`, `data_path()`, `model_path()`, `cli(*args)` | The values above; run `kor-cli` |

## Sharing

A plugin is a single `.py` file, so sharing it means putting the file online:
a GitHub repository, a gist or any raw link. Others install it from the
*Get and share plugins* card on the Plugins page, or with:

```bash
python -m kor install https://github.com/you/repo/blob/main/my_skill.py
python -m kor install https://gist.github.com/you/0123abcd
python -m kor install pattern-tutor          # an id from the community list
python -m kor search tutor                   # browse the community list
python -m kor list                           # installed plugins
python -m kor remove my_skill
python -m kor check my_skill.py              # header and syntax check before sharing
```

The installer accepts raw `.py` links, GitHub file pages, gists and `.zip`
archives, checks the header and the syntax, and offers to install the packages
listed in `requires`. Plugins are programs that run on your computer: install
only what you trust.

To list your plugin in the community list (shown in the app), open a pull
request that adds an entry to
[`community/index.json`](../community/index.json):

```json
{
  "id": "plateau-breaker",
  "type": "skill",
  "name": "Plateau breaker",
  "name.tr": "Plato kırıcı",
  "description": "Halves the learning rate when the validation loss stops improving.",
  "author": "your-name",
  "tags": "learning-rate",
  "url": "https://raw.githubusercontent.com/you/repo/main/plateau.py"
}
```

## Bundled plugins

| Plugin | Kind | What it does |
|---|---|---|
| `pattern_tutor.py` | skill | Learns the number patterns of dialogue data in any language ("What is 7 plus 5?" -> "12") and, when tests fail, adds fresh examples of the failing pattern with new numbers |
| `weak_spots.py` | skill | Trains more often on the examples that look most like the failed questions |
| `lr_doctor.py` | skill | Lowers the learning rate on plateaus; restores the best weights when the loss jumps; `--stop` ends training when nothing helps |
| `sample_watch.py` | skill | Logs text samples during training and charts their variety (falling variety = the model is looping) |
| `clean_text.py` | tool | Removes invisible characters, odd spaces and typographic quotes; `--dedupe` drops repeated question/answer pairs |
| `math_dialogs.py` | tool | Generates arithmetic dialogue in the interface language plus question tests that are *not* in the data (`--append` adds to the loaded text) |
| `quiz.py` | tool | Asks the current model every question test and writes `report.txt` |

On `examples/diyalog_tr.txt` (65.7K parameters, 4,901 steps, the same seed),
the default run answers **0 of 12** held-out questions. With `pattern_tutor`
and `weak_spots` turned on it answers **3 of 12**, and the validation loss on
the untouched validation text drops from 0.299 to 0.233.

## Protocol

Skills can be written in any language that reads standard input and writes
standard output; `kor.skill` implements this for Python.

Kor writes one JSON object per line:

| Message | When |
|---|---|
| `{"type": "hello", "protocol": 1, "args": [...], "lang": "en", "model": {...}, "data": {...}, "config": {...}, "tests": [...]}` | once, at start |
| `{"type": "event", "event": "start" \| "step" \| "eval" \| "tests" \| "end", "step": ..., ...}` | for every subscribed event |
| `{"type": "reply", "id": N, "ok": true, "value": ...}` | answer to a command with an `id` |
| `{"type": "bye"}` | training is over |

The skill answers with lines that start with `@kor ` followed by JSON; every
other line becomes a log line. It first sends
`{"op": "ready", "events": ["eval", "tests"], "every": 0}` (`every` = step
event period), then for each event any number of commands and finally
`{"op": "done"}`. Commands: `log {text, tone}`, `metric {name, value}`,
`lr_factor {value}`, `lr {value}`, `stop {reason}`, `save_best`,
`restore_best`, `add_text {text}`, `clear_added`, `mix {value}`,
`generate {prompt, length, temperature, line}`, `ask {question, expected}`.
Commands that return something (`restore_best`, `add_text`, `generate`, `ask`)
carry an `id` and the skill waits for the matching reply.

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

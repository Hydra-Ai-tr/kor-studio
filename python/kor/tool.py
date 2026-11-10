"""Helpers for tool plugins: scripts you run from the Plugins page.

Kor Studio runs a tool with these environment variables:

    KOR_OUT    folder for the tool's results (a new one for every run)
    KOR_DATA   path of the loaded dataset, or empty
    KOR_MODEL  a copy of the current model (.kor), or empty
    KOR_TESTS  the current question tests, one "question => answer" per line
    KOR_CLI    path of kor-cli, used here to run the model
    KOR_LANG   interface language: en, tr or ru

After a successful run (exit code 0) Kor Studio picks up:

    KOR_OUT/data.txt, data.csv or data.tsv   loaded as the new dataset
    KOR_OUT/tests.txt                        becomes the question tests

A tool can also be run by hand; then set the variables yourself (or pass
paths to the functions below) and results go to the current folder.
"""

import os
import shutil
import subprocess
from collections import namedtuple

TestResult = namedtuple("TestResult", "passed question expected answer")


def _env(name):
    return os.environ.get(name, "") or None


def out_dir():
    d = os.environ.get("KOR_OUT") or os.getcwd()
    os.makedirs(d, exist_ok=True)
    return d


def data_path():
    return _env("KOR_DATA")


def model_path():
    return _env("KOR_MODEL")


def lang():
    return os.environ.get("KOR_LANG", "en")


def tr(en, tr=None, ru=None):
    """Picks the text for the interface language (English is the fallback)."""
    return {"tr": tr, "ru": ru}.get(lang()) or en


def read_data(path=None):
    """Text of the loaded dataset (or of `path`), or None."""
    path = path or data_path()
    if not path:
        return None
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def write_data(text, ext="txt"):
    """Writes a dataset that Kor Studio loads when the tool finishes."""
    path = os.path.join(out_dir(), "data." + ext)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def parse_tests(text):
    """[(question, answer)] from "question => answer" lines."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        q, a = line.split("=>", 1)
        if q.strip() and a.strip():
            pairs.append((q.strip(), a.strip()))
    return pairs


def format_tests(pairs):
    return "".join(f"{q} => {a}\n" for q, a in pairs)


def read_tests(path=None):
    """The current question tests (or the ones in `path`)."""
    path = path or _env("KOR_TESTS")
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return parse_tests(f.read())


def write_tests(pairs):
    """Writes question tests that Kor Studio uses when the tool finishes."""
    path = os.path.join(out_dir(), "tests.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(format_tests(pairs))
    return path


def cli_path():
    """kor-cli: $KOR_CLI, next to this library's folder, or on PATH."""
    exe = _env("KOR_CLI")
    if exe:
        return exe
    name = "kor-cli.exe" if os.name == "nt" else "kor-cli"
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isfile(os.path.join(here, name)):
        return os.path.join(here, name)
    return shutil.which("kor-cli")


def cli(*args):
    """Runs kor-cli and returns its output; raises RuntimeError on failure."""
    exe = cli_path()
    if not exe:
        raise RuntimeError("kor-cli was not found (set KOR_CLI)")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([exe, *map(str, args)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", creationflags=flags)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr).strip() or "kor-cli failed")
    return r.stdout


def _model(model):
    m = model or model_path()
    if not m:
        raise RuntimeError(tr("There is no model. Build or train one first.",
                              "Model yok. Önce bir model oluştur veya eğit.",
                              "Модели нет. Сначала создайте или обучите её."))
    return m


def generate(prompt="", length=200, temperature=0.8, model=None, line=False):
    """Continues `prompt` with the model. With line=True stops at the end of the line."""
    args = ["generate", _model(model), prompt, "--len", length, "--temp", temperature]
    if line:
        args.append("--line")
    out = cli(*args)
    return out[:-1] if out.endswith("\n") else out


def run_tests(pairs, model=None, data=None):
    """Asks the model every (question, expected) pair and returns TestResult items.

    `data` is a text dataset used to detect the dialogue format
    (e.g. "User: ... / Bot: ..."); it defaults to the loaded dataset.
    """
    if not pairs:
        return []
    path = os.path.join(out_dir(), "quiz-tests.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(format_tests(pairs))
    args = ["test", _model(model), path, "--tsv"]
    data = data or data_path()
    if data and data.lower().endswith(".txt"):
        args += ["--data", data]
    results = []
    for line in cli(*args).splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            results.append(TestResult(parts[0] == "PASS", parts[1], parts[2], parts[3]))
    return results


def ask(question, model=None, data=None):
    """The model's answer to one question (in the dialogue format of the data)."""
    r = run_tests([(question, "?")], model, data)
    return r[0].answer if r else ""

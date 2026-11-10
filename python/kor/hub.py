"""Sharing plugins: install from a link, a file or the community list.

    python -m kor install <link | file | id>   add a plugin to your plugins folder
    python -m kor search [words]               the community list
    python -m kor list                         installed plugins
    python -m kor remove <file | id>           delete an installed plugin
    python -m kor new <name> [--tool]          start a new plugin from a template
    python -m kor check <file>                 validate a plugin before sharing it
    python -m kor path                         print the plugins folder

Links can be raw .py files, GitHub file pages (github.com/.../blob/...),
gists (gist.github.com/<user>/<id>) or .zip archives with plugins inside.
Plugins are Python code that runs on your computer: install only what you trust.
"""

import io
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

INDEX_URL = "https://raw.githubusercontent.com/Hydra-Ai-tr/kor-studio/main/community/index.json"
HEADER_LINES = 40
LANGS = ("", ".tr", ".ru")


# ---------------------------------------------------------------- folders

def plugins_dir():
    """The user's plugins folder (the one Kor Studio scans), created on demand."""
    d = os.environ.get("KOR_PLUGINS_DIR")
    if not d:
        if os.name == "nt":
            base = os.environ.get("APPDATA") or os.path.expanduser("~")
            d = os.path.join(base, "KorStudio", "plugins")
        else:
            base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
            d = os.path.join(base, "korstudio", "plugins")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- headers

def read_header(text):
    """The header of a plugin as a dict, or None when the text is not a plugin.

    Keys: name, description (+ .tr / .ru), args, type ("tool" or "skill"),
    requires (list), version, author, url.
    """
    info = {}
    marked = False
    for n, line in enumerate(text.splitlines()[:HEADER_LINES]):
        line = line.lstrip("\ufeff").strip()
        if not line or line.startswith("#!"):
            continue
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        if not marked:
            if body != "kor-plugin":
                return None
            marked = True
            continue
        if ":" in body:
            k, v = body.split(":", 1)
            info[k.strip()] = v.strip()
    if not marked:
        return None
    info["type"] = "skill" if info.get("type", "").lower() == "skill" else "tool"
    info["requires"] = [r for r in re.split(r"[,\s]+", info.get("requires", "")) if r]
    return info


def read_file_header(path):
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            return read_header(f.read(8192))
    except OSError:
        return None


def installed(folder=None):
    """[(file name, header)] of the plugins in the folder."""
    folder = folder or plugins_dir()
    out = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".py"):
            h = read_file_header(os.path.join(folder, f))
            if h is not None:
                out.append((f, h))
    return out


# ---------------------------------------------------------------- downloads

def _get(url, limit=5 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "kor-studio"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("the download is too large")
    return data


def _raw_url(url):
    """GitHub file pages -> raw file links."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)$", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return url


def _fetch_sources(src):
    """[(file name, bytes)] for a link or a local path."""
    if os.path.exists(src):
        if src.lower().endswith(".zip"):
            with open(src, "rb") as f:
                return _from_zip(f.read())
        with open(src, "rb") as f:
            return [(os.path.basename(src), f.read())]
    if not re.match(r"https?://", src):
        raise RuntimeError(f"not a file, link or community id: {src}")
    m = re.match(r"https?://gist\.github\.com/(?:[^/]+/)?([0-9a-fA-F]+)", src)
    if m:
        gist = json.loads(_get("https://api.github.com/gists/" + m.group(1)))
        files = []
        for name, f in (gist.get("files") or {}).items():
            if name.endswith(".py"):
                body = f.get("content")
                if f.get("truncated") or body is None:
                    body = _get(f["raw_url"]).decode("utf-8")
                files.append((name, body.encode("utf-8")))
        return files
    url = _raw_url(src)
    data = _get(url)
    path = urllib.parse.urlparse(url).path
    if path.lower().endswith(".zip") or data[:4] == b"PK\x03\x04":
        return _from_zip(data)
    name = os.path.basename(path) or "plugin.py"
    if not name.endswith(".py"):
        name += ".py"
    return [(name, data)]


def _from_zip(data):
    files = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.filename.endswith(".py") and not info.is_dir() and info.file_size < 5 * 1024 * 1024:
                files.append((os.path.basename(info.filename), z.read(info)))
    return files


def _safe_name(name):
    name = re.sub(r"[^\w.\-]+", "_", os.path.basename(name), flags=re.UNICODE)
    return name if name.endswith(".py") else name + ".py"


# ---------------------------------------------------------------- index

def community(url=None):
    """The community list: [{"id", "name", "type", "description", "author", "url", ...}]."""
    url = url or os.environ.get("KOR_INDEX") or INDEX_URL
    try:
        if os.path.exists(url):
            with open(url, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(_get(url).decode("utf-8"))
    except Exception:
        # offline: the list that shipped with Kor Studio
        fallback = os.environ.get("KOR_INDEX_FALLBACK")
        if not fallback or not os.path.exists(fallback):
            raise
        with open(fallback, encoding="utf-8") as f:
            data = json.load(f)
    return data.get("plugins", []) if isinstance(data, dict) else data


def search(words="", url=None):
    words = [w.lower() for w in words.split()]
    out = []
    for p in community(url):
        hay = " ".join(str(p.get(k, "")) for k in ("id", "name", "description", "author", "type", "tags")).lower()
        if all(w in hay for w in words):
            out.append(p)
    return out


# ---------------------------------------------------------------- install

def pip_install(packages, quiet=False):
    """Installs Python packages for the current interpreter (user site)."""
    if not packages:
        return True
    cmd = [sys.executable, "-m", "pip", "install", "--user", "--disable-pip-version-check", *packages]
    if quiet:
        cmd.insert(4, "-q")
    return subprocess.call(cmd) == 0


def missing_packages(requires):
    """The entries of a requires list whose module cannot be imported."""
    import importlib.util
    out = []
    for r in requires:
        mod = re.split(r"[<>=!~\[; ]", r, 1)[0].replace("-", "_")
        alias = {"scikit_learn": "sklearn", "pillow": "PIL", "pyyaml": "yaml", "opencv_python": "cv2"}
        mod = alias.get(mod.lower(), mod)
        if importlib.util.find_spec(mod) is None:
            out.append(r)
    return out


def install(src, folder=None, deps=True, yes=False, index=None, log=print):
    """Installs the plugins found at `src`; returns the installed file paths."""
    folder = folder or plugins_dir()
    if not os.path.exists(src) and not re.match(r"https?://", src):
        entry = next((p for p in community(index) if p.get("id") == src or p.get("name") == src), None)
        if not entry:
            raise RuntimeError(f"\"{src}\" is not in the community list")
        src = entry["url"]
    files = _fetch_sources(src)
    done = []
    for name, data in files:
        text = data.decode("utf-8-sig", errors="replace")
        h = read_header(text)
        if h is None:
            log(f"skipped {name}: not a Kor plugin (no '# kor-plugin' header)")
            continue
        try:
            compile(text, name, "exec")
        except SyntaxError as e:
            log(f"skipped {name}: syntax error on line {e.lineno}")
            continue
        path = os.path.join(folder, _safe_name(name))
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        log(f"installed {h.get('name') or name} ({h['type']}) -> {path}")
        done.append(path)
        need = missing_packages(h["requires"]) if deps else []
        if need:
            if yes or _confirm(f"{name} needs: {' '.join(need)}. Install with pip?"):
                if not pip_install(need):
                    log("pip could not install: " + " ".join(need))
            else:
                log("not installed: " + " ".join(need))
    if not done:
        raise RuntimeError("no Kor plugin was found at " + src)
    return done


def _confirm(question):
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes", "e", "evet", "д", "да")
    except EOFError:
        return False


def remove(name, folder=None):
    folder = folder or plugins_dir()
    for f, h in installed(folder):
        if name in (f, f[:-3], h.get("name")):
            os.remove(os.path.join(folder, f))
            return os.path.join(folder, f)
    raise RuntimeError(f"\"{name}\" is not installed in {folder}")


def check(path):
    """Problems that would stop a plugin from working (empty list = fine)."""
    problems = []
    h = read_file_header(path)
    if h is None:
        return ["the first comment lines must start with '# kor-plugin'"]
    if not h.get("name"):
        problems.append("add a '# name:' line")
    if not h.get("description"):
        problems.append("add a '# description:' line")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            py_compile.compile(path, cfile=os.path.join(tmp, "x.pyc"), doraise=True)
    except py_compile.PyCompileError as e:
        problems.append("syntax error: " + str(e.msg).strip())
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        body = f.read()
    if h["type"] == "skill" and ".run()" not in body:
        problems.append("a skill must call skill.run() at the end")
    return problems


TEMPLATE_SKILL = '''# kor-plugin
# type: skill
# name: {title}
# description: Describe what your skill does during training.
# version: 0.1
# author: {author}

import kor

skill = kor.Skill()


@skill.on_start
def start(run, ev):
    run.log(run.tr("{title} is watching the training",
                   "{title} eğitimi izliyor",
                   "{title} следит за обучением"))


@skill.on_eval
def evaluate(run, ev):
    # ev.val_loss, ev.improved, ev.step ... see docs/plugins.md
    run.metric("val loss x100", ev.val_loss * 100)


@skill.on_tests
def tests(run, ev):
    for r in ev.failed[:3]:
        run.log(f"{{r.question}} -> {{r.answer}} (expected {{r.expected}})", "warn")


if __name__ == "__main__":
    skill.run()
'''

TEMPLATE_TOOL = '''# kor-plugin
# name: {title}
# description: Describe what your tool does.
# args:
# version: 0.1
# author: {author}

import kor

text = kor.read_data()
if text is None:
    raise SystemExit(kor.tr("Load a text dataset first.", "Önce bir metin veri seti yükle.",
                            "Сначала загрузите текстовые данные."))

kor.write_data(text)
print(kor.tr("Done", "Bitti", "Готово"))
'''


def new(name, tool=False, folder=None):
    folder = folder or plugins_dir()
    base = re.sub(r"\s+", "_", name.strip()).lower()
    path = os.path.join(folder, _safe_name(base))
    title = name.strip().replace("_", " ").title() if name.islower() else name.strip()
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, _safe_name(f"{base}_{n}"))
    if n > 1:
        title = f"{title} {n}"
    author = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write((TEMPLATE_TOOL if tool else TEMPLATE_SKILL).format(title=title, author=author))
    return path


# ---------------------------------------------------------------- command line

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = {a for a in argv if a.startswith("--") and "=" not in a}
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    words = [a for a in argv if not a.startswith("--")]
    # "--dir X" and "--index X" may also be written with a space
    for key in ("dir", "index"):
        if "--" + key in argv:
            i = argv.index("--" + key)
            if i + 1 < len(argv):
                opts[key] = argv[i + 1]
                if argv[i + 1] in words:
                    words.remove(argv[i + 1])
    folder = opts.get("dir")
    if folder:
        os.makedirs(folder, exist_ok=True)
    as_json = "--json" in flags
    cmd = words[0] if words else "help"
    rest = words[1:]
    try:
        if cmd == "install" and rest:
            for src in rest:
                install(src, folder, deps="--no-deps" not in flags, yes="--yes" in flags, index=opts.get("index"))
        elif cmd == "search":
            found = search(" ".join(rest), opts.get("index"))
            if as_json:
                print(json.dumps(found, ensure_ascii=False))
            elif not found:
                print("nothing found")
            for p in [] if as_json else found:
                print(f"{p.get('id', ''):<20} {p.get('type', 'tool'):<6} {p.get('name', '')} - {p.get('description', '')}")
        elif cmd == "list":
            items = installed(folder)
            if as_json:
                print(json.dumps([dict(h, file=f) for f, h in items], ensure_ascii=False))
            else:
                print(folder or plugins_dir())
                for f, h in items:
                    print(f"  {f:<24} {h['type']:<6} {h.get('name', '')}")
        elif cmd == "remove" and rest:
            print("removed " + remove(rest[0], folder))
        elif cmd == "new" and rest:
            print(new(" ".join(rest), tool="--tool" in flags, folder=folder))
        elif cmd == "check" and rest:
            bad = check(rest[0])
            for p in bad:
                print("problem: " + p)
            print("ok" if not bad else f"{len(bad)} problem(s)")
            return 1 if bad else 0
        elif cmd == "path":
            print(folder or plugins_dir())
        else:
            print(__doc__.strip())
            return 0 if cmd == "help" else 2
    except Exception as e:  # report every failure as one line
        print("error: " + str(e), file=sys.stderr)
        return 1
    return 0

"""Training skills: Python code that runs next to the trainer and steers it.

A skill is a normal Python file with a Kor plugin header::

    # kor-plugin
    # type: skill
    # name: My skill
    # description: What it does.

    import kor

    skill = kor.Skill()

    @skill.on_eval
    def check(run, ev):
        if ev.val_loss > 3:
            run.scale_lr(0.5)
            run.log("Loss is high, halving the learning rate", "warn")

    if __name__ == "__main__":
        skill.run()

Handlers receive the `Run` (what the skill can do) and an `Event` (what just
happened). Everything a handler does takes effect before the next training
step: the trainer waits until the handler returns.

Events
    on_start         training is about to begin
    on_step(every=N) every N steps
    on_eval          after each validation (ev.val_loss, ev.val_acc, ev.improved)
    on_tests         after the question tests (ev.passed, ev.results)
    on_end           training finished (ev.state: done, stopped, early_stop,
                     target, hook_stop or error)
"""

import json
import math
import os
import sys
import threading
import traceback
from collections import namedtuple

__all__ = ["Skill", "Run", "Event", "Answer", "TestCase", "SkillError"]

PREFIX = "@kor "
EVENTS = ("start", "step", "eval", "tests", "end")

Answer = namedtuple("Answer", "answer passed")
TestCase = namedtuple("TestCase", "question expected")
TestRun = namedtuple("TestRun", "question expected answer passed")


class SkillError(RuntimeError):
    """Kor refused a command (for example an unknown one)."""


class _Bye(Exception):
    pass


class Info(dict):
    """A dict whose keys can also be read as attributes (missing ones give None)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.get(name)


class Event(Info):
    """What just happened. Common fields:

    kind, step, total, elapsed (seconds), loss (smoothed training loss), lr,
    val_loss, val_acc, best_val, best_step (None when not known yet),
    improved (eval), passed / total_tests / results (tests), state (end).
    """

    @property
    def progress(self):
        """Share of the run that is done, 0..1."""
        return self.step / self.total if self.total else 0.0

    @property
    def failed(self):
        """tests: the test runs that did not pass."""
        return [r for r in (self.results or []) if not r.passed]


# ---------------------------------------------------------------- output

class _Channel:
    """Writes protocol lines; user output is kept on lines of its own."""

    def __init__(self, raw):
        self.raw = raw
        self.lock = threading.RLock()
        self.partial = ""

    def user(self, text):
        with self.lock:
            text = self.partial + text
            lines = text.split("\n")
            self.partial = lines.pop()
            for line in lines:
                self._put(line)

    def send(self, obj):
        with self.lock:
            if self.partial:
                self._put(self.partial)
                self.partial = ""
            self._put(PREFIX + json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_jsonable))

    def finish(self):
        with self.lock:
            if self.partial:
                self._put(self.partial)
                self.partial = ""

    def _put(self, line):
        try:
            self.raw.write(line + "\n")
            self.raw.flush()
        except (OSError, ValueError):
            pass


def _jsonable(x):
    if hasattr(x, "item"):  # numpy numbers
        return x.item()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def _finite(v, name):
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {v!r}")
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


class _Stream:
    """Stands in for sys.stdout / sys.stderr while a skill runs."""

    def __init__(self, channel):
        self.channel = channel
        self.encoding = "utf-8"

    def write(self, s):
        self.channel.user(str(s))
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


# ---------------------------------------------------------------- run

class Run:
    """What a skill can see and do during training.

    Read-only information
        args      command-line arguments given to the skill (list of str)
        model     model info: kind ("text"/"table"), parameters, trainable,
                  vocab_size, context, width, layers, heads, classes ...
        data      dataset info: path, kind, train, validation, characters,
                  question_tag, answer_tag
        config    training settings: steps, batch, lr, eval_every, test_every,
                  optimizer, keep_best, patience ...
        tests     the question tests (list of TestCase)
        lang      interface language: "en", "tr" or "ru"
        state     a dict you can use to keep your own values between events
        event     the latest Event
    """

    def __init__(self, hello, channel, stdin):
        self._ch = channel
        self._in = stdin
        self._next_id = 1
        self._lr_factor = 1.0
        self.hello = hello
        self.args = list(hello.get("args") or [])
        self.lang = hello.get("lang") or "en"
        self.model = Info(hello.get("model") or {})
        self.data = Info(hello.get("data") or {})
        self.config = Info(hello.get("config") or {})
        self.tests = [TestCase(t.get("question", ""), t.get("expected", "")) for t in hello.get("tests") or []]
        self.state = {}
        self.event = Event()
        self._chars = set(self.data.get("characters") or "")

    # ------------------------------------------------------------ helpers

    def tr(self, en, tr=None, ru=None):
        """Picks the text for the interface language (English is the fallback)."""
        return {"tr": tr, "ru": ru}.get(self.lang) or en

    def opt(self, name, default=None):
        """Reads an option from the arguments: --name value, --name=value or name=value.

        The value is converted to the type of `default`; a bare --name gives True.
        """
        keys = ("--" + name, name)
        value = None
        found = False
        for i, a in enumerate(self.args):
            for k in keys:
                if a == "--" + name:
                    nxt = self.args[i + 1] if i + 1 < len(self.args) else None
                    value, found = (nxt if nxt is not None and not nxt.startswith("--") else True), True
                elif a.startswith(k + "="):
                    value, found = a[len(k) + 1:], True
        if not found:
            return default
        if isinstance(default, bool):
            return value is True or str(value).lower() in ("1", "true", "yes", "on")
        if default is not None and value is not True:
            try:
                return type(default)(value)
            except (TypeError, ValueError):
                return default
        return value

    @property
    def is_text(self):
        return self.model.get("kind") == "text"

    @property
    def q_tag(self):
        return self.data.get("question_tag") or ""

    @property
    def a_tag(self):
        return self.data.get("answer_tag") or ""

    def can_write(self, text):
        """True when the model knows every character of `text` (text models)."""
        return not self._chars or all(c in self._chars or c == "\n" for c in text)

    def format_pair(self, question, answer):
        """A question and its answer in the dialogue format of the data."""
        if self.q_tag or self.a_tag:
            return f"{self.q_tag}{question}\n{self.a_tag}{answer}\n"
        return f"{question}\n{answer}\n"

    # ------------------------------------------------------------ protocol

    def _send(self, op, **fields):
        fields["op"] = op
        self._ch.send(fields)

    def _call(self, op, **fields):
        rid = self._next_id
        self._next_id += 1
        fields["id"] = rid
        self._send(op, **fields)
        while True:
            line = self._in.readline()
            if not line:
                raise _Bye()
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("type") == "bye":
                raise _Bye()
            if msg.get("type") == "reply" and msg.get("id") == rid:
                if not msg.get("ok", True):
                    raise SkillError(msg.get("error") or op + " failed")
                return msg.get("value")

    # ------------------------------------------------------------ actions

    def log(self, text, tone="info"):
        """Writes a line to the training log. tone: info, good, warn, bad or muted."""
        self._send("log", text=str(text), tone=tone)

    def metric(self, name, value):
        """Records a value; Kor Studio draws it as a chart next to the loss."""
        self._send("metric", name=str(name), value=_finite(value, "metric value"))

    def set_lr_factor(self, factor):
        """Learning rate = scheduled rate * factor (1 = unchanged)."""
        self._lr_factor = _finite(factor, "factor")
        if self._lr_factor <= 0:
            raise ValueError("factor must be > 0")
        self._send("lr_factor", value=self._lr_factor)

    def scale_lr(self, mult):
        """Multiplies the current learning-rate factor (0.5 halves the rate)."""
        self.set_lr_factor(self._lr_factor * _finite(mult, "mult"))
        return self._lr_factor

    @property
    def lr_factor(self):
        return self._lr_factor

    def set_lr(self, lr):
        """Uses a fixed learning rate from now on (0 goes back to the schedule)."""
        self._send("lr", value=max(0.0, _finite(lr, "lr")))

    def stop(self, reason=""):
        """Ends training after this event (the best weights are kept as usual)."""
        self._send("stop", reason=str(reason))

    def save_best(self):
        """Keeps the current weights as the best ones."""
        self._send("save_best")

    def restore_best(self):
        """Goes back to the best weights kept so far; False if there are none."""
        return bool(self._call("restore_best"))

    def add_text(self, text):
        """Text models: adds training text from now on. Characters the model does
        not know are dropped. Returns how many characters were added."""
        return int(self._call("add_text", text=str(text)) or 0)

    def add_pairs(self, pairs):
        """Adds (question, answer) pairs in the dialogue format of the data."""
        return self.add_text("".join(self.format_pair(q, a) for q, a in pairs))

    def clear_added(self):
        """Forgets all text added by skills."""
        self._send("clear_added")

    def mix(self, share):
        """Share of each batch taken from the added text, 0..1 (None = automatic)."""
        self._send("mix", value=-1.0 if share is None else min(1.0, max(0.0, _finite(share, "share"))))

    def set_sampling(self, align=None, question_weight=None):
        """Text models: how training rows are drawn from now on.

        align: share of rows that start at the beginning of an example (a
        question of dialogue data), 0..1. question_weight: how much the question
        text counts in the loss compared with the answers, 0..1. None keeps the
        current value."""
        a = -1.0 if align is None else min(1.0, max(0.0, _finite(align, "align")))
        q = -1.0 if question_weight is None else min(1.0, max(0.0, _finite(question_weight, "question_weight")))
        self._send("sampling", align=a, question_weight=q)

    def set_dropout(self, p):
        """Text models: dropout of the transformer blocks from now on (0 = off)."""
        self._send("dropout", value=min(0.9, max(0.0, _finite(p, "p"))))

    def generate(self, prompt="", length=200, temperature=0.8, line=False):
        """Text models: continues `prompt` with the current weights. With line=True
        stops at the end of the line."""
        return self._call("generate", prompt=str(prompt), length=int(length),
                          temperature=_finite(temperature, "temperature"), line=bool(line)) or ""

    def ask(self, question, expected=""):
        """Text models: the model's answer to `question` (greedy, in the dialogue
        format). Returns Answer(answer, passed)."""
        v = self._call("ask", question=str(question), expected=str(expected)) or {}
        return Answer(v.get("answer", ""), bool(v.get("passed")))


# ---------------------------------------------------------------- skill

class Skill:
    """Collects handlers and runs the skill. See the module documentation."""

    def __init__(self, name=None):
        self.name = name
        self._handlers = {e: [] for e in EVENTS}

    def _add(self, event, fn, every=0):
        self._handlers[event].append((fn, every))
        return fn

    def on_start(self, fn):
        return self._add("start", fn)

    def on_eval(self, fn):
        return self._add("eval", fn)

    def on_tests(self, fn):
        return self._add("tests", fn)

    def on_end(self, fn):
        return self._add("end", fn)

    def on_step(self, fn=None, every=10):
        """@skill.on_step or @skill.on_step(every=25)."""
        if fn is None:
            return lambda f: self._add("step", f, max(1, int(every)))
        return self._add("step", fn, max(1, int(every)))

    def _every(self):
        steps = [e for _, e in self._handlers["step"]]
        if not steps:
            return 0
        g = steps[0]
        for e in steps[1:]:
            g = math.gcd(g, e)
        return g

    def run(self):
        """Talks to Kor until training ends. Call it at the bottom of the file."""
        if os.environ.get("KOR_SKILL") != "1":
            sys.stderr.write(
                "This is a Kor Studio training skill. Turn it on in the Plugins page of\n"
                "Kor Studio, or train with: kor-cli train data.txt --skill " +
                os.path.basename(sys.argv[0]) + "\n")
            sys.exit(2)
        stdin = sys.stdin
        channel = _Channel(sys.__stdout__)
        sys.stdout = _Stream(channel)
        sys.stderr = _Stream(channel)
        try:
            self._loop(stdin, channel)
        finally:
            channel.finish()
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

    def _loop(self, stdin, channel):
        line = stdin.readline()
        if not line:
            return
        hello = json.loads(line)
        run = Run(hello, channel, stdin)
        wanted = [e for e in EVENTS if self._handlers[e]]
        channel.send({"op": "ready", "events": wanted, "every": self._every()})
        errors = 0
        while True:
            line = stdin.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            kind = msg.get("type")
            if kind == "bye":
                return
            if kind != "event":
                continue
            ev = Event(msg)
            ev["kind"] = msg.get("event")
            ev["results"] = [TestRun(r.get("question", ""), r.get("expected", ""), r.get("answer", ""),
                                     bool(r.get("passed"))) for r in msg.get("results") or []]
            run.event = ev
            try:
                for fn, every in self._handlers.get(ev.kind, []):
                    if ev.kind == "step" and every and ev.step % every:
                        continue
                    fn(run, ev)
            except _Bye:
                return
            except Exception as e:  # keep training going; report the problem
                errors += 1
                tb = traceback.format_exc().rstrip().splitlines()
                run.log(f"{type(e).__name__}: {e}", "bad")
                for t in tb[-6:-1]:
                    channel.user(t + "\n")
                if errors >= 20:
                    run.log("too many errors, the skill stops", "bad")
                    channel.send({"op": "done"})
                    return
            channel.send({"op": "done"})

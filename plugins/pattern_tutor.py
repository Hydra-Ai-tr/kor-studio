# kor-plugin
# type: skill
# name: Pattern tutor
# name.tr: Kalıp öğretmeni
# name.ru: Репетитор по шаблонам
# description: Learns the number patterns of dialogue data (for example "What is 7 plus 5?" -> "12") in any language. When a question test fails, it writes fresh examples of the same pattern with new numbers and adds them to the training text - never the test questions themselves.
# description.tr: Diyalog verisindeki sayı kalıplarını ("7 artı 5 kaç eder?" -> "12" gibi) her dilde kendisi öğrenir. Bir soru testi başarısız olunca aynı kalıptan yeni sayılarla taze örnekler yazıp eğitim metnine ekler - test sorularının kendisini asla eklemez.
# description.ru: Сам находит числовые шаблоны в диалогах (например «Сколько будет 7 плюс 5?» -> «12») на любом языке. Когда проверочный вопрос не проходит, пишет новые примеры того же шаблона с другими числами и добавляет их в обучающий текст — сами проверочные вопросы никогда.
# args: --per-fail 40 --start 600
# version: 1.0
# author: Kor Studio

import random
import re

import kor

skill = kor.Skill()
NUM = re.compile(r"\d+")

OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b if b and a % b == 0 else None,
    "max": max,
    "min": min,
}


def read_pairs(path, q_tag, a_tag):
    """(question, answer) pairs from "q_tag ... / a_tag ..." dialogue text."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    pairs = []
    for q, a in zip(lines, lines[1:]):
        if q.startswith(q_tag) and a.startswith(a_tag):
            pairs.append((q[len(q_tag):].strip(), a[len(a_tag):].strip()))
    return pairs


def shape(question, answer=None):
    """The question with its numbers replaced, the numbers, and the answer template."""
    qn = [int(x) for x in NUM.findall(question)]
    q_tpl = NUM.sub("{}", question)
    if answer is None:
        return q_tpl, qn, None, None
    an = [int(x) for x in NUM.findall(answer)]
    parts, result = [], None
    for j, v in enumerate(an):
        if j < len(qn) and v == qn[j]:
            parts.append("{q%d}" % j)
        elif result is None:
            parts.append("{r}")
            result = v
        else:
            return q_tpl, qn, None, None
    if result is None:
        return q_tpl, qn, None, None
    pieces = NUM.split(answer)
    a_tpl = pieces[0] + "".join(p + s for p, s in zip(parts, pieces[1:]))
    return q_tpl, qn, a_tpl, result


class Pattern:
    def __init__(self, q_tpl, a_tpl):
        self.q_tpl, self.a_tpl = q_tpl, a_tpl
        self.examples = []
        self.op = None

    def fit(self):
        for name, fn in OPS.items():
            if all(fn(a, b) == r for (a, b), r in self.examples):
                self.op = name
                break
        xs = [a for (a, _), _ in self.examples]
        ys = [b for (_, b), _ in self.examples]
        self.lo, self.hi = min(xs + ys), max(xs + ys)
        self.a_ge_b = all(a >= b for (a, b), _ in self.examples)
        return self.op is not None

    def make(self, rng):
        for _ in range(50):
            a, b = rng.randint(self.lo, self.hi), rng.randint(self.lo, self.hi)
            if self.a_ge_b and a < b:
                a, b = b, a
            r = OPS[self.op](a, b)
            if r is None or r < 0:
                continue
            q = self.q_tpl.format(a, b)
            ans = self.a_tpl.replace("{q0}", str(a)).replace("{q1}", str(b)).replace("{r}", str(r))
            return q, ans
        return None


@skill.on_start
def start(run, ev):
    run.state["patterns"] = {}
    if not run.is_text or not (run.q_tag and run.a_tag) or not run.data.path:
        run.log(run.tr("Needs dialogue text data (e.g. User: / Bot: lines); staying idle.",
                       "Diyalog biçiminde metin verisi gerekir (Kullanıcı: / Asistan: gibi); boşta bekliyorum.",
                       "Нужны текстовые диалоги (строки вида User: / Bot:); жду без действий."), "warn")
        return
    groups = {}
    for q, a in read_pairs(run.data.path, run.q_tag, run.a_tag):
        q_tpl, qn, a_tpl, r = shape(q, a)
        if a_tpl is None or len(qn) != 2:
            continue
        p = groups.setdefault((q_tpl, a_tpl), Pattern(q_tpl, a_tpl))
        p.examples.append(((qn[0], qn[1]), r))
    patterns = {}
    for (q_tpl, _), p in groups.items():
        if len(p.examples) >= 5 and p.fit():
            best = patterns.get(q_tpl)
            if best is None or len(p.examples) > len(best.examples):
                patterns[q_tpl] = p
    run.state["patterns"] = patterns
    run.state["tests"] = {t.question.strip().lower() for t in run.tests}
    run.state["rng"] = random.Random(1)
    run.state["added"] = 0
    if not patterns:
        run.log(run.tr("No number patterns found in the data; staying idle.",
                       "Veride sayı kalıbı bulunamadı; boşta bekliyorum.",
                       "Числовых шаблонов в данных нет; жду без действий."), "warn")
        return
    names = ", ".join(f"{p.q_tpl.format('a', 'b')} ({p.op})" for p in patterns.values())
    run.log(run.tr(f"Learned {len(patterns)} patterns: {names}",
                   f"{len(patterns)} kalıp öğrenildi: {names}",
                   f"Найдено шаблонов: {len(patterns)}: {names}"), "good")
    warm = run.opt("start", 600)
    if warm > 0:
        teach(run, {q: warm for q in patterns})


def teach(run, counts):
    """Adds `counts[q_tpl]` fresh examples for every pattern; returns how many were added."""
    rng, tests, patterns = run.state["rng"], run.state["tests"], run.state["patterns"]
    pairs = []
    for q_tpl, n in counts.items():
        p = patterns[q_tpl]
        made = 0
        for _ in range(n * 4):
            if made >= n:
                break
            qa = p.make(rng)
            if qa and qa[0].strip().lower() not in tests:
                pairs.append(qa)
                made += 1
    if pairs:
        run.add_pairs(pairs)
        run.state["added"] += len(pairs)
        run.metric(run.tr("examples added", "eklenen örnek", "добавлено примеров"), run.state["added"])
    return len(pairs)


@skill.on_tests
def tests(run, ev):
    patterns = run.state.get("patterns")
    if not patterns or ev.kind == "tests" and ev.step >= ev.total:
        return
    per_fail = run.opt("per-fail", 40)
    counts = {}
    for r in ev.failed:
        q_tpl = shape(r.question)[0]
        if q_tpl in patterns:
            counts[q_tpl] = counts.get(q_tpl, 0) + per_fail
    if not counts:
        return
    n = teach(run, counts)
    fails = sum(c // per_fail for c in counts.values())
    run.log(run.tr(f"{fails} pattern questions failed: added {n} fresh examples",
                   f"{fails} kalıp sorusu başarısız: {n} taze örnek eklendi",
                   f"Не прошло вопросов по шаблонам: {fails}; добавлено новых примеров: {n}"))


if __name__ == "__main__":
    skill.run()

# kor-plugin
# type: skill
# name: Weak-spot coach
# name.tr: Zayıf nokta koçu
# name.ru: Тренер слабых мест
# description: After every round of question tests it finds the training examples that look most like the failed questions and trains on them more often. The test questions themselves are never added, so the tests stay honest.
# description.tr: Her soru testi turundan sonra başarısız sorulara en çok benzeyen eğitim örneklerini bulur ve onlarla daha sık eğitir. Test sorularının kendisi asla eklenmez; testler dürüst kalır.
# description.ru: После каждой проверки находит обучающие примеры, больше всего похожие на непройденные вопросы, и тренирует на них чаще. Сами проверочные вопросы не добавляются, поэтому проверка остаётся честной.
# args: --similar 3 --repeat 4
# version: 1.0
# author: Kor Studio

import re

import kor

skill = kor.Skill()
WORD = re.compile(r"\w+", re.UNICODE)


def words(text):
    return {w for w in WORD.findall(text.lower()) if not w.isdigit()}


def read_pairs(path, q_tag, a_tag):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    return [(q[len(q_tag):].strip(), a[len(a_tag):].strip())
            for q, a in zip(lines, lines[1:]) if q.startswith(q_tag) and a.startswith(a_tag)]


@skill.on_start
def start(run, ev):
    run.state["pairs"] = []
    if not run.is_text or not (run.q_tag and run.a_tag) or not run.data.path:
        run.log(run.tr("Needs dialogue text data; staying idle.",
                       "Diyalog biçiminde metin verisi gerekir; boşta bekliyorum.",
                       "Нужны текстовые диалоги; жду без действий."), "warn")
        return
    tests = {t.question.strip().lower() for t in run.tests}
    seen, pairs = set(), []
    for q, a in read_pairs(run.data.path, run.q_tag, run.a_tag):
        key = q.lower()
        if key in tests or key in seen:
            continue
        seen.add(key)
        pairs.append((q, a, words(q)))
    run.state["pairs"] = pairs
    run.state["boosted"] = 0
    run.log(run.tr(f"{len(pairs)} distinct training questions indexed",
                   f"{len(pairs)} farklı eğitim sorusu dizinlendi",
                   f"Проиндексировано разных вопросов: {len(pairs)}"))


@skill.on_tests
def tests(run, ev):
    pairs = run.state.get("pairs")
    if not pairs or ev.step >= ev.total or not ev.failed:
        return
    k, repeat = run.opt("similar", 3), run.opt("repeat", 4)
    chosen = []
    for r in ev.failed:
        target = words(r.question)
        if not target:
            continue
        scored = sorted(pairs, key=lambda p: len(target & p[2]) / len(target | p[2]), reverse=True)
        for q, a, w in scored[:k]:
            if target & w:
                chosen.append((q, a))
    if not chosen:
        return
    run.add_pairs(chosen * repeat)
    run.state["boosted"] += len(chosen)
    run.metric(run.tr("examples boosted", "güçlendirilen örnek", "усилено примеров"), run.state["boosted"])
    sample = chosen[0][0]
    run.log(run.tr(f"{len(ev.failed)} tests failed: training more on {len(chosen)} similar examples (e.g. \"{sample}\")",
                   f"{len(ev.failed)} test başarısız: benzer {len(chosen)} örnekle daha çok eğitiliyor (ör. \"{sample}\")",
                   f"Не прошло проверок: {len(ev.failed)}; больше тренирую на {len(chosen)} похожих примерах (напр. «{sample}»)"))


if __name__ == "__main__":
    skill.run()

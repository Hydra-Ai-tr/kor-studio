# kor-plugin
# type: skill
# name: Sample watcher
# name.tr: Örnek izleyici
# name.ru: Наблюдатель за текстом
# description: Every few steps lets the model write a little text and puts it in the training log, together with a "variety" chart (share of distinct 3-letter pieces): a falling value means the model is getting stuck in loops.
# description.tr: Birkaç adımda bir modele biraz metin yazdırıp eğitim günlüğüne koyar; yanında bir "çeşitlilik" grafiği çizer (farklı 3 harflik parçaların oranı): değer düşüyorsa model döngüye takılıyor demektir.
# description.ru: Каждые несколько шагов даёт модели написать немного текста и пишет его в журнал обучения, а также рисует график «разнообразия» (доля разных трёхбуквенных кусочков): падение значит, что модель зацикливается.
# args: --every 0 --length 120
# version: 1.0
# author: Kor Studio

import kor

skill = kor.Skill()


def variety(text, n=3):
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


@skill.on_step(every=10)
def step(run, ev):
    every = run.opt("every", 0) or max(10, (ev.total // 8) // 10 * 10)
    if ev.step % every or not run.is_text:
        return
    prompt = run.q_tag if run.q_tag else ""
    text = run.generate(prompt, length=run.opt("length", 120), temperature=0.8)
    shown = (prompt + text).strip().replace("\n", " / ")
    run.log(f"{ev.step}: {shown[:200]}", "muted")
    run.metric(run.tr("variety %", "çeşitlilik %", "разнообразие %"), 100 * variety(text))


@skill.on_tests
def tests(run, ev):
    run.state["failed"] = ev.failed


@skill.on_end
def end(run, ev):
    for r in run.state.get("failed", [])[:3]:
        run.log(run.tr(f"still wrong: {r.question} -> {r.answer} (expected {r.expected})",
                       f"hâlâ yanlış: {r.question} -> {r.answer} (beklenen {r.expected})",
                       f"всё ещё неверно: {r.question} -> {r.answer} (ожидалось {r.expected})"), "warn")


if __name__ == "__main__":
    skill.run()

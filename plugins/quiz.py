# kor-plugin
# name: Quiz the model
# name.tr: Modeli sına
# name.ru: Экзамен для модели
# description: Asks the current text model every question test (or the tests in a file given as the argument), prints each answer and writes a report. Also shows a free-running sample of the model's text.
# description.tr: Mevcut metin modeline her soru testini (veya argüman olarak verilen dosyadaki testleri) sorar, cevapları yazdırır ve bir rapor kaydeder. Ayrıca modelin serbest ürettiği bir metin örneğini gösterir.
# description.ru: Задаёт текущей текстовой модели все проверочные вопросы (или из файла, указанного аргументом), печатает ответы и сохраняет отчёт. Также показывает образец свободно сгенерированного текста.
# args:

import os
import sys

import kor


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    pairs = kor.read_tests(files[0]) if files else kor.read_tests()
    if not pairs:
        print(kor.tr("No question tests. Write some on Training > Question tests.",
                     "Soru testi yok. Eğitim > Soru testleri bölümünde yaz.",
                     "Нет проверочных вопросов. Добавьте их в Обучение > Проверка вопросами."))
        return 1
    try:
        results = kor.run_tests(pairs)
    except RuntimeError as e:
        print(e)
        return 1

    lines = []
    for i, r in enumerate(results, 1):
        mark = "OK  " if r.passed else "FAIL"
        lines.append(f"{i:3}. {mark} {r.question}")
        lines.append(f"          {kor.tr('answer', 'cevap', 'ответ')}:   {r.answer or '-'}")
        if not r.passed:
            lines.append(f"          {kor.tr('expected', 'beklenen', 'ожидалось')}: {r.expected}")
    passed = sum(r.passed for r in results)
    lines.append("")
    lines.append(kor.tr("Passed", "Geçen", "Пройдено") + f": {passed}/{len(results)} ({100 * passed // max(1, len(results))}%)")
    print("\n".join(lines))

    try:
        sample = kor.generate("", length=240, temperature=0.8)
        print("\n--- " + kor.tr("sample", "örnek", "образец") + " ---\n" + sample)
    except RuntimeError:
        pass

    report = os.path.join(kor.out_dir(), "report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + kor.tr("Report", "Rapor", "Отчёт") + f": {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# kor-plugin
# name: Math dialogue generator
# name.tr: Matematik diyaloğu üret
# name.ru: Генератор диалогов по арифметике
# description: Writes question/answer dialogue about adding, subtracting and multiplying small numbers in the interface language, and question tests made of questions that are NOT in the data, so they measure whether the model really learned arithmetic. --append adds the dialogue to the loaded text.
# description.tr: Arayüz dilinde küçük sayılarla toplama, çıkarma ve çarpma üzerine soru/cevap diyaloğu yazar. Soru testlerini veride OLMAYAN sorulardan seçer; böylece model aritmetiği gerçekten öğrenmiş mi ölçülür. --append diyaloğu yüklü metnin sonuna ekler.
# description.ru: Создаёт диалог вопрос/ответ о сложении, вычитании и умножении небольших чисел на языке интерфейса, а также проверочные вопросы, которых НЕТ в данных — они показывают, выучила ли модель арифметику. --append добавляет диалог к загруженному тексту.
# args: --count 800 --max 20

import argparse
import random
import sys

import kor

FORMATS = {
    "en": ("User: ", "Assistant: ", {"+": "plus", "-": "minus", "*": "times"},
           "What is {a} {op} {b}?", "{a} {op} {b} is {r}."),
    "tr": ("Kullanıcı: ", "Asistan: ", {"+": "artı", "-": "eksi", "*": "çarpı"},
           "{a} {op} {b} kaç eder?", "{a} {op} {b}, {r} eder."),
    "ru": ("Пользователь: ", "Ассистент: ", {"+": "плюс", "-": "минус", "*": "умножить на"},
           "Сколько будет {a} {op} {b}?", "{a} {op} {b} будет {r}."),
}


def all_problems(top):
    probs = []
    for a in range(top + 1):
        for b in range(top + 1):
            probs.append((a, "+", b, a + b))
            if a >= b:
                probs.append((a, "-", b, a - b))
            if a <= 10 and b <= 10:
                probs.append((a, "*", b, a * b))
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=800, help="dialogue pairs to write")
    ap.add_argument("--max", type=int, default=20, help="largest number used")
    ap.add_argument("--tests", type=int, default=12, help="held-out question tests")
    ap.add_argument("--append", action="store_true", help="add to the loaded text dataset")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    q_tag, a_tag, ops, q_fmt, a_fmt = FORMATS.get(kor.lang(), FORMATS["en"])
    probs = all_problems(max(2, args.max))
    random.Random(args.seed).shuffle(probs)
    n_tests = min(args.tests, len(probs) // 4)
    tests, train = probs[:n_tests], probs[n_tests:n_tests + args.count]

    def qa(p):
        a, op, b, r = p
        return q_fmt.format(a=a, op=ops[op], b=b), a_fmt.format(a=a, op=ops[op], b=b, r=r)

    lines = []
    for p in train:
        q, a = qa(p)
        lines += [q_tag + q, a_tag + a]
    text = "\n".join(lines) + "\n"
    if args.append:
        base = kor.read_data()
        if base is None:
            print(kor.tr("--append needs a loaded text dataset.", "--append için yüklü bir metin verisi gerekir.",
                         "Для --append нужен загруженный текст."))
            return 1
        text = base.rstrip("\n") + "\n" + text

    print(kor.tr(f"{len(train)} dialogue pairs written, {len(tests)} held-out question tests.",
                 f"{len(train)} diyalog çifti yazıldı, {len(tests)} soru testi veride yok.",
                 f"Записано пар диалога: {len(train)}, отложенных вопросов: {len(tests)}."))
    for p in train[:3]:
        q, a = qa(p)
        print("  " + q_tag + q + "\n  " + a_tag + a)
    kor.write_data(text)
    kor.write_tests([qa(p) for p in tests])
    return 0


if __name__ == "__main__":
    sys.exit(main())

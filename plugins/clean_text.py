# kor-plugin
# name: Clean text
# name.tr: Metni temizle
# name.ru: Очистка текста
# description: Tidies the loaded text dataset: invisible characters, odd spaces, typographic quotes and repeated blank lines. With --dedupe it also drops repeated question/answer pairs. Fewer distinct characters means a smaller, faster model. The cleaned text is loaded back.
# description.tr: Yüklü metin verisini toparlar: görünmez karakterler, garip boşluklar, tipografik tırnaklar ve art arda boş satırlar. --dedupe ile tekrarlanan soru/cevap çiftlerini de siler. Daha az farklı karakter, daha küçük ve hızlı model demektir. Temizlenen metin geri yüklenir.
# description.ru: Приводит в порядок загруженный текст: невидимые символы, лишние пробелы, типографские кавычки и повторные пустые строки. С --dedupe удаляет повторяющиеся пары вопрос/ответ. Меньше разных символов — меньше и быстрее модель. Очищенный текст загружается обратно.
# args: --dedupe

import collections
import re
import sys
import unicodedata

import kor

REPLACE = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\t": " ",
}
INVISIBLE = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"}
TAG = re.compile(r"^([^:]{1,22}): ")


def clean(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    out = []
    for ch in text:
        if ch in INVISIBLE:
            continue
        ch = REPLACE.get(ch, ch)
        if ch != "\n" and unicodedata.category(ch[0]) == "Cc":
            continue
        out.append(ch)
    lines = [re.sub(r" {2,}", " ", l).strip() for l in "".join(out).split("\n")]
    result, blank = [], 0
    for l in lines:
        blank = blank + 1 if not l else 0
        if blank <= 1:
            result.append(l)
    return "\n".join(result).strip() + "\n"


def dialogue_tags(lines):
    pairs = collections.Counter()
    for a, b in zip(lines, lines[1:]):
        ta, tb = TAG.match(a), TAG.match(b)
        if ta and tb and ta.group(1) != tb.group(1):
            pairs[(ta.group(0), tb.group(0))] += 1
    if not pairs:
        return None
    (q, a), n = pairs.most_common(1)[0]
    return (q, a) if n >= 8 else None


def dedupe(text):
    lines = text.split("\n")
    tags = dialogue_tags(lines)
    if not tags:
        return text, 0
    q, a = tags
    seen, out, removed, i = set(), [], 0, 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].startswith(q) and lines[i + 1].startswith(a):
            key = (lines[i], lines[i + 1])
            if key in seen:
                removed += 1
            else:
                seen.add(key)
                out += [lines[i], lines[i + 1]]
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out), removed


def main():
    text = kor.read_data()
    if text is None or not (kor.data_path() or "").lower().endswith(".txt"):
        print(kor.tr("Load a text dataset first.", "Önce bir metin verisi yükle.", "Сначала загрузите текст."))
        return 1
    before_chars, before_vocab = len(text), len(set(text))
    cleaned = clean(text)
    removed = 0
    if "--dedupe" in sys.argv[1:]:
        cleaned, removed = dedupe(cleaned)
    counts = collections.Counter(cleaned)
    rare = sorted((c for c, n in counts.items() if n < 3 and c != "\n"), key=lambda c: counts[c])
    print(kor.tr("Characters", "Karakter", "Символов") + f": {before_chars} -> {len(cleaned)}")
    print(kor.tr("Distinct characters", "Farklı karakter", "Разных символов") + f": {before_vocab} -> {len(counts)}")
    if removed:
        print(kor.tr("Repeated pairs removed", "Silinen tekrar çift", "Удалено повторов") + f": {removed}")
    if rare:
        print(kor.tr("Rare characters (fewer than 3 uses)", "Nadir karakterler (3'ten az)", "Редкие символы (меньше 3 раз)")
              + ": " + " ".join(repr(c) for c in rare[:40]))
    path = kor.write_data(cleaned)
    print(kor.tr("Written", "Yazıldı", "Записано") + f": {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

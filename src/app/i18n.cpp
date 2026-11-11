#include "i18n.h"

static const char* const kTable[][LANG_COUNT] = {
#define X(id, en, tr, ru) {en, tr, ru},
    KOR_STRINGS(X)
#undef X
};

static int g_lang = LANG_EN;

void set_language(int lang) {
    if (lang >= 0 && lang < LANG_COUNT) g_lang = lang;
}
int get_language() { return g_lang; }

const char* language_name(int lang) {
    static const char* const names[LANG_COUNT] = {"English", "Türkçe", "Русский"};
    return lang >= 0 && lang < LANG_COUNT ? names[lang] : "";
}

const char* tr(S id) { return kTable[(int)id][g_lang]; }
const char* tr_en(S id) { return kTable[(int)id][LANG_EN]; }

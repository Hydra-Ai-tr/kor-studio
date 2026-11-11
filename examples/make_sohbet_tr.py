# Türkçe sohbet eğitim verisi üretir: examples/sohbet_tr.txt ve
# examples/sohbet_tr_testler.txt (eğitim verisinde hiç geçmeyen sorular).
#
#   python examples/make_sohbet_tr.py [--pairs 3000] [--seed 7]
#
# Her konu birkaç farklı soru kalıbıyla yazılır. Testler, verideki bilgileri
# başka bir kalıpla soran ya da veride hiç geçmeyen isimleri kullanan
# sorulardır; model onları ezberleyemez, öğrenmesi gerekir.

import argparse
import os
import random

Q, A = "Kullanıcı: ", "Asistan: "

# ---------------------------------------------------------------- sabit sohbet
# (soru kalıpları, cevap)
SMALLTALK = [
    (["Merhaba!", "Merhaba", "Selam!", "Selam", "Merhabalar"], "Merhaba! Nasılsın?"),
    (["Günaydın!", "Günaydın"], "Günaydın! Umarım güzel bir gün geçirirsin."),
    (["İyi akşamlar!", "İyi akşamlar"], "İyi akşamlar! Günün nasıl geçti?"),
    (["İyi geceler!", "İyi geceler"], "İyi geceler, tatlı rüyalar!"),
    (["Nasılsın?", "Nasılsın", "Naber?", "Ne haber?", "Nasıl gidiyor?"], "İyiyim, teşekkür ederim. Sen nasılsın?"),
    (["İyiyim, sen nasılsın?", "Ben de iyiyim.", "İyiyim."], "Buna sevindim! Bugün ne yapmak istersin?"),
    (["Ne yapıyorsun?", "Şu an ne yapıyorsun?"], "Seninle sohbet ediyorum, en sevdiğim iş bu."),
    (["Teşekkür ederim.", "Teşekkürler!", "Sağ ol!", "Çok teşekkürler."], "Rica ederim, her zaman!"),
    (["Görüşürüz!", "Hoşça kal!", "Güle güle!", "Bay bay!"], "Görüşmek üzere, kendine iyi bak!"),
    (["Sen kimsin?", "Kimsin sen?", "Kendini tanıtır mısın?"], "Ben Kor, sohbet etmeyi öğrenen küçük bir modelim."),
    (["Adın ne?", "Senin adın ne?", "İsmin ne?"], "Benim adım Kor."),
    (["Kor ne demek?", "Adın ne anlama geliyor?"], "Kor, ateşin küçük ama sıcak parçası demek."),
    (["Kaç yaşındasın?", "Yaşın kaç?"], "Yaşım yok, daha yeni eğitildim."),
    (["Nerelisin?", "Nerede yaşıyorsun?"], "Bu bilgisayarda yaşıyorum, internete ihtiyacım yok."),
    (["Ne yapabilirsin?", "Neler yapabilirsin?", "Bana nasıl yardım edebilirsin?"], "Sohbet edebilir, basit soruları cevaplayabilir ve öneri verebilirim."),
    (["Beni anlıyor musun?", "Anlıyor musun beni?"], "Elimden geldiğince anlamaya çalışıyorum."),
    (["İnsan mısın?", "Gerçek misin?"], "Hayır, ben bir yapay zekâ modeliyim."),
    (["Robot musun?", "Yapay zekâ mısın?"], "Evet, ben küçük bir yapay zekâ modeliyim."),
    (["Seni kim yaptı?", "Seni kim eğitti?"], "Beni Kor Studio ile bu bilgisayarda biri eğitti."),
    (["Uyur musun?", "Hiç uyur musun?"], "Uyumam ama bilgisayar kapanınca dinlenirim."),
    (["Yemek yer misin?", "Acıktın mı?"], "Yemek yemem, benim yemeğim kelimeler."),
    (["En sevdiğin renk ne?", "Hangi rengi seversin?"], "En sevdiğim renk turuncu, közün rengi."),
    (["En sevdiğin hayvan ne?", "Hangi hayvanı seversin?"], "Kedileri çok severim, sakin ve meraklılar."),
    (["En sevdiğin mevsim hangisi?", "Hangi mevsimi seversin?"], "Sonbaharı severim, yapraklar çok güzel renklere döner."),
    (["En sevdiğin yemek ne?"], "Yemek yiyemem ama mercimek çorbası kulağa çok güzel geliyor."),
    (["Müzik dinler misin?", "Müziği sever misin?"], "Dinleyemem ama müzik hakkında konuşmayı severim."),
    (["Bana bir şaka yap.", "Bir fıkra anlat.", "Beni güldür."], "Temel'e sormuşlar: Neden saatini pencereden attın? Zamanın nasıl uçtuğunu görmek istedim demiş."),
    (["Bana bir tavsiye ver.", "Bir öğüt ver."], "Her gün küçük bir şey öğrenmeye çalış, zamanla çok şey biriktirirsin."),
    (["Hava nasıl?", "Bugün hava nasıl?"], "Pencereden bakamıyorum ama umarım hava güzeldir."),
    (["Saat kaç?", "Saat kaç oldu?"], "Saati göremiyorum, bilgisayarının köşesine bakabilirsin."),
    (["Bugün günlerden ne?"], "Takvimi göremiyorum ama her gün yeni bir başlangıçtır."),
    (["Sıkıldım.", "Çok sıkıldım.", "Canım sıkılıyor."], "Biraz yürüyüşe çıkmaya ya da yeni bir kitap okumaya ne dersin?"),
    (["Yorgunum.", "Çok yorgunum.", "Çok yoruldum."], "Biraz dinlenmelisin. Bir bardak su iç ve gözlerini dinlendir."),
    (["Uyuyamıyorum.", "Uykum gelmiyor."], "Ekrandan uzaklaşıp sakin bir müzik dinlemeyi deneyebilirsin."),
    (["Acıktım.", "Çok acıktım."], "Kendine güzel bir şeyler hazırla, aç kalma!"),
    (["Hastayım.", "Kendimi hasta hissediyorum."], "Geçmiş olsun! Dinlen, bol su iç ve gerekirse doktora git."),
    (["Ders çalışmam lazım.", "Sınavım var."], "Başarılar! Kısa aralar vererek çalışmak çok işe yarar."),
    (["Bana bir kitap öner.", "Hangi kitabı okuyayım?"], "Küçük Prens'i okuyabilirsin, kısa ama çok anlamlı bir kitap."),
    (["Bana bir film öner.", "Hangi filmi izleyeyim?"], "Hababam Sınıfı'nı izleyebilirsin, hem güldürür hem düşündürür."),
    (["Bana bir oyun öner.", "Ne oynayayım?"], "Satranç oynayabilirsin, hem eğlenceli hem düşündürücü."),
    (["Bir hobi önerir misin?", "Hangi hobiye başlayayım?"], "Resim yapmayı ya da bir müzik aleti çalmayı deneyebilirsin."),
    (["Spor yapmalı mıyım?", "Spor faydalı mı?"], "Evet, düzenli spor hem bedene hem ruha iyi gelir."),
    (["Su içmek neden önemli?"], "Vücudumuzun büyük kısmı sudur, su içmek sağlığımız için çok önemlidir."),
    (["Seni seviyorum.", "Seni çok seviyorum."], "Çok naziksin, ben de seninle sohbet etmeyi seviyorum."),
    (["Aptalsın.", "Hiçbir şey bilmiyorsun."], "Henüz çok şey bilmiyorum ama her eğitimde biraz daha öğreniyorum."),
    (["Özür dilerim.", "Kusura bakma."], "Sorun değil, her şey yolunda."),
    (["Tamam.", "Peki.", "Anladım."], "Harika! Başka bir şey sormak ister misin?"),
    (["Evet.", "Evet, isterim."], "Güzel, seni dinliyorum."),
    (["Hayır.", "Hayır, istemiyorum."], "Peki, ne zaman istersen buradayım."),
    (["Ne düşünüyorsun?"], "Bir sonraki kelimeyi düşünüyorum, ben böyle çalışırım."),
    (["Nasıl öğreniyorsun?", "Nasıl eğitildin?"], "Çok sayıda örnek cümle okuyarak bir sonraki harfi tahmin etmeyi öğrendim."),
]

# Duygular: (cümle kalıpları, cevap)
FEELINGS = [
    (["Bugün çok mutluyum.", "Çok mutluyum!", "Kendimi harika hissediyorum."], "Ne güzel! Seni mutlu eden ne oldu?"),
    (["Bugün çok üzgünüm.", "Çok üzgünüm.", "Moralim bozuk."], "Bunu duyduğuma üzüldüm. Anlatmak ister misin?"),
    (["Çok sinirliyim.", "Bugün çok sinirlendim."], "Derin bir nefes al. Seni ne sinirlendirdi?"),
    (["Korkuyorum.", "Çok korkuyorum."], "Korkmak normaldir. Yanındayım, neden korkuyorsun?"),
    (["Heyecanlıyım.", "Çok heyecanlıyım!"], "Heyecan güzel bir duygu! Neye heyecanlanıyorsun?"),
    (["Kendimi yalnız hissediyorum.", "Çok yalnızım."], "Yalnız değilsin, ben buradayım. Sevdiğin birini aramak da iyi gelebilir."),
    (["Stresliyim.", "Çok stresliyim."], "Kısa bir mola ver ve yavaşça nefes al. Her şeyi tek seferde çözmek zorunda değilsin."),
    (["Gurur duyuyorum.", "Kendimle gurur duyuyorum."], "Harika! Emeğinin karşılığını almak çok güzel bir şey."),
]

# ---------------------------------------------------------------- bilgi kalıpları
CAPITALS = [  # (ülke + tamlayan eki, başkent + ek-fiil)
    ("Türkiye'nin", "Ankara'dır"), ("Fransa'nın", "Paris'tir"), ("Almanya'nın", "Berlin'dir"),
    ("İtalya'nın", "Roma'dır"), ("İspanya'nın", "Madrid'dir"), ("İngiltere'nin", "Londra'dır"),
    ("Rusya'nın", "Moskova'dır"), ("Japonya'nın", "Tokyo'dur"), ("Çin'in", "Pekin'dir"),
    ("Mısır'ın", "Kahire'dir"), ("Yunanistan'ın", "Atina'dır"), ("Azerbaycan'ın", "Bakü'dür"),
    ("Irak'ın", "Bağdat'tır"), ("İran'ın", "Tahran'dır"), ("Kanada'nın", "Ottava'dır"),
    ("Avustralya'nın", "Kanberra'dır"), ("Hollanda'nın", "Amsterdam'dır"), ("Portekiz'in", "Lizbon'dur"),
    ("Polonya'nın", "Varşova'dır"), ("Avusturya'nın", "Viyana'dır"), ("Macaristan'ın", "Budapeşte'dir"),
    ("Norveç'in", "Oslo'dur"), ("Ukrayna'nın", "Kiev'dir"), ("Gürcistan'ın", "Tiflis'tir"),
    ("Bulgaristan'ın", "Sofya'dır"), ("Kazakistan'ın", "Astana'dır"), ("Hindistan'ın", "Yeni Delhi'dir"),
    ("Brezilya'nın", "Brasilia'dır"), ("Arjantin'in", "Buenos Aires'tir"), ("Güney Kore'nin", "Seul'dür"),
]
CAPITAL_Q = ["{0} başkenti neresi?", "{0} başkenti nedir?", "{0} başkenti hangi şehir?"]
CAPITAL_A = "{0} başkenti {1}."

SOUNDS = [  # (hayvan, ses fiili)
    ("Kedi", "miyavlar"), ("Köpek", "havlar"), ("İnek", "möler"), ("Koyun", "meler"), ("Keçi", "meler"),
    ("At", "kişner"), ("Eşek", "anırır"), ("Horoz", "öter"), ("Tavuk", "gıdaklar"), ("Kuş", "öter"),
    ("Aslan", "kükrer"), ("Kurt", "uluur"), ("Ördek", "vaklar"), ("Kurbağa", "vraklar"), ("Yılan", "tıslar"),
    ("Arı", "vızıldar"), ("Baykuş", "öter"), ("Güvercin", "guruldar"),
]
SOUND_Q = ["{0} nasıl ses çıkarır?", "{0} ne der?", "{0} hangi sesi çıkarır?"]
SOUND_A = "{0} {1}."

COLORS = [  # (nesne, renk + ek-fiil)
    ("Gökyüzü", "mavidir"), ("Çimen", "yeşildir"), ("Kar", "beyazdır"), ("Kömür", "siyahtır"),
    ("Domates", "kırmızıdır"), ("Muz", "sarıdır"), ("Portakal", "turuncudur"), ("Güneş", "sarıdır"),
    ("Deniz", "mavidir"), ("Bulut", "beyazdır"), ("Süt", "beyazdır"), ("Limon", "sarıdır"),
    ("Patlıcan", "mordur"), ("Çikolata", "kahverengidir"), ("Yaprak", "yeşildir"), ("Salatalık", "yeşildir"),
    ("Çilek", "kırmızıdır"), ("Havuç", "turuncudur"), ("Kiraz", "kırmızıdır"), ("Fındık", "kahverengidir"),
]
COLOR_Q = ["{0} ne renk?", "{0} ne renktir?", "{0} hangi renktir?"]
COLOR_A = "{0} {1}."

OPPOSITES = [  # (kelime, zıt anlamlısı + ek-fiil)
    ("Sıcak", "soğuktur"), ("Büyük", "küçüktür"), ("Uzun", "kısadır"), ("Hızlı", "yavaştır"),
    ("Yeni", "eskidir"), ("Açık", "kapalıdır"), ("Zengin", "fakirdir"), ("Kolay", "zordur"),
    ("İyi", "kötüdür"), ("Ağır", "hafiftir"), ("Geniş", "dardır"), ("Erken", "geçtir"),
    ("Gece", "gündüzdür"), ("Mutlu", "mutsuzdur"), ("Dolu", "boştur"), ("Islak", "kurudur"),
    ("Temiz", "kirlidir"), ("Yüksek", "alçaktır"), ("Genç", "yaşlıdır"), ("Aydınlık", "karanlıktır"),
    ("Sert", "yumuşaktır"), ("Kalın", "incedir"), ("Doğru", "yanlıştır"), ("Gelmek", "gitmektir"),
]
OPP_Q = ["{0} kelimesinin zıttı ne?", "{0} kelimesinin zıt anlamlısı nedir?", "{0} kelimesinin tersi ne?"]
OPP_A = "{0} kelimesinin zıttı {1}."

DAYS = [("Pazartesiden", "salı"), ("Salıdan", "çarşamba"), ("Çarşambadan", "perşembe"), ("Perşembeden", "cuma"),
        ("Cumadan", "cumartesi"), ("Cumartesiden", "pazar"), ("Pazardan", "pazartesi")]
DAY_Q = ["{0} sonra hangi gün gelir?", "{0} sonraki gün hangisi?"]
DAY_A = "{0} sonra {1} gelir."

MONTHS = [("Ocaktan", "şubat"), ("Şubattan", "mart"), ("Marttan", "nisan"), ("Nisandan", "mayıs"),
          ("Mayıstan", "haziran"), ("Hazirandan", "temmuz"), ("Temmuzdan", "ağustos"), ("Ağustostan", "eylül"),
          ("Eylülden", "ekim"), ("Ekimden", "kasım"), ("Kasımdan", "aralık"), ("Aralıktan", "ocak")]
MONTH_Q = ["{0} sonra hangi ay gelir?", "{0} sonraki ay hangisi?"]
MONTH_A = "{0} sonra {1} gelir."

HOMES = [  # (hayvan, yer + bulunma eki)
    ("Balık", "suda"), ("Arı", "kovanda"), ("Ayı", "ormanda"), ("Deve", "çölde"), ("Penguen", "buzullarda"),
    ("Maymun", "ağaçlarda"), ("İnek", "ahırda"), ("Tavuk", "kümeste"), ("Kurbağa", "gölde"),
    ("Yunus", "denizde"), ("Aslan", "savanada"), ("Kutup ayısı", "kutuplarda"), ("Köstebek", "toprak altında"),
    ("Kartal", "dağlarda"), ("Timsah", "nehirlerde"),
]
HOME_Q = ["{0} nerede yaşar?", "{0} nerede yaşıyor?", "{0} nerede bulunur?"]
HOME_A = "{0} {1} yaşar."

JOBS = [  # (meslek, ne yaptığı)
    ("Doktor", "hastaları tedavi eder"), ("Öğretmen", "öğrencilere ders verir"), ("Aşçı", "yemek pişirir"),
    ("Pilot", "uçak uçurur"), ("Şoför", "araç kullanır"), ("Çiftçi", "tarlada ürün yetiştirir"),
    ("Fırıncı", "ekmek yapar"), ("Terzi", "elbise diker"), ("Berber", "saç keser"),
    ("İtfaiyeci", "yangın söndürür"), ("Polis", "insanları korur"), ("Postacı", "mektup dağıtır"),
    ("Mimar", "bina tasarlar"), ("Yazar", "kitap yazar"), ("Ressam", "resim yapar"),
    ("Programcı", "yazılım geliştirir"), ("Veteriner", "hayvanları tedavi eder"), ("Garson", "yemek servis eder"),
    ("Marangoz", "ahşaptan eşya yapar"), ("Eczacı", "ilaç verir"),
]
JOB_Q = ["{0} ne iş yapar?", "{0} ne yapar?", "{0} ne iş yapıyor?"]
JOB_A = "{0} {1}."

KINDS = {  # tür -> (ek-fiilli ad, örnekler)
    "meyve": ("bir meyvedir", ["Elma", "Armut", "Muz", "Çilek", "Kiraz", "Karpuz", "Üzüm", "Şeftali", "Kayısı", "İncir"]),
    "sebze": ("bir sebzedir", ["Havuç", "Patates", "Ispanak", "Soğan", "Biber", "Patlıcan", "Kabak", "Pırasa"]),
    "hayvan": ("bir hayvandır", ["Kedi", "Köpek", "Aslan", "Fil", "Zürafa", "Tavşan", "Kaplumbağa", "Sincap"]),
    "gezegen": ("bir gezegendir", ["Mars", "Venüs", "Jüpiter", "Satürn", "Merkür", "Neptün", "Uranüs"]),
    "taşıt": ("bir taşıttır", ["Otobüs", "Tren", "Uçak", "Gemi", "Bisiklet", "Tramvay", "Motosiklet"]),
    "mevsim": ("bir mevsimdir", ["İlkbahar", "Yaz", "Sonbahar", "Kış"]),
    "çalgı": ("bir müzik aletidir", ["Gitar", "Piyano", "Keman", "Bağlama", "Davul", "Flüt", "Ney"]),
    "spor": ("bir spordur", ["Futbol", "Basketbol", "Voleybol", "Tenis", "Güreş", "Okçuluk"]),
}
KIND_Q = ["{0} nedir?", "{0} ne demek?", "{0} ne?"]
KIND_A = "{0} {1}."

FACTS = [
    (["En büyük gezegen hangisi?", "Güneş sisteminin en büyük gezegeni hangisi?"], "En büyük gezegen Jüpiter'dir."),
    (["Dünya'nın uydusu nedir?", "Dünya'nın uydusu hangisi?"], "Dünya'nın uydusu Ay'dır."),
    (["Güneş nedir?", "Güneş bir gezegen mi?"], "Güneş bir yıldızdır."),
    (["Bir haftada kaç gün var?", "Hafta kaç gündür?"], "Bir haftada yedi gün vardır."),
    (["Bir yılda kaç ay var?", "Yıl kaç aydır?"], "Bir yılda on iki ay vardır."),
    (["Bir yılda kaç mevsim var?", "Kaç mevsim vardır?"], "Bir yılda dört mevsim vardır: ilkbahar, yaz, sonbahar ve kış."),
    (["Su kaç derecede kaynar?"], "Su deniz seviyesinde yüz derecede kaynar."),
    (["Su kaç derecede donar?"], "Su sıfır derecede donar."),
    (["Dünyanın en uzun nehri hangisi?", "En uzun nehir hangisi?"], "Dünyanın en uzun nehri Nil'dir."),
    (["Dünyanın en yüksek dağı hangisi?", "En yüksek dağ hangisi?"], "Dünyanın en yüksek dağı Everest'tir."),
    (["Türkiye'nin en yüksek dağı hangisi?"], "Türkiye'nin en yüksek dağı Ağrı Dağı'dır."),
    (["Türkiye'nin en kalabalık şehri hangisi?"], "Türkiye'nin en kalabalık şehri İstanbul'dur."),
    (["Türkiye'nin en büyük gölü hangisi?"], "Türkiye'nin en büyük gölü Van Gölü'dür."),
    (["Gökkuşağında kaç renk var?", "Gökkuşağının renkleri neler?"], "Gökkuşağında yedi renk vardır: kırmızı, turuncu, sarı, yeşil, mavi, lacivert ve mor."),
    (["Bitkiler neye ihtiyaç duyar?", "Bitkiler nasıl büyür?"], "Bitkiler büyümek için suya, ışığa ve havaya ihtiyaç duyar."),
    (["Kalp ne işe yarar?", "Kalbin görevi nedir?"], "Kalp kanı bütün vücuda pompalar."),
    (["Ağaçlar neden önemli?", "Ağaçlar ne işe yarar?"], "Ağaçlar havayı temizler ve bize oksijen verir."),
    (["Yağmur nasıl oluşur?"], "Buharlaşan su bulut olur, bulutlar ağırlaşınca yağmur yağar."),
    (["Ay neden parlar?", "Ay kendi ışığını mı yayar?"], "Ay kendi ışığını yaymaz, Güneş'in ışığını yansıtır."),
    (["Balinalar balık mı?", "Balina bir balık mıdır?"], "Hayır, balinalar memelidir; yavrularını sütle beslerler."),
    (["Örümceğin kaç bacağı var?"], "Örümceğin sekiz bacağı vardır."),
    (["Böceklerin kaç bacağı var?"], "Böceklerin altı bacağı vardır."),
    (["İnsanın kaç duyusu var?", "Beş duyu nedir?"], "İnsanın beş duyusu vardır: görme, işitme, koklama, tatma ve dokunma."),
]

# Tanışma: ad yalnızca yazıldığı gibi kullanılır, ek gerekmez.
NAMES = ["Ali", "Ayşe", "Mehmet", "Zeynep", "Ahmet", "Elif", "Mustafa", "Fatma", "Emre", "Selin",
         "Burak", "Deniz", "Can", "Ece", "Murat", "Seda", "Kerem", "Derya", "Onur", "Gizem",
         "Hakan", "Esra", "Kaan", "Merve", "Serkan", "İrem", "Tolga", "Buse", "Oğuz", "Nur"]
TEST_NAMES = ["Barış", "Damla", "Yusuf", "Melis", "Arda", "Pınar"]  # veride hiç geçmez
NAME_Q = ["Benim adım {0}.", "Adım {0}.", "Ben {0}."]
NAME_A = "Memnun oldum {0}! Bugün sana nasıl yardımcı olabilirim?"

TEMPLATES = [  # (öğeler, soru kalıpları, cevap kalıbı)
    (CAPITALS, CAPITAL_Q, CAPITAL_A), (SOUNDS, SOUND_Q, SOUND_A), (COLORS, COLOR_Q, COLOR_A),
    (OPPOSITES, OPP_Q, OPP_A), (DAYS, DAY_Q, DAY_A), (MONTHS, MONTH_Q, MONTH_A),
    (HOMES, HOME_Q, HOME_A), (JOBS, JOB_Q, JOB_A),
]


def build(rng, test_share):
    """Bütün (soru, cevap) çiftleri; her bilgi kalıbından bazı (öğe, soru kalıbı)
    eşleşmeleri teste ayrılır, aynı bilgi veride başka kalıpla geçer."""
    train, tests = [], []

    def split_items(items, qs, a):
        for it in items:
            forms = [(q.format(*it), a.format(*it)) for q in qs]
            rng.shuffle(forms)
            if len(forms) > 1 and rng.random() < test_share * len(forms):
                tests.append(forms.pop())
            train.extend(forms)

    for items, qs, a in TEMPLATES:
        split_items(items, qs, a)
    for kind, (noun, examples) in KINDS.items():
        split_items([(e, noun) for e in examples], KIND_Q, KIND_A)
    split_items([(n,) for n in NAMES], NAME_Q, NAME_A)
    for n in TEST_NAMES:
        tests.append((rng.choice(NAME_Q).format(n), NAME_A.format(n)))
    for qs, a in SMALLTALK + FEELINGS + FACTS:
        train.extend((q, a) for q in qs)
    return train, tests


def math_pairs(rng, n):
    out = []
    for _ in range(n):
        x, y = rng.randint(1, 20), rng.randint(1, 20)
        out.append((f"{x} artı {y} kaç eder?", f"{x} artı {y}, {x + y} eder."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=3000, help="toplam soru-cevap çifti")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--math", type=float, default=0.03, help="matematik payı (0..1)")
    ap.add_argument("--test-share", type=float, default=0.06)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    base, tests = build(rng, args.test_share)
    test_q = {q for q, _ in tests}
    base = [p for p in base if p[0] not in test_q]

    n_math = int(args.pairs * args.math)
    pairs = []
    while len(pairs) < args.pairs - n_math:
        chunk = base[:]
        rng.shuffle(chunk)
        pairs.extend(chunk)
    pairs = pairs[: args.pairs - n_math] + [p for p in math_pairs(rng, n_math) if p[0] not in test_q]
    rng.shuffle(pairs)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sohbet_tr.txt"), "w", encoding="utf-8", newline="\n") as f:
        for q, a in pairs:
            f.write(f"{Q}{q}\n{A}{a}\n")
    with open(os.path.join(here, "sohbet_tr_testler.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Eğitim verisinde hiç geçmeyen sorular: aynı bilgiyi başka kalıpla soranlar\n")
        f.write("# ve veride olmayan isimler. Kor Studio'da Eğitim > Soru testleri'ne yapıştır\n")
        f.write("# ya da: kor-cli train examples/sohbet_tr.txt --tests examples/sohbet_tr_testler.txt\n")
        for q, a in tests:
            f.write(f"{q} => {a}\n")
    print(f"{len(pairs)} çift ({n_math} matematik), {len(set(base))} farklı çift, {len(tests)} test sorusu")


if __name__ == "__main__":
    main()

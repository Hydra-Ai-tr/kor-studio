<p align="center"><img src="../assets/logo.png" width="128" alt="Kor Studio logosu"></p>
<h1 align="center">Kor Studio</h1>
<p align="center"><b>Kendi yapay zekâ modelini elindeki donanımla eğit.</b><br>
<a href="../README.md">English</a> · Türkçe · <a href="README.ru.md">Русский</a></p>

Kor Studio, **kendi sinir ağlarını tasarlaman, eğitmen ve denemen** için ücretsiz ve açık kaynaklı
bir masaüstü atölyesidir. Eğitim motoru, altında hiçbir yapay zekâ kütüphanesi olmadan sıfırdan
C++17 ile yazıldı ve kendini bilgisayarına göre ayarlar; böylece küçük modeller sıradan bir
işlemcide eğitilir.

*Kor*, ateşin közüdür: bir ateşi başlatmaya yeten küçük bir parıltı.

![Eğitim ekranı](screenshots/demo.gif)

## Neler yapar

- **Veri** – düz metin (dil modeli; karakter ya da büyük metinler için alt-kelime parçalarıyla),
  CSV tablosu (sınıflandırıcı ya da regresyon modeli), resim klasörü (resim üreten model) veya
  WAV dosyaları (ses üreten model) yükle; hepsi sıfırdan eğitilir. Tür otomatik algılanır.
- **İnce ayar** – kayıtlı bir `.kor` modeli yükleyip yeni veriyle eğitmeye devam et (tam eğitim
  veya LoRA). Modelin bilmediği yeni karakterler modele eklenir.
- **Model** – ağı görsel olarak tasarla ya da *Sistemime göre otomatik ayarla* düğmesine bas;
  Kor belleğine, işlemcine ve verine uygun boyutu seçsin.
- **Eğitim** – canlı kayıp eğrileri, doğrulama, doğruluk, hız, kalan süre, duraklat/sürdür/durdur,
  erken durdurma ve dosyaya da kaydedilen bir eğitim logu.
- **Kaldığı yerden devam** – eğitim kendini dakikada bir, duraklatınca ve durdurunca kaydeder
  (ağırlıklar, optimizer durumu, en iyi ağırlıklar, grafikler). Elektrik kesilir ya da program
  kapanırsa bir sonraki açılışta *Eğitime devam et* ile tam kaldığı adımdan sürer. Kayıt önce
  geçici dosyaya yazılır, bir önceki kayıt da yedek olarak tutulur; yarım yazılmış dosya eğitimi
  bozmaz.
- **Sohbet** – eğitimi biten modeller otomatik kaydedilir; Sohbet sayfasında bir `.kor` modeli seç
  ve onunla konuşarak dene (sıcaklık, cevap uzunluğu, konuşmayı hatırlama ayarlarıyla).
- **Projeler** – bir proje klasörü veri setini, modelini, eğitim ayarlarını ve soru testlerini bir
  arada tutar; eğitim bitince ve program kapanırken kendini kaydeder. Ana sayfada son projeler,
  son işlemler, model ve veri seti sayıları ile canlı CPU, RAM ve GPU belleği kullanımı görünür.
  `Ctrl+K` sayfa ve işlem arar, `Ctrl+S` projeyi kaydeder, `Ctrl+O` proje açar, `Ctrl+,` ayarları
  açar, ``Ctrl+` `` proje klasöründe `kor-cli` hazır bir terminal açar.
- **Soru testleri** – önce eğit, sonra sor: `soru => beklenen cevap` satırları yaz (diyalog
  verisinde otomatik doldurulur). Test soruları eğitim verisinden çıkarılır, yani model hep
  hiç görmediği sorularla sınanır: ezberi değil gerçekten öğrenip öğrenmediğini ölçer. Model
  eğitim sırasında sınanır, en çok soruyu bilen ağırlıklar korunur ve istenirse bütün cevaplar
  doğru olunca eğitim kendiliğinden durur.
- **Python eğitim becerileri** – eğitimle birlikte çalışıp onu canlı yönlendiren eklentiler:
  öğrenme hızını değiştirir, modelin zayıf olduğu yerlere eğitim örneği ekler, en iyi ağırlıklara
  döner, erken durdurur ve kendi grafiklerini çizer. `kor` kütüphanesiyle birkaç satırda yazılır.
  Örnek diyalog verisinde hazır `pattern_tutor` ve `weak_spots` becerileri, modelin hiç görmediği
  12 sorudan doğru cevapladığı sayıyı 0'dan 3'e çıkarır.
- **Araçlar ve paylaşım** – veri hazırlayan, test yazan ve modeli sınayan Python araçları. Eklentiler
  bir bağlantıyla (GitHub, gist, .zip) ya da topluluk listesinden uygulama içinde veya
  `python -m kor install` ile kurulur ([docs/plugins.md](plugins.md)).
- **Deneme** – eğitim sürerken bile metin, resim veya ses üret ya da tahmin yap; resimler PNG,
  sesler WAV olarak kaydedilir.
- **Kaydet / Yükle** – modeller açık [`.kor` biçiminde](kor-format.md) saklanır.
- **Üç dil** – Türkçe, İngilizce ve Rusça; çalışırken değiştirilebilir.

## Neden hafif

| Teknik | Kazancı |
|---|---|
| Otomatik ayar | Model boyutu, batch ve iş parçacığı sayısı boş belleğe, çekirdek sayısına ve veri boyutuna göre seçilir |
| Gradyan biriktirme | Bellekte tek küçük batch varken büyük etkili batch |
| LoRA ince ayar | Ağırlıkların çok küçük bir kısmını eğiterek eğitilmiş modeli uyarlama |
| Yarı hassasiyetli kayıt | Kayıtlı modeller diskte yarı yer kaplar |
| Çalışma anında SIMD seçimi | İşlemci destekliyorsa AVX2/FMA, desteklemiyorsa taşınabilir çekirdekler |

### Dürüstçe: ne beklemeli

Kor Studio v0.5 eğitimi **işlemcide** yapar. Küçük modeller içindir: tablo ağları ve birkaç milyon
parametreye kadar transformer'lar; bunlar bir dizüstünde saniyeler ya da dakikalar içinde eğitilir.
Resim ve ses modelleri daha yavaştır (her piksel ya da örnek bir parçadır); iyi sonuç için onlarca
dakika ile saatler gerekir. Basit şekiller ve kısa melodiler bekle, fotoğraf ya da konuşma değil. ChatGPT boyutunda bir modeli dizüstünde eğitmez; bunu hiçbir program
yapamaz. Ekran kartı desteği yol haritasındadır.

## Derleme

Gerekenler: CMake 3.20+, C++17 derleyicisi (GCC, Clang veya MSVC), Git ve OpenGL 3.0.
GLFW ve Dear ImGui, CMake tarafından otomatik indirilir.

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/bin/kor-studio
```

### Komut satırı

```bash
kor-cli train examples/diyalog_tr.txt --target 1 --log egitim.log --out sohbet.kor
                                         # diyalog: bütün soru testleri geçince dur
kor-cli test sohbet.kor testler.txt --data examples/diyalog_tr.txt
kor-cli generate sohbet.kor "Kullanıcı: Merhaba" --len 200
kor-cli train buyuk.txt --subword 2000 --out llm.kor        # alt-kelime dil modeli
kor-cli train yeni.txt --from llm.kor --lora 8 --out yeni.kor   # ince ayar
kor-cli train buyuk.txt --out llm.kor --resume   # yarıda kalan eğitime devam (Ctrl+C kaydedip durdurur)
kor-cli demo-images resimler && kor-cli train resimler --out resim.kor
kor-cli generate resim.kor --count 16 --out yeni.png
kor-cli demo-sounds melodiler && kor-cli train melodiler --out ses.kor
kor-cli generate ses.kor --seconds 3 --out yeni.wav
```

## Lisans

[PolyForm Noncommercial 1.0.0](../LICENSE). Kaynak kodu açık: Kor Studio'yu ticari olmayan her amaçla
(kişisel projeler, araştırma, eğitim, okullar ve dernekler) kullanabilir, inceleyebilir, değiştirebilir
ve paylaşabilirsiniz. Kor Studio'yu ya da değiştirilmiş bir sürümünü satmak veya başka ticari amaçla
kullanmak yasaktır. Kendi yazdığınız eklentiler ve eğittiğiniz modeller size aittir.

0.4.0 ve önceki sürümler MIT lisansıyla yayımlandı; o koşullarla alınan kopyalar o lisansı korur.

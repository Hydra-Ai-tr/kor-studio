// Kor Studio - interface strings in English, Turkish and Russian.
//
// To add a string: append one X(Id, "English", "Türkçe", "Русский") line.
// To add a language: add a column here and bump LANG_COUNT in i18n.cpp.
// printf-style placeholders must appear in the same order in every language.
#pragma once

#define KOR_STRINGS(X) \
    X(Tagline, "Train your own AI model on the hardware you already have", "Kendi yapay zekâ modelini elindeki donanımla eğit", "Обучайте собственную ИИ-модель на том железе, что у вас есть") \
    X(NavHome, "Home", "Ana Sayfa", "Главная") \
    X(NavData, "Dataset", "Veri Seti", "Данные") \
    X(NavModel, "Model", "Model", "Модель") \
    X(NavTrain, "Training", "Eğitim", "Обучение") \
    X(NavTest, "Playground", "Deneme", "Проверка") \
    X(NavFiles, "Save / Load", "Kaydet / Yükle", "Сохранить / Загрузить") \
    X(Language, "Language", "Dil", "Язык") \
    X(HomeIntro, "Kor Studio is a free, open-source workshop for designing, training and testing your own neural networks. Its engine is written from scratch in C++ and tunes itself to your computer, so small models train on an ordinary CPU - no expensive graphics card required.", "Kor Studio, kendi sinir ağlarını tasarlaman, eğitmen ve denemen için ücretsiz ve açık kaynaklı bir atölyedir. Motoru sıfırdan C++ ile yazıldı ve kendini bilgisayarına göre ayarlar; böylece küçük modeller sıradan bir işlemcide eğitilir - pahalı bir ekran kartı gerekmez.", "Kor Studio — бесплатная мастерская с открытым исходным кодом для проектирования, обучения и проверки собственных нейросетей. Движок написан с нуля на C++ и сам подстраивается под ваш компьютер, поэтому небольшие модели обучаются на обычном процессоре — дорогая видеокарта не нужна.") \
    X(YourSystem, "Your system", "Sistemin", "Ваша система") \
    X(Cpu, "Processor", "İşlemci", "Процессор") \
    X(Threads, "Threads", "İş parçacığı", "Потоки") \
    X(ThreadsFmt, "%d logical, %d used for training", "%d mantıksal, eğitimde %d kullanılır", "%d логических, для обучения — %d") \
    X(Ram, "Memory", "Bellek", "Память") \
    X(RamFmt, "%.1f GB total, %.1f GB free", "%.1f GB toplam, %.1f GB boş", "%.1f ГБ всего, %.1f ГБ свободно") \
    X(Simd, "Acceleration", "Hızlandırma", "Ускорение") \
    X(Gpu, "Graphics", "Ekran kartı", "Видеокарта") \
    X(GpuNote, "Training runs on the CPU by default; text models can train on the graphics card (OpenCL), usually several times faster.", "Eğitim varsayılan olarak işlemcide çalışır; metin modelleri ekran kartında (OpenCL) eğitilebilir, genelde birkaç kat daha hızlıdır.", "По умолчанию обучение идёт на процессоре; текстовые модели можно обучать на видеокарте (OpenCL), обычно в разы быстрее.") \
    X(UseGpu, "Train on the graphics card", "Ekran kartında eğit", "Обучать на видеокарте") \
    X(UseGpuHelp, "Text models run their whole forward and backward pass on the graphics card through OpenCL; the optimizer stays on the CPU. At the start one step of your data runs on both and the results are compared; if the card is missing, gives other results or the model does not fit in its memory, training falls back to the CPU. The log says what is used. 'kor-cli gpu' measures the speed-up on this computer.", "Metin modellerinin ileri ve geri geçişinin tamamı OpenCL ile ekran kartında çalışır; optimizer işlemcide kalır. Başta verinizden bir adım ikisinde de çalıştırılıp sonuçlar karşılaştırılır; kart yoksa, farklı sonuç verirse ya da model belleğine sığmazsa eğitim işlemciye döner. Neyin kullanıldığı günlükte yazar. 'kor-cli gpu' bu bilgisayardaki hızlanmayı ölçer.", "Прямой и обратный проход текстовых моделей целиком идёт на видеокарте через OpenCL; оптимизатор остаётся на процессоре. В начале один шаг ваших данных считается на обоих и результаты сравниваются; если карты нет, она даёт другие результаты или модель не помещается в её память, обучение возвращается на процессор. Что используется, видно в журнале. 'kor-cli gpu' измеряет ускорение на этом компьютере.") \
    X(QuickStart, "Quick start", "Hızlı başlangıç", "Быстрый старт") \
    X(Step1, "Load text, a CSV table, images or sound", "Metin, CSV tablosu, resim veya ses yükle", "Загрузите текст, таблицу CSV, изображения или звук") \
    X(Step2, "Let Kor pick a model that fits your computer", "Kor bilgisayarına uygun modeli seçsin", "Позвольте Kor подобрать модель под ваш компьютер") \
    X(Step3, "Train and watch the loss fall", "Eğit ve kaybın düşüşünü izle", "Обучайте и следите, как падает ошибка") \
    X(Step4, "Test the model and save it", "Modeli dene ve kaydet", "Проверьте модель и сохраните её") \
    X(Start, "Get started", "Başla", "Начать") \
    X(LightTitle, "Why it is light", "Neden hafif", "Почему это легко") \
    X(Light1, "Auto-tuning: model size, batch and threads are chosen from your RAM and CPU", "Otomatik ayar: model boyutu, batch ve iş parçacıkları RAM ve işlemcine göre seçilir", "Автонастройка: размер модели, батч и потоки подбираются по вашей памяти и процессору") \
    X(Light2, "Gradient accumulation: large effective batches with little memory", "Gradyan biriktirme: az bellekle büyük etkili batch", "Накопление градиентов: большой эффективный батч при малой памяти") \
    X(Light3, "LoRA fine-tuning: adapt a model by training only a tiny fraction of its weights", "LoRA ince ayar: ağırlıkların çok küçük bir kısmını eğiterek modeli uyarlama", "Дообучение LoRA: адаптация модели обучением лишь малой доли весов") \
    X(Light4, "Half-precision files: saved models take half the disk space", "Yarı hassasiyetli dosyalar: kayıtlı modeller diskte yarı yer kaplar", "Файлы половинной точности: модели занимают вдвое меньше места") \
    X(Light5, "Multi-core SIMD kernels chosen at runtime for your CPU", "İşlemcine göre çalışma anında seçilen çok çekirdekli SIMD çekirdekleri", "Многоядерные SIMD-ядра, выбираемые под ваш процессор при запуске") \
    X(DataHelp, "Choose what the model will learn from. Text trains a language model, a table a classifier or a regressor, images a picture generator and sound a sound generator - all from scratch.", "Modelin neyden öğreneceğini seç. Metin bir dil modeli, tablo bir sınıflandırıcı veya regresyon modeli, resimler bir resim üretici, ses de bir ses üretici eğitir - hepsi sıfırdan.", "Выберите, на чём будет учиться модель. Текст обучает языковую модель, таблица — классификатор или регрессор, изображения — генератор картинок, звук — генератор звука; всё с нуля.") \
    X(KindText, "Text (.txt)", "Metin (.txt)", "Текст (.txt)") \
    X(KindTable, "Table (.csv)", "Tablo (.csv)", "Таблица (.csv)") \
    X(KindImages, "Images", "Resimler", "Изображения") \
    X(KindSound, "Sound (.wav)", "Ses (.wav)", "Звук (.wav)") \
    X(FolderNote, "Images are read from the folder of the chosen file, subfolders included. For sound you can pick one WAV file or a folder.", "Resimler seçilen dosyanın klasöründen (alt klasörler dahil) okunur. Ses için tek bir WAV dosyası veya bir klasör seçebilirsin.", "Изображения читаются из папки выбранного файла, включая подпапки. Для звука можно выбрать один WAV-файл или папку.") \
    X(Subword, "Subword tokens", "Alt-kelime parçaları", "Токены-подслова") \
    X(SubwordHelp, "Off: one token per character. On: frequent pieces of words become single tokens, as in large language models, so the model sees more text at once. Good for larger texts.", "Kapalı: her karakter bir parça. Açık: sık geçen kelime parçaları tek parça olur (büyük dil modellerindeki gibi), model bir bakışta daha çok metin görür. Büyük metinler için iyidir.", "Выкл.: один токен на символ. Вкл.: частые части слов становятся отдельными токенами, как в больших языковых моделях, и модель видит больше текста сразу. Хорошо для больших текстов.") \
    X(SubwordSize, "Pieces", "Parça sayısı", "Частей") \
    X(ImgSize, "Image size", "Resim boyutu", "Размер изображения") \
    X(ImgColors, "Colours", "Renk sayısı", "Цветов") \
    X(Mirror, "Mirrored copies", "Ayna kopyalar", "Зеркальные копии") \
    X(MirrorHelp, "Adds a left-right flipped copy of every training image: twice the data for free. Turn off for pictures with text or a fixed direction.", "Her eğitim resminin sağ-sol çevrilmiş kopyasını ekler: bedavaya iki kat veri. Yazı içeren veya yönü önemli resimlerde kapat.", "Добавляет зеркальную копию каждого обучающего изображения: вдвое больше данных бесплатно. Отключите для картинок с текстом или с важным направлением.") \
    X(SampleRate, "Sample rate", "Örnekleme hızı", "Частота дискретизации") \
    X(ExampleImages, "Example pictures", "Örnek resimler", "Примеры картинок") \
    X(ExampleSounds, "Example melodies", "Örnek melodiler", "Примеры мелодий") \
    X(ImagesLbl, "Images", "Resim", "Изображений") \
    X(ColorsLbl, "Colours", "Renk", "Цветов") \
    X(Seconds, "Seconds", "Saniye", "Секунд") \
    X(SoundFiles, "Files", "Dosya", "Файлов") \
    X(Tokens, "Tokens", "Parça", "Токенов") \
    X(Skipped, "%d files could not be read and were skipped.", "%d dosya okunamadı ve atlandı.", "Не удалось прочитать и пропущено файлов: %d.") \
    X(GenImages, "Generate images", "Resim üret", "Создать изображения") \
    X(GenSound, "Generate sound", "Ses üret", "Создать звук") \
    X(HowMany, "How many", "Kaç tane", "Сколько") \
    X(Play, "Play", "Çal", "Играть") \
    X(SavedTo, "Saved: %s", "Kaydedildi: %s", "Сохранено: %s") \
    X(OpenFolder, "Open folder", "Klasörü aç", "Открыть папку") \
    X(GenImgEmpty, "New pictures drawn by the model will appear here.", "Modelin çizdiği yeni resimler burada görünecek.", "Здесь появятся новые картинки, нарисованные моделью.") \
    X(GenSndEmpty, "New sound made by the model will appear here.", "Modelin ürettiği yeni ses burada görünecek.", "Здесь появится новый звук, созданный моделью.") \
    X(Generating, "Generating... %.0f%%", "Üretiliyor... %%%.0f", "Создание... %.0f%%") \
    X(ImageModelFmt, "Image model: %dx%d pixels, %d colours", "Görüntü modeli: %dx%d piksel, %d renk", "Модель изображений: %dx%d пикселей, %d цветов") \
    X(SoundModelFmt, "Sound model: %d Hz, 256 levels", "Ses modeli: %d Hz, 256 seviye", "Модель звука: %d Гц, 256 уровней") \
    X(MediaCtxNote, "An image model always sees one whole image: its context is fixed at size x size + 1.", "Görüntü modeli her zaman bir resmin tamamını görür: bağlamı boyut x boyut + 1 olarak sabittir.", "Модель изображений всегда видит картинку целиком: её контекст фиксирован — размер x размер + 1.") \
    X(PathHint, "File path", "Dosya yolu", "Путь к файлу") \
    X(Browse, "Browse...", "Gözat...", "Обзор...") \
    X(Load, "Load", "Yükle", "Загрузить") \
    X(ExampleText, "Use example text", "Örnek metni kullan", "Пример текста") \
    X(ExampleTable, "Use example table", "Örnek tabloyu kullan", "Пример таблицы") \
    X(ExampleMissing, "Example files were not found next to the program.", "Örnek dosyalar programın yanında bulunamadı.", "Файлы примеров не найдены рядом с программой.") \
    X(Loaded, "Loaded", "Yüklendi", "Загружено") \
    X(Characters, "Characters", "Karakter", "Символов") \
    X(VocabSize, "Distinct characters", "Farklı karakter", "Разных символов") \
    X(TrainVal, "Training / validation", "Eğitim / doğrulama", "Обучение / проверка") \
    X(Preview, "Preview", "Önizleme", "Просмотр") \
    X(Rows, "Rows", "Satır", "Строк") \
    X(Features, "Input columns", "Girdi sütunu", "Входных столбцов") \
    X(Target, "Target column", "Hedef sütun", "Целевой столбец") \
    X(Task, "Task", "Görev", "Задача") \
    X(Classification, "Classification (%d classes)", "Sınıflandırma (%d sınıf)", "Классификация (классов: %d)") \
    X(Regression, "Regression", "Regresyon", "Регрессия") \
    X(BadCells, "%d non-numeric cells were replaced with 0", "Sayısal olmayan %d hücre 0 ile değiştirildi", "Нечисловых ячеек заменено на 0: %d") \
    X(NoData, "Load a dataset first.", "Önce bir veri kümesi yükle.", "Сначала загрузите данные.") \
    X(GoData, "Go to Data", "Veri sayfasına git", "К данным") \
    X(Busy, "Stop training to change this.", "Bunu değiştirmek için eğitimi durdur.", "Остановите обучение, чтобы изменить это.") \
    X(ModelHelp, "Design the network yourself or let Kor choose a size that fits your computer and data.", "Ağı kendin tasarla ya da Kor bilgisayarına ve verine uygun boyutu seçsin.", "Спроектируйте сеть сами или позвольте Kor выбрать размер под ваш компьютер и данные.") \
    X(AutoBtn, "Auto-configure for my system", "Sistemime göre otomatik ayarla", "Автонастройка под мою систему") \
    X(AutoDone, "Auto-configured: %s model", "Otomatik ayarlandı: %s model", "Настроено автоматически: модель %s") \
    X(Arch, "Architecture", "Mimari", "Архитектура") \
    X(Heads, "Attention heads", "Dikkat başlığı", "Головы внимания") \
    X(HeadDim, "Width per head", "Başlık başına genişlik", "Ширина на голову") \
    X(Layers, "Layers", "Katman", "Слои") \
    X(Context, "Context length (characters)", "Bağlam uzunluğu (karakter)", "Длина контекста (символы)") \
    X(Width, "Model width: %d", "Model genişliği: %d", "Ширина модели: %d") \
    X(AddLayer, "Add layer", "Katman ekle", "Добавить слой") \
    X(Units, "units", "nöron", "нейронов") \
    X(LDense, "Dense", "Yoğun (Dense)", "Полносвязный") \
    X(Input, "Input", "Girdi", "Вход") \
    X(Output, "Output", "Çıktı", "Выход") \
    X(Params, "Parameters", "Parametre", "Параметры") \
    X(EstRam, "Estimated training memory", "Tahmini eğitim belleği", "Оценка памяти для обучения") \
    X(Budget, "safe budget %.0f MB", "güvenli sınır %.0f MB", "безопасный предел %.0f МБ") \
    X(Fits, "Fits your computer", "Bilgisayarına sığar", "Подходит вашему компьютеру") \
    X(TooBig, "Too large for the free memory - reduce the size or the batch", "Boş bellek için fazla büyük - boyutu veya batch'i küçült", "Слишком велика для свободной памяти — уменьшите размер или батч") \
    X(Built, "Model is built", "Model oluşturuldu", "Модель создана") \
    X(Trainable, "trainable", "eğitilebilir", "обучаемых") \
    X(ResetModel, "Reset model", "Modeli sıfırla", "Сбросить модель") \
    X(ResetHint, "Reset the model to change its architecture.", "Mimarisini değiştirmek için modeli sıfırla.", "Сбросьте модель, чтобы изменить архитектуру.") \
    X(LoraTitle, "LoRA fine-tuning", "LoRA ince ayar", "Дообучение LoRA") \
    X(LoraHelp, "Freezes the model and trains small adapter matrices instead. Much lighter - ideal for adapting a trained model to new data.", "Modeli dondurur ve yerine küçük adaptör matrislerini eğitir. Çok daha hafiftir - eğitilmiş bir modeli yeni veriye uyarlamak için idealdir.", "Замораживает модель и обучает вместо неё небольшие матрицы-адаптеры. Намного легче — идеально для адаптации обученной модели к новым данным.") \
    X(Rank, "Rank", "Rank", "Ранг") \
    X(EnableLora, "Enable LoRA", "LoRA'yı etkinleştir", "Включить LoRA") \
    X(MergeLora, "Merge adapters into model", "Adaptörleri modele birleştir", "Влить адаптеры в модель") \
    X(BuildModel, "Build model", "Modeli oluştur", "Создать модель") \
    X(Steps, "Steps", "Adım", "Шаги") \
    X(BatchSize, "Batch size", "Batch boyutu", "Размер батча") \
    X(Accum, "Gradient accumulation", "Gradyan biriktirme", "Накопление градиентов") \
    X(EffBatch, "effective batch %d", "etkili batch %d", "эффективный батч %d") \
    X(LR, "Learning rate", "Öğrenme oranı", "Скорость обучения") \
    X(WD, "Weight decay", "Ağırlık azaltma", "Затухание весов") \
    X(ThreadsUse, "CPU threads", "İşlemci iş parçacığı", "Потоки процессора") \
    X(EvalEvery, "Validate every (steps)", "Doğrulama aralığı (adım)", "Проверка каждые (шагов)") \
    X(Optimizer, "Optimizer", "Optimize edici", "Оптимизатор") \
    X(Settings, "Settings", "Ayarlar", "Настройки") \
    X(StartTrain, "Start training", "Eğitimi başlat", "Начать обучение") \
    X(ContinueTrain, "Continue training", "Eğitime devam et", "Продолжить обучение") \
    X(FineTuneHelp, "Fine-tuning: training continues from the weights of the loaded model with a lower learning rate. Turn on LoRA below to change it more gently.", "İnce ayar: eğitim, yüklenen modelin ağırlıklarından daha düşük bir öğrenme hızıyla devam eder. Daha yumuşak değiştirmek için aşağıdan LoRA'yı aç.", "Дообучение: обучение продолжается с весов загруженной модели с меньшей скоростью обучения. Включите LoRA ниже, чтобы менять её мягче.") \
    X(FineTuneAdded, "%d new characters of the data were added to the model.", "Verideki %d yeni karakter modele eklendi.", "В модель добавлено новых символов из данных: %d.") \
    X(FineTuneDropped, "%d characters of the data are unknown to the model and are skipped while LoRA is on.", "Verideki %d karakteri model tanımıyor; LoRA açıkken atlanıyor.", "Модель не знает символов из данных: %d; пока включена LoRA, они пропускаются.") \
    X(Pause, "Pause", "Duraklat", "Пауза") \
    X(Resume, "Resume", "Sürdür", "Продолжить") \
    X(Stop, "Stop", "Durdur", "Стоп") \
    X(StIdle, "Ready", "Hazır", "Готово к запуску") \
    X(StRunning, "Training", "Eğitiliyor", "Обучение") \
    X(StPaused, "Paused", "Duraklatıldı", "Пауза") \
    X(StDone, "Finished", "Tamamlandı", "Завершено") \
    X(StStopped, "Stopped", "Durduruldu", "Остановлено") \
    X(StError, "Error", "Hata", "Ошибка") \
    X(Loss, "Training loss", "Eğitim kaybı", "Ошибка обучения") \
    X(ValLoss, "Validation loss", "Doğrulama kaybı", "Ошибка проверки") \
    X(Accuracy, "Accuracy", "Doğruluk", "Точность") \
    X(Speed, "Speed", "Hız", "Скорость") \
    X(TokPerSec, "characters per second", "karakter / saniye", "символов в секунду") \
    X(SampPerSec, "samples per second", "örnek / saniye", "примеров в секунду") \
    X(Elapsed, "Elapsed", "Geçen süre", "Прошло") \
    X(Eta, "Remaining", "Kalan süre", "Осталось") \
    X(PlotEmpty, "The loss curve will appear here.", "Kayıp eğrisi burada görünecek.", "Здесь появится график ошибки.") \
    X(NoModel, "There is no model yet. Train one or load a saved model.", "Henüz model yok. Bir model eğit veya kayıtlı bir modeli yükle.", "Модели пока нет. Обучите её или загрузите сохранённую.") \
    X(Prompt, "Prompt", "Başlangıç metni", "Начальный текст") \
    X(Temperature, "Creativity (temperature)", "Yaratıcılık (sıcaklık)", "Креативность (температура)") \
    X(Length, "Length (characters)", "Uzunluk (karakter)", "Длина (символы)") \
    X(Generate, "Generate", "Üret", "Сгенерировать") \
    X(Predict, "Predict", "Tahmin et", "Предсказать") \
    X(Result, "Result", "Sonuç", "Результат") \
    X(InputsLbl, "Input values", "Girdi değerleri", "Входные значения") \
    X(SaveModel, "Save model", "Modeli kaydet", "Сохранить модель") \
    X(LoadModel, "Load model", "Modeli yükle", "Загрузить модель") \
    X(Half, "Half precision (half the file size)", "Yarı hassasiyet (yarı dosya boyutu)", "Половинная точность (файл вдвое меньше)") \
    X(Save, "Save", "Kaydet", "Сохранить") \
    X(Saved, "Saved.", "Kaydedildi.", "Сохранено.") \
    X(LoadedModel, "Model loaded.", "Model yüklendi.", "Модель загружена.") \
    X(Failed, "Failed: %s", "Başarısız: %s", "Не удалось: %s") \
    X(FilesHelp, "Models are stored in the open .kor format, documented in the repository.", "Modeller, depoda belgelenen açık .kor biçiminde saklanır.", "Модели хранятся в открытом формате .kor, описанном в репозитории.") \
    X(TextModel, "Text model (transformer)", "Metin modeli (transformer)", "Текстовая модель (трансформер)") \
    X(TableModel, "Table model (MLP)", "Tablo modeli (MLP)", "Табличная модель (MLP)") \
    X(ImageModel, "Image model (transformer)", "Görüntü modeli (transformer)", "Модель изображений (трансформер)") \
    X(SoundModel, "Sound model (transformer)", "Ses modeli (transformer)", "Модель звука (трансформер)") \
    X(NavPlugins, "Tools", "Araçlar", "Инструменты") \
    X(KeepBest, "Keep the best weights", "En iyi ağırlıkları koru", "Сохранять лучшие веса") \
    X(KeepBestHelp, "When training ends, the model returns to the weights that had the lowest validation loss.", "Eğitim bitince model, doğrulama kaybının en düşük olduğu ağırlıklara geri döner.", "По окончании обучения модель возвращается к весам с наименьшей ошибкой проверки.") \
    X(Patience, "Early stop after (checks)", "Erken durdurma (kontrol)", "Ранняя остановка (проверок)") \
    X(PatienceHelp, "Stop when the validation loss has not improved for this many checks in a row. 0 = never stop early.", "Doğrulama kaybı art arda bu kadar kontrolde iyileşmezse eğitimi durdur. 0 = erken durdurma yok.", "Остановить, если ошибка проверки не улучшалась столько проверок подряд. 0 = не останавливать.") \
    X(TestEvery, "Question tests every (steps)", "Soru testi aralığı (adım)", "Вопросы каждые (шагов)") \
    X(TestEveryHelp, "How often the model is quizzed during training. 0 = only at the end.", "Modelin eğitim sırasında ne sıklıkla sınanacağı. 0 = sadece sonda.", "Как часто модель проверяется вопросами во время обучения. 0 = только в конце.") \
    X(StopOnPass, "Stop when all question tests pass", "Tüm soru testleri geçince dur", "Остановить, когда все вопросы пройдены") \
    X(AlignRows, "Start rows at examples", "Satırları örnek başından başlat", "Начинать строки с примеров") \
    X(AlignHelp, "Most training rows start at the beginning of a question, so every answer is learned together with its whole question. Dialogue data only.", "Eğitim satırlarının çoğu bir sorunun başından başlar; böylece her cevap sorusunun tamamıyla birlikte öğrenilir. Sadece diyalog verisinde.", "Большинство обучающих строк начинается с начала вопроса, поэтому каждый ответ изучается вместе со всем вопросом. Только для диалогов.") \
    X(QuestionWeight, "Question weight", "Soru ağırlığı", "Вес вопросов") \
    X(QuestionWeightHelp, "How much predicting the question text counts compared with the answers (1 = the same). Lower values put the model's capacity into the answers. Dialogue data only.", "Soru metnini tahmin etmenin cevaplara göre ne kadar sayılacağı (1 = aynı). Düşük değerler modelin kapasitesini cevaplara verir. Sadece diyalog verisinde.", "Насколько предсказание текста вопроса важно по сравнению с ответами (1 = одинаково). Меньшие значения отдают ёмкость модели ответам. Только для диалогов.") \
    X(DropoutLbl, "Dropout", "Dropout", "Dropout") \
    X(DropoutHelp, "Randomly switches off part of the network while training, against memorizing. Helps larger models on small data; 0 = off.", "Eğitim sırasında ağın bir kısmını rastgele kapatır, ezberlemeye karşı. Küçük veride büyük modellere yardımcı olur; 0 = kapalı.", "Во время обучения случайно отключает часть сети, против запоминания. Помогает большим моделям на малых данных; 0 = выкл.") \
    X(EmaLbl, "Average the weights (EMA)", "Ağırlıkların ortalamasını al (EMA)", "Усреднять веса (EMA)") \
    X(EmaHelp, "Keeps a moving average of the weights and uses it for checks, tests and the final model. It is usually smoother and generalizes better.", "Ağırlıkların hareketli ortalamasını tutar; kontroller, testler ve son model için onu kullanır. Genellikle daha kararlıdır ve daha iyi genelleştirir.", "Хранит скользящее среднее весов и использует его для проверок, тестов и итоговой модели. Обычно оно стабильнее и лучше обобщает.") \
    X(TabChart, "Chart", "Grafik", "График") \
    X(TabLog, "Log", "Log", "Журнал") \
    X(TabTests, "Question tests", "Soru testleri", "Проверка вопросами") \
    X(LogEmpty, "The training log will appear here when you start training.", "Eğitimi başlattığında eğitim logu burada görünecek.", "Журнал обучения появится здесь после запуска обучения.") \
    X(LogHeader, "Kor Studio %s - training log, %s", "Kor Studio %s - eğitim logu, %s", "Kor Studio %s — журнал обучения, %s") \
    X(LogData, "Data: %s", "Veri: %s", "Данные: %s") \
    X(LogModel, "Model: %s, %s parameters (%s trainable)", "Model: %s, %s parametre (%s eğitilebilir)", "Модель: %s, параметров %s (обучаемых %s)") \
    X(LogCfg, "Settings: %d steps, batch %d x %d, learning rate %g, %s, %d threads", "Ayarlar: %d adım, batch %d x %d, öğrenme oranı %g, %s, %d iş parçacığı", "Настройки: шагов %d, батч %d x %d, скорость обучения %g, %s, потоков %d") \
    X(LogTests, "Question tests: %d", "Soru testi: %d", "Проверочных вопросов: %d") \
    X(LogHeldOut, "Test questions held out: %d lines removed from the training data, so the tests ask questions the model has never seen", "Test soruları ayrıldı: eğitim verisinden %d satır çıkarıldı, testler modelin hiç görmediği soruları soruyor", "Проверочные вопросы отделены: из обучающих данных удалено строк: %d, тесты задают вопросы, которых модель не видела") \
    X(HeldOutTooMuch, "after removing the test questions too little training text is left - use fewer tests", "test soruları çıkarılınca çok az eğitim metni kalıyor - daha az test kullan", "после удаления проверочных вопросов осталось слишком мало текста - используйте меньше тестов") \
    X(LogStart, "Training started: %s trainable parameters, %d threads, %s kernels", "Eğitim başladı: %s eğitilebilir parametre, %d iş parçacığı, %s çekirdekleri", "Обучение началось: обучаемых параметров %s, потоков %d, ядра %s") \
    X(LogEval, "step %d   loss %.4f   validation %.4f", "adım %d   kayıp %.4f   doğrulama %.4f", "шаг %d   ошибка %.4f   проверка %.4f") \
    X(LogAcc, "   accuracy %.1f%%", "   doğruluk %%%.1f", "   точность %.1f%%") \
    X(LogBest, "   (best so far)", "   (şimdiye kadarki en iyi)", "   (лучшая пока)") \
    X(LogProbes, "step %d   question tests passed: %d/%d   e.g. %s", "adım %d   geçen soru testi: %d/%d   örnek: %s", "шаг %d   пройдено вопросов: %d/%d   напр. %s") \
    X(LogEarly, "step %d   early stop: validation did not improve in %d checks", "adım %d   erken durdurma: doğrulama %d kontroldür iyileşmedi", "шаг %d   ранняя остановка: проверка не улучшалась %d раз") \
    X(LogTarget, "step %d   all question tests reached the target (%d/%d) - training stopped", "adım %d   soru testleri hedefe ulaştı (%d/%d) - eğitim durduruldu", "шаг %d   вопросы достигли цели (%d/%d) — обучение остановлено") \
    X(LogRestored, "Returned to the best weights from step %d (validation %.4f)", "%d. adımdaki en iyi ağırlıklara dönüldü (doğrulama %.4f)", "Возврат к лучшим весам шага %d (проверка %.4f)") \
    X(LogRestoredTests, "Returned to the weights from step %d, which passed the most question tests (%d/%d)", "En çok soru testini geçen %d. adımdaki ağırlıklara dönüldü (%d/%d)", "Возврат к весам шага %d, прошедшим больше всего вопросов (%d/%d)") \
    X(LogDone, "Finished at step %d after %s", "%d. adımda tamamlandı, süre %s", "Завершено на шаге %d за %s") \
    X(LogStopped, "Stopped at step %d", "%d. adımda durduruldu", "Остановлено на шаге %d") \
    X(LogError, "Error: %s", "Hata: %s", "Ошибка: %s") \
    X(LogManual, "Manual question test: %d/%d passed", "Elle soru testi: %d/%d geçti", "Ручная проверка вопросами: пройдено %d/%d") \
    X(LogFile, "Saved to %s", "Kaydedildiği yer: %s", "Сохраняется в %s") \
    X(CopyLog, "Copy", "Kopyala", "Копировать") \
    X(OpenLogs, "Open logs folder", "Log klasörünü aç", "Открыть папку журналов") \
    X(TestsHelp, "Train, then quiz: write one test per line as  question => expected answer. During training and once at the end the model answers every question; a test passes when the answer matches exactly. Test questions are removed from the training data, so they check questions the model has never seen.", "Önce eğit, sonra sına: her satıra bir test yaz:  soru => beklenen cevap. Eğitim sırasında ve sonunda model her soruyu cevaplar; cevap birebir aynıysa test geçer. Test soruları eğitim verisinden çıkarılır, yani model hiç görmediği sorularla sınanır.", "Обучить, затем проверить: по одному тесту в строке:  вопрос => ожидаемый ответ. Во время обучения и в конце модель отвечает на каждый вопрос; тест пройден, если ответ совпадает точно. Проверочные вопросы удаляются из обучающих данных, поэтому модель проверяется на вопросах, которых не видела.") \
    X(DialogDetected, "Dialogue data: each question is asked as \"%s...\" and the answer is read after \"%s\".", "Diyalog verisi: her soru \"%s...\" olarak sorulur, cevap \"%s\" sonrasından okunur.", "Диалоговые данные: вопрос задаётся как \"%s...\", ответ читается после \"%s\".") \
    X(NoDialog, "Plain text: the model continues each question and the rest of the line is compared.", "Düz metin: model her soruyu devam ettirir ve satırın geri kalanı karşılaştırılır.", "Обычный текст: модель продолжает вопрос, и остаток строки сравнивается.") \
    X(UseDataQuestions, "Take questions from the data", "Soruları veriden al", "Взять вопросы из данных") \
    X(RunTestsNow, "Run tests now", "Testleri şimdi çalıştır", "Проверить сейчас") \
    X(TestsCount, "%d tests", "%d test", "тестов: %d") \
    X(TestsPassed, "Passed: %d / %d", "Geçen: %d / %d", "Пройдено: %d / %d") \
    X(TestsHistory, "Passed tests over time", "Zaman içinde geçen testler", "Пройденные тесты со временем") \
    X(ResultsAt, "Results at step %d", "%d. adımdaki sonuçlar", "Результаты на шаге %d") \
    X(ResultsManual, "Results of the manual run", "Elle çalıştırmanın sonuçları", "Результаты ручной проверки") \
    X(ColQuestion, "Question", "Soru", "Вопрос") \
    X(ColExpected, "Expected", "Beklenen", "Ожидается") \
    X(ColAnswer, "Model's answer", "Modelin cevabı", "Ответ модели") \
    X(Pass, "pass", "geçti", "да") \
    X(Fail, "fail", "kaldı", "нет") \
    X(NoTests, "No tests yet. Add lines such as:  What is the capital of Italy? => Rome", "Henüz test yok. Şöyle satırlar ekle:  İtalya'nın başkenti neresi? => Roma", "Тестов пока нет. Добавьте строки вида:  Столица Италии? => Рим") \
    X(NeedTextModel, "Build or train a text model to run the tests.", "Testleri çalıştırmak için bir metin modeli oluştur veya eğit.", "Создайте или обучите текстовую модель, чтобы запустить тесты.") \
    X(PluginsHelp, "Python plugins extend Kor Studio: skills steer training while it runs, tools prepare data and tests. Write your own with the kor library and share them with a link (docs/plugins.md).", "Python eklentileri Kor Studio'yu genişletir: beceriler eğitimi çalışırken yönlendirir, araçlar veri ve test hazırlar. kor kütüphanesiyle kendi eklentini yaz ve bir bağlantıyla paylaş (docs/plugins.md).", "Плагины на Python расширяют Kor Studio: навыки управляют обучением на ходу, инструменты готовят данные и тесты. Пишите свои с библиотекой kor и делитесь ссылкой (docs/plugins.md).") \
    X(SkillsTitle, "Training skills", "Eğitim becerileri", "Навыки обучения") \
    X(SkillsHelp, "Skills run alongside training and steer it live: they can change the learning rate, add training examples, stop early and keep the best weights. Turned-on skills join every training run.", "Beceriler eğitimle birlikte çalışır ve onu canlı yönlendirir: öğrenme hızını değiştirebilir, eğitim örneği ekleyebilir, erken durdurabilir ve en iyi ağırlıkları tutabilirler. Açık beceriler her eğitime katılır.", "Навыки работают вместе с обучением и управляют им на ходу: меняют скорость обучения, добавляют примеры, останавливают раньше и сохраняют лучшие веса. Включённые навыки участвуют в каждом обучении.") \
    X(ToolsTitle, "Tools", "Araçlar", "Инструменты") \
    X(ToolsHelp, "Tools run once when you press Run: they prepare data, write question tests or quiz the model.", "Araçlar Çalıştır'a basınca bir kez çalışır: veri hazırlar, soru testi yazar veya modeli sınar.", "Инструменты запускаются по кнопке «Запустить»: готовят данные, пишут проверочные вопросы или проверяют модель.") \
    X(SkillOn, "On", "Açık", "Вкл.") \
    X(SkillOff, "Off", "Kapalı", "Выкл.") \
    X(NoSkills, "No skills yet. Install one below or create your own.", "Henüz beceri yok. Aşağıdan birini kur ya da kendin oluştur.", "Навыков пока нет. Установите ниже или создайте свой.") \
    X(NewSkill, "New skill", "Yeni beceri", "Новый навык") \
    X(NewSkillDone, "Created %s. Open it in your editor, then press Refresh list.", "%s oluşturuldu. Düzenleyicinde aç, sonra Listeyi yenile'ye bas.", "Создан %s. Откройте его в редакторе и нажмите «Обновить список».") \
    X(ShareTitle, "Get and share plugins", "Eklenti edin ve paylaş", "Получить и поделиться") \
    X(ShareHelp, "Paste a link to a .py file, a GitHub page, a gist or a .zip, or pick one from the community list. To share yours, put the file online and send the link.", "Bir .py dosyasının, GitHub sayfasının, gist'in veya .zip'in bağlantısını yapıştır ya da topluluk listesinden seç. Kendi eklentini paylaşmak için dosyayı internete koy ve bağlantıyı gönder.", "Вставьте ссылку на файл .py, страницу GitHub, gist или .zip либо выберите из каталога сообщества. Чтобы поделиться своим, выложите файл в сеть и отправьте ссылку.") \
    X(InstallHint, "Link, file path or community id", "Bağlantı, dosya yolu veya topluluk kimliği", "Ссылка, путь к файлу или id из каталога") \
    X(Install, "Install", "Kur", "Установить") \
    X(Installed, "installed", "kurulu", "установлен") \
    X(TrustWarn, "Plugins are Python programs that run on your computer. Install only plugins you trust.", "Eklentiler bilgisayarında çalışan Python programlarıdır. Yalnızca güvendiğin eklentileri kur.", "Плагины — это программы на Python, которые выполняются на вашем компьютере. Устанавливайте только те, которым доверяете.") \
    X(Community, "Community list", "Topluluk listesi", "Каталог сообщества") \
    X(CommunityLoading, "Loading the community list...", "Topluluk listesi yükleniyor...", "Загрузка каталога...") \
    X(CommunityEmpty, "The community list could not be loaded.", "Topluluk listesi yüklenemedi.", "Не удалось загрузить каталог.") \
    X(InstallRunning, "Installing...", "Kuruluyor...", "Установка...") \
    X(InstallDone, "Installed - the plugin is in your list now.", "Kuruldu - eklenti artık listende.", "Установлено - плагин уже в списке.") \
    X(InstallFailed, "Installation failed, see the output below.", "Kurulum başarısız, aşağıdaki çıktıya bak.", "Установка не удалась, см. вывод ниже.") \
    X(NeedsPackages, "Needs: %s", "Gerekli: %s", "Требуется: %s") \
    X(SkillsActive, "Skills", "Beceriler", "Навыки") \
    X(SkillsNoPython, "Skills were skipped: Python 3 was not found.", "Beceriler atlandı: Python 3 bulunamadı.", "Навыки пропущены: Python 3 не найден.") \
    X(SkillsManage, "Manage", "Yönet", "Настроить") \
    X(AddedText, "+%s characters added", "+%s karakter eklendi", "добавлено символов: %s") \
    X(LrFactor, "learning rate x%.2f", "öğrenme hızı x%.2f", "скорость x%.2f") \
    X(LrFixed, "learning rate %g", "öğrenme hızı %g", "скорость %g") \
    X(TabSkills, "Skills", "Beceriler", "Навыки") \
    X(SkillMetricsEmpty, "The skills have not reported any values yet.", "Beceriler henüz bir değer bildirmedi.", "Навыки ещё не сообщили значений.") \
    X(LogHookLr, "learning rate x%.3g", "öğrenme hızı x%.3g", "скорость обучения x%.3g") \
    X(LogHookLrFixed, "fixed learning rate %g", "sabit öğrenme hızı %g", "фиксированная скорость обучения %g") \
    X(LogHookData, "+%lld characters of training text (%lld added in total)", "+%lld karakter eğitim metni (toplam %lld eklendi)", "+%lld символов обучающего текста (всего добавлено %lld)") \
    X(LogHookCleared, "added training text cleared", "eklenen eğitim metni temizlendi", "добавленный текст очищен") \
    X(LogHookSaved, "kept these weights as the best", "bu ağırlıklar en iyi olarak saklandı", "эти веса сохранены как лучшие") \
    X(LogHookRestored, "went back to the best weights (step %lld)", "en iyi ağırlıklara dönüldü (adım %lld)", "возврат к лучшим весам (шаг %lld)") \
    X(LogHookSampling, "training rows: %.0f%% start at examples, question weight %.2f", "eğitim satırları: %%%.0f örnek başından, soru ağırlığı %.2f", "обучающие строки: %.0f%% с начала примеров, вес вопросов %.2f") \
    X(LogHookDropout, "dropout set to %.2f", "dropout %.2f yapıldı", "dropout изменён на %.2f") \
    X(LogHookStop, "stopping training: %s", "eğitim durduruluyor: %s", "обучение останавливается: %s") \
    X(GpuFull, "text models train fully on the GPU", "metin modelleri tamamen ekran kartında eğitilir", "текстовые модели полностью обучаются на GPU") \
    X(GpuPartial, "; this model does not fit in GPU memory, so only parts of it run there", "; bu model ekran kartı belleğine sığmıyor, yalnızca bazı kısımları orada çalışır", "; модель не помещается в память GPU, там выполняется только её часть") \
    X(GpuChecked, "; checked against the CPU on this data (difference ", "; bu veride CPU ile karşılaştırıldı (fark ", "; проверено по CPU на этих данных (разница ") \
    X(GpuDiffers, "; the GPU results differ from the CPU (", "; ekran kartı sonuçları CPU'dan farklı (", "; результаты GPU отличаются от CPU (") \
    X(GpuDiffersTail, "), the model trains on the CPU", "), model CPU'da eğitilir", "), модель обучается на CPU") \
    X(GpuResident, "weights, gradients and optimizer stay on the GPU", "ağırlıklar, gradyanlar ve optimizer ekran kartında kalır", "веса, градиенты и оптимизатор остаются на GPU") \
    X(GpuOptCpu, "the optimizer runs on the CPU: ", "optimizer CPU'da çalışır: ", "оптимизатор работает на CPU: ") \
    X(GpuProblem, "the GPU ran into a problem; training continues with the optimizer on the CPU", "ekran kartında sorun çıktı; eğitim, optimizer CPU'da çalışarak sürüyor", "на GPU возникла проблема; обучение продолжается с оптимизатором на CPU") \
    X(GpuFailed, "the GPU failed; training continues on the CPU from the last weights it handed over", "ekran kartı çalışmayı bıraktı; eğitim, kartın son verdiği ağırlıklarla CPU'da sürüyor", "GPU отказал; обучение продолжается на CPU с последних переданных весов") \
    X(PythonChecking, "Looking for Python...", "Python aranıyor...", "Поиск Python...") \
    X(PythonFound, "%s is ready", "%s hazır", "%s готов") \
    X(PythonMissing, "Python 3 was not found. Install it from python.org (tick \"Add python.exe to PATH\"), then press Check again.", "Python 3 bulunamadı. python.org adresinden kur (\"Add python.exe to PATH\" kutusunu işaretle), sonra Tekrar dene'ye bas.", "Python 3 не найден. Установите его с python.org (отметьте \"Add python.exe to PATH\") и нажмите «Проверить снова».") \
    X(CheckAgain, "Check again", "Tekrar dene", "Проверить снова") \
    X(OpenPlugins, "Open my plugins folder", "Eklenti klasörümü aç", "Открыть мою папку плагинов") \
    X(RefreshList, "Refresh list", "Listeyi yenile", "Обновить список") \
    X(PluginDirs, "Plugins are read from:", "Eklentilerin okunduğu klasörler:", "Плагины загружаются из:") \
    X(NoPlugins, "No plugins found.", "Eklenti bulunamadı.", "Плагины не найдены.") \
    X(Arguments, "Arguments", "Argümanlar", "Аргументы") \
    X(Run, "Run", "Çalıştır", "Запустить") \
    X(PluginRunning, "Running %s...", "%s çalışıyor...", "Выполняется %s...") \
    X(PluginDone, "%s finished.", "%s tamamlandı.", "%s завершён.") \
    X(PluginFailed, "%s failed (exit code %d).", "%s başarısız oldu (çıkış kodu %d).", "%s завершился с ошибкой (код %d).") \
    X(PluginCancelled, "%s was stopped.", "%s durduruldu.", "%s остановлен.") \
    X(PluginNoStart, "Python could not be started.", "Python başlatılamadı.", "Не удалось запустить Python.") \
    X(PluginData, "New dataset from the plugin is loaded: %s", "Eklentinin ürettiği yeni veri yüklendi: %s", "Загружены новые данные от плагина: %s") \
    X(PluginTests, "The plugin wrote %d question tests (Training > Question tests).", "Eklenti %d soru testi yazdı (Eğitim > Soru testleri).", "Плагин записал проверочных вопросов: %d (Обучение > Проверка вопросами).") \
    X(PluginBusy, "Stop training before running a plugin.", "Eklenti çalıştırmadan önce eğitimi durdur.", "Остановите обучение перед запуском плагина.") \
    X(PluginOutput, "Output", "Çıktı", "Вывод") \
    X(Clear, "Clear", "Temizle", "Очистить") \
    X(Workspace, "WORKSPACE", "ÇALIŞMA ALANI", "РАБОЧАЯ ОБЛАСТЬ") \
    X(TrainHelp, "Train the model, watch it learn and quiz it with questions it has never seen.", "Modeli eğit, öğrenişini izle ve hiç görmediği sorularla sına.", "Обучайте модель, следите за прогрессом и проверяйте её на вопросах, которых она не видела.") \
    X(TestHelp, "Generate text, pictures or sound, or run predictions with the current model - even while it trains.", "Mevcut modelle metin, resim veya ses üret ya da tahmin yap - eğitim sürerken bile.", "Генерируйте текст, картинки или звук либо получайте предсказания текущей модели - даже во время обучения.") \
    X(Source, "Source file", "Kaynak dosya", "Исходный файл") \
    X(OrExample, "Or start with an example:", "Ya da bir örnekle başla:", "Или начните с примера:") \
    X(Workflow, "Workflow", "İş akışı", "Порядок работы") \
    X(StepOf, "Step %d of %d", "Adım %d / %d", "Шаг %d из %d") \
    X(PluginFolders, "Plugin folders (%d)", "Eklenti klasörleri (%d)", "Папки плагинов (%d)") \
    X(ModelType, "Type", "Tür", "Тип") \
    X(GenEmpty, "Generated text will appear here.", "Üretilen metin burada görünecek.", "Здесь появится сгенерированный текст.") \
    X(StepLbl, "Step %d", "Adım %d", "Шаг %d") \
    X(NoResults, "No results yet. Run the tests or start training.", "Henüz sonuç yok. Testleri çalıştır veya eğitimi başlat.", "Результатов пока нет. Запустите проверку или обучение.") \
    X(Format, "Format", "Biçim", "Формат") \
    X(FmtDialogue, "Dialogue", "Diyalog", "Диалог") \
    X(FmtPlain, "Plain text", "Düz metin", "Обычный текст") \
    X(ModelFile, "Model file", "Model dosyası", "Файл модели") \
    X(SaveHelp, "Writes the current model to the file above.", "Mevcut modeli yukarıdaki dosyaya yazar.", "Записывает текущую модель в указанный файл.") \
    X(LoadHelp, "Replaces the current model with the one in the file.", "Mevcut modelin yerine dosyadaki modeli koyar.", "Заменяет текущую модель моделью из файла.") \
    X(Network, "Network", "Ağ yapısı", "Структура сети") \
    X(MemoryFit, "Memory", "Bellek", "Память") \
    X(Latest, "latest: %d / %d", "son: %d / %d", "последний: %d / %d") \
    X(NavChat, "Chat", "Sohbet", "Чат") \
    X(ChatHelp, "Load a trained text model (.kor) and talk to it to see how it answers.", "Eğitimi bitmiş bir metin modelini (.kor) yükle ve onunla sohbet ederek nasıl cevap verdiğini dene.", "Загрузите обученную текстовую модель (.kor) и поговорите с ней, чтобы увидеть, как она отвечает.") \
    X(ChatModels, "Models", "Modeller", "Модели") \
    X(ChatModelsHelp, "Finished trainings are saved here automatically. You can also open any .kor file.", "Biten eğitimler buraya otomatik kaydedilir. İstediğin başka bir .kor dosyasını da açabilirsin.", "Завершённые обучения сохраняются сюда автоматически. Можно открыть и любой другой файл .kor.") \
    X(ChatNoModels, "No saved models yet. Finish a training or open a .kor file.", "Henüz kayıtlı model yok. Bir eğitimi bitir ya da bir .kor dosyası aç.", "Сохранённых моделей пока нет. Завершите обучение или откройте файл .kor.") \
    X(ChatUseCurrent, "Use the model in training", "Eğitimdeki modeli kullan", "Взять модель из обучения") \
    X(Refresh, "Refresh", "Yenile", "Обновить") \
    X(ChatInputHint, "Write a message and press Enter...", "Bir mesaj yaz ve Enter'a bas...", "Напишите сообщение и нажмите Enter...") \
    X(ChatSend, "Send", "Gönder", "Отправить") \
    X(ChatClear, "New chat", "Yeni sohbet", "Новый чат") \
    X(ChatEmpty, "Say hello to start the conversation.", "Sohbete başlamak için bir merhaba yaz.", "Поздоровайтесь, чтобы начать разговор.") \
    X(ChatNoModel, "Choose a model above to start chatting.", "Sohbete başlamak için yukarıdan bir model seç.", "Выберите модель выше, чтобы начать разговор.") \
    X(ChatNotText, "This is not a text model. Picture and sound models can be tried on the Test page.", "Bu bir metin modeli değil. Resim ve ses modellerini Dene sayfasında deneyebilirsin.", "Это не текстовая модель. Модели картинок и звука можно попробовать на странице проверки.") \
    X(ChatMemory, "Remember the conversation", "Konuşmayı hatırla", "Помнить разговор") \
    X(ChatMemoryHelp, "Earlier messages are given to the model too, as far as its context reaches. Turn it off to ask every question on its own, like the question tests do.", "Önceki mesajlar da modele verilir (bağlamının yettiği kadar). Kapatırsan her soru tek başına sorulur, soru testlerindeki gibi.", "Предыдущие сообщения тоже передаются модели (сколько вмещает контекст). Выключите, чтобы каждый вопрос задавался отдельно, как в проверке вопросами.") \
    X(ChatTags, "Dialogue tags", "Diyalog etiketleri", "Метки диалога") \
    X(ChatTagsHelp, "The line beginnings the model learned, e.g. 'Kullanıcı: ' and 'Asistan: '. Models trained with this version remember them by themselves.", "Modelin öğrendiği satır başları, örn. 'Kullanıcı: ' ve 'Asistan: '. Bu sürümle eğitilen modeller bunları kendileri hatırlar.", "Начала строк, которые выучила модель, например 'Kullanıcı: ' и 'Asistan: '. Модели, обученные этой версией, помнят их сами.") \
    X(ChatUser, "You", "Sen", "Вы") \
    X(ChatBot, "Model", "Model", "Модель") \
    X(ChatMaxLen, "Longest answer (tokens)", "En uzun cevap (token)", "Самый длинный ответ (токены)") \
    X(ChatModelFmt, "%s parameters, context %d tokens", "%s parametre, bağlam %d token", "%s параметров, контекст %d токенов") \
    X(ChatUnknown, "(the model knows none of these characters)", "(model bu karakterlerin hiçbirini bilmiyor)", "(модель не знает ни одного из этих символов)") \
    X(ResumeTitle, "Unfinished training found", "Yarım kalan eğitim bulundu", "Найдено незавершённое обучение") \
    X(ResumeText, "The last training stopped at step %d of %d after %s. It was saved automatically and can continue from where it left off.", "Son eğitim %d / %d. adımda, %s sonra kaldı. Otomatik kaydedilmişti; kaldığı yerden devam edebilir.", "Последнее обучение остановилось на шаге %d из %d через %s. Оно было сохранено автоматически и может продолжиться с того же места.") \
    X(ResumeData, "Data: %s", "Veri: %s", "Данные: %s") \
    X(ResumeGo, "Continue training", "Eğitime devam et", "Продолжить обучение") \
    X(ResumeDiscard, "Discard", "Sil", "Удалить") \
    X(ResumeWarn, "Continue or discard the unfinished training first - a new training would replace it.", "Önce yarım kalan eğitime devam et ya da onu sil - yeni bir eğitim onun yerine geçer.", "Сначала продолжите или удалите незавершённое обучение — новое обучение заменит его.") \
    X(ResumeFailed, "Could not continue: %s", "Devam edilemedi: %s", "Не удалось продолжить: %s") \
    X(ResumeNoData, "the data was not found: %s", "veri bulunamadı: %s", "данные не найдены: %s") \
    X(AutosaveLbl, "Autosaved at step %d", "Otomatik kaydedildi: adım %d", "Автосохранение: шаг %d") \
    X(AutosaveWait, "Autosave every minute", "Dakikada bir otomatik kayıt", "Автосохранение раз в минуту") \
    X(AutosaveHelp, "The training saves itself every minute, when it is paused and when it is stopped. If the computer turns off or the program closes, the next start offers to continue from the last save.", "Eğitim kendini dakikada bir, duraklatılınca ve durdurulunca kaydeder. Bilgisayar kapanır ya da program kapanırsa, bir sonraki açılışta son kayıttan devam etmeyi önerir.", "Обучение сохраняет себя каждую минуту, при паузе и при остановке. Если компьютер выключится или программа закроется, при следующем запуске можно будет продолжить с последнего сохранения.") \
    X(AutosaveFailed, "Autosave failed: %s", "Otomatik kayıt başarısız: %s", "Автосохранение не удалось: %s") \
    X(LogResumed, "Continuing the saved training from step %d", "Kayıtlı eğitime %d. adımdan devam ediliyor", "Продолжение сохранённого обучения с шага %d") \
    X(LogSavedModel, "Finished model saved: %s", "Biten model kaydedildi: %s", "Готовая модель сохранена: %s") \
    X(LogCheckpointFail, "Autosave failed at step %d: %s", "%d. adımda otomatik kayıt başarısız: %s", "Автосохранение на шаге %d не удалось: %s") \
    X(HeroSub, "Build, train, test and ship your AI models.", "Yapay zekâ modellerini geliştir, eğit, test et ve dağıt.", "Создавайте, обучайте, проверяйте и распространяйте свои ИИ-модели.") \
    X(NewModel, "New model", "Yeni Model Oluştur", "Новая модель") \
    X(OpenProject, "Open project", "Proje Aç", "Открыть проект") \
    X(ChatWithModel, "Chat with AI", "AI ile Sohbet", "Чат с ИИ") \
    X(SysResources, "System resources", "Sistem Kaynakları", "Системные ресурсы") \
    X(CpuLoad, "CPU", "CPU", "ЦП") \
    X(RamLoad, "RAM", "RAM", "ОЗУ") \
    X(Kernels, "Kernels", "Çekirdekler", "Ядра") \
    X(QuickActions, "Quick actions", "Hızlı İşlemler", "Быстрые действия") \
    X(QaTrain, "Start model training", "Model Eğitimi Başlat", "Начать обучение модели") \
    X(QaWatch, "Watch the training", "Eğitimi izle", "Следить за обучением") \
    X(QaImport, "Import a dataset", "Veri Seti İçe Aktar", "Импортировать данные") \
    X(QaTest, "Test the model", "Modeli Test Et", "Проверить модель") \
    X(QaNewProject, "Create a project", "Proje Oluştur", "Создать проект") \
    X(Shortcuts, "Shortcuts", "Kısa Yollar", "Горячие клавиши") \
    X(ScSearch, "Quick search", "Hızlı Arama", "Быстрый поиск") \
    X(ScSave, "Save project", "Projeyi kaydet", "Сохранить проект") \
    X(RecentProjects, "Recent projects", "Son Projeler", "Недавние проекты") \
    X(SeeAll, "See all", "Tümünü Gör", "Все") \
    X(NoProjects, "No projects yet. Create one to keep the data, model and settings together.", "Henüz proje yok. Veriyi, modeli ve ayarları bir arada tutmak için bir proje oluştur.", "Проектов пока нет. Создайте проект, чтобы хранить данные, модель и настройки вместе.") \
    X(Activity, "Activity", "Aktivite", "Активность") \
    X(NoActivity, "Nothing yet - load data or open a project.", "Henüz bir şey yok - veri yükle ya da bir proje aç.", "Пока ничего нет — загрузите данные или откройте проект.") \
    X(AllLogs, "Open the training logs", "Tüm logları görüntüle", "Открыть журналы обучения") \
    X(TileModels, "Models", "Modeller", "Модели") \
    X(TileModelsSub, "saved", "kayıtlı", "сохранено") \
    X(TileProjects, "Projects", "Projeler", "Проекты") \
    X(TileTrainIdle, "ready", "hazır", "готово") \
    X(TileCpuSub, "Training uses the CPU", "Eğitim işlemcide", "Обучение на ЦП") \
    X(AgoNow, "just now", "az önce", "только что") \
    X(AgoMin, "%d min ago", "%d dk önce", "%d мин назад") \
    X(AgoHour, "%d h ago", "%d saat önce", "%d ч назад") \
    X(AgoDay, "%d days ago", "%d gün önce", "%d дн назад") \
    X(ActTrainStart, "Training started: %s", "Eğitim başladı: %s", "Обучение началось: %s") \
    X(ActTrainDone, "Training finished: %s", "Eğitim bitti: %s", "Обучение завершено: %s") \
    X(ActTrainStop, "Training stopped: %s", "Eğitim durduruldu: %s", "Обучение остановлено: %s") \
    X(ActResumed, "Training continued: %s", "Eğitime devam edildi: %s", "Обучение продолжено: %s") \
    X(ActModelSaved, "Model saved: %s", "Model kaydedildi: %s", "Модель сохранена: %s") \
    X(ActDataLoaded, "Dataset processed: %s", "Veri seti işlendi: %s", "Набор данных обработан: %s") \
    X(ActProjectSaved, "Project updated: %s", "Proje güncellendi: %s", "Проект обновлён: %s") \
    X(ActProjectOpened, "Project opened: %s", "Proje açıldı: %s", "Проект открыт: %s") \
    X(ActTrainRunning, "Training running: %s", "Eğitim sürüyor: %s", "Идёт обучение: %s") \
    X(NavProjects, "Projects", "Projeler", "Проекты") \
    X(ProjectsHelp, "A project keeps a dataset, its model, the training settings and the question tests together in one folder.", "Bir proje; veri setini, modelini, eğitim ayarlarını ve soru testlerini tek bir klasörde bir arada tutar.", "Проект хранит набор данных, его модель, настройки обучения и проверочные вопросы вместе в одной папке.") \
    X(NewProject, "New project", "Yeni proje", "Новый проект") \
    X(ProjectName, "Project name", "Proje adı", "Название проекта") \
    X(CreateBtn, "Create", "Oluştur", "Создать") \
    X(CancelBtn, "Cancel", "İptal", "Отмена") \
    X(NewProjectHelp, "A folder is created in Documents/Kor Studio Projects. The loaded data (single files) and the model are saved into it.", "Belgeler/Kor Studio Projects içinde bir klasör oluşturulur. Yüklü veri (tek dosyaysa) ve model içine kaydedilir.", "В папке Документы/Kor Studio Projects создаётся папка. В неё сохраняются загруженные данные (если это один файл) и модель.") \
    X(ProjectLbl, "PROJECT", "PROJE", "ПРОЕКТ") \
    X(NoProjectOpen, "No project open", "Açık proje yok", "Проект не открыт") \
    X(ProjectSaved, "Project saved.", "Proje kaydedildi.", "Проект сохранён.") \
    X(ProjectFailed, "Project error: %s", "Proje hatası: %s", "Ошибка проекта: %s") \
    X(ProjectModelFmt, "model: %s parameters", "model: %s parametre", "модель: %s параметров") \
    X(ProjectNoModel, "no model yet", "henüz model yok", "модели пока нет") \
    X(ProjectAuto, "The open project saves itself when a training finishes and when the app closes.", "Açık proje, bir eğitim bittiğinde ve program kapanırken kendini kaydeder.", "Открытый проект сохраняется сам, когда обучение завершается и когда программа закрывается.") \
    X(ProjectsFolder, "Projects folder", "Proje klasörü", "Папка проектов") \
    X(RemoveFromList, "Remove from the list", "Listeden kaldır", "Убрать из списка") \
    X(ProjectCurrent, "open", "açık", "открыт") \
    X(ProjectExists, "a project with this name already exists", "bu adla bir proje zaten var", "проект с таким названием уже существует") \
    X(PaletteHint, "Search pages and actions...", "Sayfa ya da işlem ara...", "Поиск страниц и действий...") \
    X(DiskLbl, "Disk", "Disk", "Диск") \
    X(NoMatches, "Nothing matches.", "Eşleşen bir şey yok.", "Ничего не найдено.") \
    X(WorkspaceTitle, "Workspace", "Çalışma Alanı", "Рабочая область") \
    X(FoldSidebar, "Fold the sidebar", "Kenar çubuğunu daralt", "Свернуть боковую панель") \
    X(UnfoldSidebar, "Unfold the sidebar", "Kenar çubuğunu aç", "Развернуть боковую панель") \
    X(DiskUsage, "Disk usage", "Disk Kullanımı", "Использование диска") \
    X(LocalAccount, "Local account", "Yerel hesap", "Локальная учётная запись") \
    X(Notifications, "Notifications", "Bildirimler", "Уведомления") \
    X(NoNotifications, "No notifications yet.", "Henüz bildirim yok.", "Уведомлений пока нет.") \
    X(CloudOk, "Community plugins: ready (Python found). Click to browse.", "Topluluk eklentileri: hazır (Python bulundu). Göz atmak için tıkla.", "Плагины сообщества: готово (Python найден). Нажмите, чтобы открыть.") \
    X(CloudOff, "Community plugins need Python 3. Click for details.", "Topluluk eklentileri için Python 3 gerekli. Ayrıntılar için tıkla.", "Для плагинов сообщества нужен Python 3. Нажмите, чтобы узнать больше.") \
    X(Terminal, "Terminal", "Terminal", "Терминал") \
    X(OpenConfigDir, "Open the settings folder", "Ayar klasörünü aç", "Открыть папку настроек") \
    X(OpenModelsDir, "Open the models folder", "Model klasörünü aç", "Открыть папку моделей") \
    X(AboutLine, "Kor Studio %s - free for non-commercial use (PolyForm Noncommercial 1.0.0).", "Kor Studio %s - ticari olmayan kullanım için ücretsiz (PolyForm Noncommercial 1.0.0).", "Kor Studio %s - бесплатно для некоммерческого использования (PolyForm Noncommercial 1.0.0).") \
    X(ProjectDesc, "Description (optional)", "Açıklama (isteğe bağlı)", "Описание (необязательно)") \
    X(ProjectParamsDesc, "%s-parameter model", "%s parametreli model", "модель на %s параметров") \
    X(TileDatasets, "Datasets", "Veri Setleri", "Наборы данных") \
    X(TileTotal, "Total: %s", "Toplam: %s", "Всего: %s") \
    X(TileTrainings, "Trainings", "Eğitimler", "Обучения") \
    X(TileRunning, "Running", "Devam Ediyor", "Идёт") \
    X(TileActive, "Active", "Aktif", "Активна") \
    X(TileReady, "Ready", "Hazır", "Готово") \
    X(TileNoModel, "None loaded", "Yüklü değil", "Не загружена") \
    X(TileDetected, "Detected", "Algılandı", "Обнаружен") \
    X(TileNoGpu, "None", "Yok", "Нет") \
    X(UpdateAvailable, "New version available: %s", "Yeni sürüm var: %s", "Доступна новая версия: %s") \
    X(UpdateDownload, "Download", "İndir", "Скачать") \
    X(UpdateAtStart, "Check for updates at startup", "Açılışta güncellemeleri denetle", "Проверять обновления при запуске") \
    X(UpdateNow, "Check now", "Şimdi denetle", "Проверить сейчас") \
    X(UpdateLatest, "You have the latest version.", "En güncel sürümü kullanıyorsun.", "У вас последняя версия.") \
    X(UpdateChecking, "Checking for updates...", "Güncellemeler denetleniyor...", "Проверка обновлений...") \
    X(UpdateFailed, "Could not check for updates: %s", "Güncellemeler denetlenemedi: %s", "Не удалось проверить обновления: %s") \
    X(Updates, "Updates", "Güncellemeler", "Обновления") \
    X(TabCompare, "Compare", "Karşılaştır", "Сравнение") \
    X(CompareEmpty, "Every finished or stopped training is kept here, so you can compare the curves and results of different settings.", "Biten veya durdurulan her eğitim burada saklanır; farklı ayarların eğrilerini ve sonuçlarını karşılaştırabilirsin.", "Здесь хранится каждое завершённое или остановленное обучение, чтобы сравнивать кривые и результаты разных настроек.") \
    X(CompareHint, "Tick the trainings to draw on the chart. The best value of each column is green.", "Grafikte görmek istediğin eğitimleri işaretle. Her sütundaki en iyi değer yeşildir.", "Отметьте обучения для графика. Лучшее значение в каждом столбце — зелёное.") \
    X(ColData, "Data", "Veri", "Данные") \
    X(ColDate, "Date", "Tarih", "Дата") \
    X(ColBestVal, "Best val. loss", "En iyi doğr. kaybı", "Лучшая ошибка проверки") \
    X(DeleteRun, "Delete this record", "Bu kaydı sil", "Удалить запись") \
    X(RunStopped, "stopped", "durduruldu", "остановлено") \
    X(ServeApi, "Serve as API", "API olarak sun", "Запустить API") \
    X(ServeApiHelp, "Starts kor-cli serve in a terminal: an OpenAI-compatible API at http://127.0.0.1:8080/v1 that other programs can use. Only this computer can reach it.", "kor-cli serve'ü bir terminalde başlatır: diğer programların kullanabileceği, http://127.0.0.1:8080/v1 adresinde OpenAI uyumlu bir API. Yalnızca bu bilgisayardan erişilebilir.", "Запускает kor-cli serve в терминале: API, совместимый с OpenAI, по адресу http://127.0.0.1:8080/v1 для других программ. Доступен только с этого компьютера.") \
    X(Tools, "Tools:", "Araçlar:", "Инструменты:") \
    X(CleanData, "Clean data", "Veriyi temizle", "Очистить данные") \
    X(CleanHelp, "A tidy copy is saved as %s and loaded; your file stays as it is.", "Temiz bir kopya %s olarak kaydedilip yüklenir; asıl dosyan değişmez.", "Чистая копия сохраняется как %s и загружается; исходный файл не меняется.") \
    X(CleanSpaces, "Fix spaces (line ends, tabs, double spaces)", "Boşlukları düzelt (satır sonları, sekmeler, çift boşluklar)", "Исправить пробелы (концы строк, табуляции, двойные пробелы)") \
    X(CleanEmpty, "Remove empty lines", "Boş satırları kaldır", "Удалить пустые строки") \
    X(CleanDupes, "Remove repeats", "Tekrarları kaldır", "Удалить повторы") \
    X(CleanDupesHelp, "Dialogue: the same question with the same answer. Plain text: the same line, if it has at least 24 characters.", "Diyalogda: aynı soru ve aynı cevap. Düz metinde: en az 24 karakterlik aynı satır.", "Диалог: тот же вопрос с тем же ответом. Обычный текст: та же строка длиной от 24 символов.") \
    X(CleanBroken, "Remove questions without an answer and answers without a question", "Cevapsız soruları ve sorusuz cevapları kaldır", "Удалить вопросы без ответа и ответы без вопроса") \
    X(CleanPreview, "%d lines become %d: %d repeats, %d empty and %d broken lines go, %d lines are tidied.", "%d satır %d satıra iner: %d tekrar, %d boş ve %d yarım satır kaldırılır, %d satır düzeltilir.", "%d строк станет %d: удаляются повторы (%d), пустые (%d) и неполные (%d) строки, исправляется строк: %d.") \
    X(CleanLoad, "Clean and load", "Temizle ve yükle", "Очистить и загрузить") \
    X(CleanNothing, "Nothing to clean: the data is already tidy.", "Temizlenecek bir şey yok: veri zaten düzenli.", "Нечего очищать: данные уже в порядке.") \
    X(CleanDone, "Cleaned: %d repeats and %d other lines removed, %d lines tidied. Saved as %s.", "Temizlendi: %d tekrar ve %d başka satır kaldırıldı, %d satır düzeltildi. %s olarak kaydedildi.", "Очищено: удалено повторов %d и других строк %d, исправлено строк %d. Сохранено как %s.") \
    X(ImportJsonl, "Import JSONL", "JSONL içe aktar", "Импорт JSONL") \
    X(ImportJsonlHelp, "Chat datasets in JSON Lines (OpenAI messages, ShareGPT, Alpaca, prompt/completion) become question-answer text saved next to the file.", "JSON Lines sohbet verileri (OpenAI messages, ShareGPT, Alpaca, prompt/completion) soru-cevap metnine çevrilip dosyanın yanına kaydedilir.", "Наборы чатов в JSON Lines (OpenAI messages, ShareGPT, Alpaca, prompt/completion) превращаются в текст вопрос-ответ рядом с файлом.") \
    X(ImportDone, "Imported %d dialogues (%s) and saved them as %s.", "%d diyalog içe aktarıldı (%s), %s olarak kaydedildi.", "Импортировано диалогов: %d (%s), сохранено как %s.") \
    X(CantWrite, "cannot write %s", "%s yazılamadı", "не удалось записать %s")

enum class S {
#define X(id, en, tr, ru) id,
    KOR_STRINGS(X)
#undef X
    COUNT
};

enum Lang { LANG_EN = 0, LANG_TR = 1, LANG_RU = 2, LANG_COUNT = 3 };

void set_language(int lang);
int get_language();
const char* language_name(int lang);  // in its own language
const char* tr(S id);
const char* tr_en(S id);

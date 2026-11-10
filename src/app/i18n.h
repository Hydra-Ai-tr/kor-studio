// Kor Studio - interface strings in English, Turkish and Russian.
//
// To add a string: append one X(Id, "English", "Türkçe", "Русский") line.
// To add a language: add a column here and bump LANG_COUNT in i18n.cpp.
// printf-style placeholders must appear in the same order in every language.
#pragma once

#define KOR_STRINGS(X) \
    X(Tagline, "Train your own AI model on the hardware you already have", "Kendi yapay zekâ modelini elindeki donanımla eğit", "Обучайте собственную ИИ-модель на том железе, что у вас есть") \
    X(NavHome, "Home", "Ana Sayfa", "Главная") \
    X(NavData, "Data", "Veri", "Данные") \
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
    X(GpuNote, "Training currently runs on the CPU. A GPU backend is on the roadmap.", "Eğitim şu an işlemcide çalışır. Ekran kartı desteği yol haritasında.", "Сейчас обучение идёт на процессоре. Поддержка видеокарт — в планах.") \
    X(QuickStart, "Quick start", "Hızlı başlangıç", "Быстрый старт") \
    X(Step1, "1.  Load a text file or a CSV table", "1.  Bir metin dosyası veya CSV tablosu yükle", "1.  Загрузите текстовый файл или таблицу CSV") \
    X(Step2, "2.  Let Kor pick a model that fits your computer", "2.  Kor bilgisayarına uygun modeli seçsin", "2.  Позвольте Kor подобрать модель под ваш компьютер") \
    X(Step3, "3.  Train and watch the loss fall", "3.  Eğit ve kaybın düşüşünü izle", "3.  Обучайте и следите, как падает ошибка") \
    X(Step4, "4.  Test the model and save it", "4.  Modeli dene ve kaydet", "4.  Проверьте модель и сохраните её") \
    X(Start, "Get started", "Başla", "Начать") \
    X(LightTitle, "Why it is light", "Neden hafif", "Почему это легко") \
    X(Light1, "Auto-tuning: model size, batch and threads are chosen from your RAM and CPU", "Otomatik ayar: model boyutu, batch ve iş parçacıkları RAM ve işlemcine göre seçilir", "Автонастройка: размер модели, батч и потоки подбираются по вашей памяти и процессору") \
    X(Light2, "Gradient accumulation: large effective batches with little memory", "Gradyan biriktirme: az bellekle büyük etkili batch", "Накопление градиентов: большой эффективный батч при малой памяти") \
    X(Light3, "LoRA fine-tuning: adapt a model by training only a tiny fraction of its weights", "LoRA ince ayar: ağırlıkların çok küçük bir kısmını eğiterek modeli uyarlama", "Дообучение LoRA: адаптация модели обучением лишь малой доли весов") \
    X(Light4, "Half-precision files: saved models take half the disk space", "Yarı hassasiyetli dosyalar: kayıtlı modeller diskte yarı yer kaplar", "Файлы половинной точности: модели занимают вдвое меньше места") \
    X(Light5, "Multi-core SIMD kernels chosen at runtime for your CPU", "İşlemcine göre çalışma anında seçilen çok çekirdekli SIMD çekirdekleri", "Многоядерные SIMD-ядра, выбираемые под ваш процессор при запуске") \
    X(DataHelp, "Choose what the model will learn from. Text trains a small language model; a table trains a classifier or a regressor.", "Modelin neyden öğreneceğini seç. Metin küçük bir dil modeli eğitir; tablo bir sınıflandırıcı veya regresyon modeli eğitir.", "Выберите, на чём будет учиться модель. Текст обучает небольшую языковую модель; таблица — классификатор или регрессор.") \
    X(KindText, "Text (.txt)", "Metin (.txt)", "Текст (.txt)") \
    X(KindTable, "Table (.csv)", "Tablo (.csv)", "Таблица (.csv)") \
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
    X(TokPerSec, "%.0f chars/s", "%.0f karakter/sn", "%.0f симв/с") \
    X(SampPerSec, "%.0f samples/s", "%.0f örnek/sn", "%.0f примеров/с") \
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
    X(TableModel, "Table model (MLP)", "Tablo modeli (MLP)", "Табличная модель (MLP)")

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

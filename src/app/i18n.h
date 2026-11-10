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
    X(Step1, "Load a text file or a CSV table", "Bir metin dosyası veya CSV tablosu yükle", "Загрузите текстовый файл или таблицу CSV") \
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
    X(NavPlugins, "Plugins", "Eklentiler", "Плагины") \
    X(KeepBest, "Keep the best weights", "En iyi ağırlıkları koru", "Сохранять лучшие веса") \
    X(KeepBestHelp, "When training ends, the model returns to the weights that had the lowest validation loss.", "Eğitim bitince model, doğrulama kaybının en düşük olduğu ağırlıklara geri döner.", "По окончании обучения модель возвращается к весам с наименьшей ошибкой проверки.") \
    X(Patience, "Early stop after (checks)", "Erken durdurma (kontrol)", "Ранняя остановка (проверок)") \
    X(PatienceHelp, "Stop when the validation loss has not improved for this many checks in a row. 0 = never stop early.", "Doğrulama kaybı art arda bu kadar kontrolde iyileşmezse eğitimi durdur. 0 = erken durdurma yok.", "Остановить, если ошибка проверки не улучшалась столько проверок подряд. 0 = не останавливать.") \
    X(TestEvery, "Question tests every (steps)", "Soru testi aralığı (adım)", "Вопросы каждые (шагов)") \
    X(TestEveryHelp, "How often the model is quizzed during training. 0 = only at the end.", "Modelin eğitim sırasında ne sıklıkla sınanacağı. 0 = sadece sonda.", "Как часто модель проверяется вопросами во время обучения. 0 = только в конце.") \
    X(StopOnPass, "Stop when all question tests pass", "Tüm soru testleri geçince dur", "Остановить, когда все вопросы пройдены") \
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
    X(LogHookStop, "stopping training: %s", "eğitim durduruluyor: %s", "обучение останавливается: %s") \
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
    X(TestHelp, "Generate text or run predictions with the current model - even while it trains.", "Mevcut modelle metin üret veya tahmin yap - eğitim sürerken bile.", "Генерируйте текст или получайте предсказания текущей модели - даже во время обучения.") \
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
    X(Latest, "latest: %d / %d", "son: %d / %d", "последний: %d / %d")

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

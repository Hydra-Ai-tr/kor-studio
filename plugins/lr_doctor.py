# kor-plugin
# type: skill
# name: Learning-rate doctor
# name.tr: Öğrenme hızı doktoru
# name.ru: Доктор скорости обучения
# description: Watches the validation loss. When it stops improving the learning rate is lowered; when it suddenly jumps (the run is diverging) the best weights are restored and the rate is cut. With --stop it ends training once lowering the rate no longer helps.
# description.tr: Doğrulama kaybını izler. İyileşme durunca öğrenme hızını düşürür; kayıp aniden fırlarsa (eğitim dağılıyorsa) en iyi ağırlıklara geri döner ve hızı keser. --stop ile hızı düşürmek artık işe yaramadığında eğitimi bitirir.
# description.ru: Следит за ошибкой на проверке. Когда улучшения прекращаются, снижает скорость обучения; при резком скачке (обучение расходится) возвращает лучшие веса и урезает скорость. С --stop завершает обучение, когда снижение скорости уже не помогает.
# args: --patience 2 --factor 0.5 --spike 1.25 --min 0.1
# version: 1.0
# author: Kor Studio

import kor

skill = kor.Skill()


@skill.on_start
def start(run, ev):
    run.state.update(stale=0)
    run.log(run.tr(
        f"Watching: lower the rate after {run.opt('patience', 2)} evaluations without progress",
        f"İzleniyor: {run.opt('patience', 2)} değerlendirme boyunca ilerleme olmazsa hız düşecek",
        f"Слежу: снижу скорость после {run.opt('patience', 2)} проверок без улучшения"))


@skill.on_eval
def evaluate(run, ev):
    patience = run.opt("patience", 2)
    factor = run.opt("factor", 0.5)
    spike = run.opt("spike", 1.25)
    floor = run.opt("min", 0.1)
    s = run.state
    if ev.improved:
        s["stale"] = 0
    elif ev.best_val and ev.val_loss > ev.best_val * spike:
        s["stale"] = 0
        if run.restore_best():
            run.scale_lr(factor)
            run.log(run.tr(
                f"Validation loss jumped to {ev.val_loss:.3f} (best {ev.best_val:.3f}): restored the best weights, rate x{run.lr_factor:.2f}",
                f"Doğrulama kaybı {ev.val_loss:.3f} değerine fırladı (en iyi {ev.best_val:.3f}): en iyi ağırlıklara dönüldü, hız x{run.lr_factor:.2f}",
                f"Ошибка на проверке подскочила до {ev.val_loss:.3f} (лучшая {ev.best_val:.3f}): вернул лучшие веса, скорость x{run.lr_factor:.2f}"),
                "warn")
    else:
        s["stale"] += 1
        if s["stale"] >= patience:
            s["stale"] = 0
            if run.lr_factor * factor < floor:
                if run.opt("stop", False):
                    run.stop(run.tr("lowering the learning rate no longer helps",
                                    "öğrenme hızını düşürmek artık işe yaramıyor",
                                    "снижение скорости больше не помогает"))
            else:
                run.scale_lr(factor)
                run.log(run.tr(
                    f"No progress for {patience} evaluations: learning rate x{run.lr_factor:.2f}",
                    f"{patience} değerlendirmedir ilerleme yok: öğrenme hızı x{run.lr_factor:.2f}",
                    f"Нет улучшения {patience} проверки подряд: скорость x{run.lr_factor:.2f}"))
    run.metric(run.tr("learning-rate factor", "öğrenme hızı çarpanı", "множитель скорости"), run.lr_factor)


if __name__ == "__main__":
    skill.run()

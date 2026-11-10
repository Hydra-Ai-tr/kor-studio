"""kor - the Python library for Kor Studio plugins.

Two kinds of plugins use it:

* Skills run during training and steer it live (learning rate, extra training
  text, early stopping, best weights, samples, custom metrics)::

      import kor
      skill = kor.Skill()

      @skill.on_eval
      def watch(run, ev):
          if not ev.improved:
              run.scale_lr(0.7)

      if __name__ == "__main__":
          skill.run()

* Tools run once from the Plugins page and prepare data or tests::

      import kor
      text = kor.read_data()
      kor.write_data(text.lower())

Sharing: ``python -m kor install <link>``, ``python -m kor search``,
``python -m kor new "My skill"``. See docs/plugins.md.
"""

__version__ = "0.4.0"

from .skill import Answer, Event, Run, Skill, SkillError, TestCase  # noqa: F401
from .tool import (  # noqa: F401
    TestResult,
    ask,
    cli,
    cli_path,
    data_path,
    format_tests,
    generate,
    lang,
    model_path,
    out_dir,
    parse_tests,
    read_data,
    read_tests,
    run_tests,
    tr,
    write_data,
    write_tests,
)

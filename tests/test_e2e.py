import os

from neptune import init_run

from src.neptune_detectron2 import NeptuneHook
from tests.utils import get_images


def test_e2e(cfg, trainer):

    run = init_run()

    get_images()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer.resume_or_load(resume=False)

    hook = NeptuneHook(run=run, log_checkpoints=True, log_model=True)
    trainer.register_hooks([hook])
    trainer.train()

    assert run.exists("training/config")

    assert run.exists("training/model/checkpoints/checkpoint_iter_0")

    assert run.exists("training/model/checkpoints/checkpoint_final")

    assert isinstance(run["training/model/summary"].fetch(), str)

    cls_accuracy_vals = run["training/metrics/fast_rcnn/cls_accuracy"].fetch_values()
    assert 0 < cls_accuracy_vals.iloc[-1]["value"] <= 1
    run.stop()

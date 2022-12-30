import os

import neptune.new as neptune

from src.neptune_detectron2 import NeptuneHook
from tests.utils import get_images


def test_e2e(cfg, trainer):

    run = neptune.init_run()
    run_id = run["sys/id"].fetch()

    get_images()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer.resume_or_load(resume=False)

    hook = NeptuneHook(run=run, log_checkpoints=True, log_model=True)
    trainer.register_hooks([hook])
    trainer.train()

    npt_run = neptune.init_run(with_id=run_id)

    assert npt_run.exists("training/config")

    assert npt_run.exists("model/checkpoints/checkpoint_iter_0")

    assert npt_run.exists("model/checkpoints/checkpoint_final")

    assert isinstance(npt_run["training/model/summary"].fetch(), str)

    cls_accuracy_vals = npt_run["training/metrics/fast_rcnn/cls_accuracy"].fetch_values()
    assert 0 < cls_accuracy_vals.iloc[-1]["value"] <= 1

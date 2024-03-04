#
# Copyright (c) 2022, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This module contains the NeptuneHook class.

The class is used for automatic metadata logging to Neptune, during training and validation of detectron2 models."""

from __future__ import annotations

__all__ = [
    "__version__",
    "NeptuneHook",
]

import os
import warnings

import detectron2
from detectron2.checkpoint import Checkpointer
from detectron2.engine import hooks

from neptune_detectron2.impl.version import __version__

try:
    from neptune import Run
    from neptune.handler import Handler
    from neptune.internal.utils import verify_type
    from neptune.types import File
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new.metadata_containers import Run
    from neptune.new.handler import Handler
    from neptune.new.internal.utils import verify_type
    from neptune.new.types import File
    from neptune.new.utils import stringify_unsupported

from torch.nn import Module

INTEGRATION_VERSION_KEY = "source_code/integrations/detectron2"


class NeptuneHook(hooks.HookBase):
    """Hook implementation that sends the logs to Neptune.

    Args:
        run: Neptune run object. You can also pass a namespace handler object;
            for example, run["test"], in which case all metadata is logged under
            the "test" namespace inside the run.
        base_namespace: Root namespace where all metadata logged by the hook is stored.
        metrics_update_freq: How often NeptuneHook should log metrics (and checkpoints, if
            log_checkpoints is set to True). The value must be greater than zero.
            Example: Setting metrics_update_freq=10 will log metrics on every 10th iteration.
        log_model: Whether to upload the final model checkpoint, whenever it is saved by the Trainer.
            Expects CheckpointHook to be present.
        log_checkpoints: Whether to upload checkpoints whenever they are saved by the Trainer.
            Expects CheckpointHook to be present.

    Example:
        import neptune
        neptune_run = neptune.init_run()
        neptune_hook = NeptuneHook(
            run=neptune_run,
            log_checkpoints=True,  # Log model checkpoints
            metrics_update_freq=10,  # Upload metrics and checkpoints every 10th epoch
        )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/detectron2/
        API reference: https://docs.neptune.ai/api/integrations/detectron2/
    """

    def __init__(
        self,
        run: Run | Handler,
        *,
        base_namespace: str = "training",
        metrics_update_freq: int = 20,
        log_model: bool = False,
        log_checkpoints: bool = False,
    ):
        verify_type("run", run, (Run, Handler))

        self._run = run

        self._metrics_update_freq = metrics_update_freq
        self.log_model = log_model
        self.log_checkpoints = log_checkpoints

        self._verify_metrics_update_freq()

        if base_namespace.endswith("/"):
            base_namespace = base_namespace[:-1]

        self.base_handler = self._run[base_namespace]

        self._root_object = self._run.get_root_object() if isinstance(self._run, Handler) else self._run

    def _verify_metrics_update_freq(self) -> None:
        if not isinstance(self._metrics_update_freq, int):
            raise TypeError(
                f"metrics_update_freq should be of type int. Got {type(self._metrics_update_freq)} instead."
            )
        if self._metrics_update_freq <= 0:
            raise ValueError(f"metrics_update_freq should be greater than 0. Got {self._metrics_update_freq}.")

    def _log_integration_version(self) -> None:
        self._root_object[INTEGRATION_VERSION_KEY] = detectron2.__version__

    def _log_config(self) -> None:
        if hasattr(self.trainer, "cfg") and isinstance(self.trainer.cfg, dict):
            self.base_handler["config"] = stringify_unsupported(self.trainer.cfg)

    def _log_model(self) -> None:
        if hasattr(self.trainer, "model") and isinstance(self.trainer.model, Module):
            self.base_handler["model/summary"] = str(self.trainer.model)

    def _log_checkpoint(self, final: bool = False) -> None:
        if not self._can_save_checkpoint():
            warnings.warn("Checkpointer not present for the current trainer.")
            return

        self.trainer.checkpointer.save(f"neptune_iter_{self.trainer.iter}")
        neptune_model_path = "model/checkpoints/checkpoint_{}"

        neptune_model_path = neptune_model_path.format("final" if final else f"iter_{self.trainer.iter}")

        checkpoint_path = self.trainer.checkpointer.get_checkpoint_file()

        with open(checkpoint_path, "rb") as fp:
            self.base_handler[neptune_model_path] = File.from_stream(fp)
        os.remove(checkpoint_path)

    def _log_metrics(self) -> None:
        storage = detectron2.utils.events.get_event_storage()
        for k, (v, _) in storage.latest_with_smoothing_hint(self._metrics_update_freq).items():
            self.base_handler[f"metrics/{k}"].append(v)

    def _can_save_checkpoint(self) -> bool:
        return hasattr(self.trainer, "checkpointer") and isinstance(self.trainer.checkpointer, Checkpointer)

    def _should_perform_after_step(self) -> bool:
        return self.trainer.iter % self._metrics_update_freq == 0

    def before_train(self) -> None:
        """Logs detectron2 version used, the config that the trainer uses, and the underlying model summary."""
        self._log_integration_version()
        self._log_config()
        self._log_model()

    def after_step(self) -> None:
        """Logs metrics after step and optionally the model checkpoint."""
        if not self._should_perform_after_step():
            return

        self._log_metrics()

        if self.log_checkpoints:
            self._log_checkpoint()

    def after_train(self) -> None:
        """Optionally saves the final model checkpoint. Syncs the run."""
        if self.log_model:
            self._log_checkpoint(final=True)

        self._root_object.sync()

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

__all__ = [
    "__version__",
    "NeptuneHook",
]

import os
import warnings
from typing import (
    Any,
    Optional,
)

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type

except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type

import detectron2
from detectron2.checkpoint import Checkpointer
from detectron2.engine import hooks
from neptune.new.metadata_containers import Run
from neptune.new.types import File
from torch.nn import Module

from neptune_detectron2.impl.version import __version__

INTEGRATION_VERSION_KEY = "source_code/integrations/detectron2"


class NeptuneHook(hooks.HookBase):
    """Hook implementation that sends the logs to Neptune.

    Args:
        run: Pass a Neptune run object if you want to continue logging to an existing run.
            Learn more about resuming runs in the docs: https://docs.neptune.ai/logging/to_existing_object
        base_namespace: In the Neptune run, the root namespace that will contain all the logged metadata.
        smoothing_window_size: How often NeptuneHook should log metrics (and checkpoints, if
            log_checkpoints is set to True). The value must be greater than zero.
            Example: Setting smoothing_window_size=10 will log metrics on every 10th epoch.
        log_model: Whether to upload the final model checkpoint, whenever it is saved by the Trainer.
            Expects CheckpointHook to be present.
        log_checkpoints: Whether to upload checkpoints whenever they are saved by the Trainer.
            Expects CheckpointHook to be present.
        **kwargs (optional):
            Additional keyword arguments to be passed directly to the neptune.init_run() function when a new run is
            created. For details, see the docs: https://docs.neptune.ai/api/neptune/#init_run

    Examples:

        Creating a hook that logs the metadata to a new Neptune run, with optional arguments:

            neptune_hook = NeptuneHook(
                log_checkpoints=True,  # Log model checkpoints
                smoothing_window_size=10,  # Upload metrics and checkpoints every 10th epoch
                capture_stdout=False,  # Don't capture standard out stream (kwarg for the Neptune run)
            )

        Creating a hook that sends the logs to an existing Neptune run object:

            import neptune.new as neptune
            neptune_run = neptune.init_run()
            neptune_hook = NeptuneHook(run=neptune_run)
    """

    def __init__(
        self,
        *,
        run: Optional[Run] = None,
        base_namespace: str = "training",
        smoothing_window_size: int = 20,
        log_model: bool = False,
        log_checkpoints: bool = False,
        **kwargs: Any,
    ):
        self._window_size = smoothing_window_size
        self.log_model = log_model
        self.log_checkpoints = log_checkpoints

        self._verify_window_size()

        self._run = neptune.init_run(**kwargs) if not isinstance(run, Run) else run

        verify_type("run", self._run, Run)

        if base_namespace.endswith("/"):
            self._base_namespace = base_namespace[:-1]

        self.base_handler = self._run[base_namespace]

    def _verify_window_size(self) -> None:
        if self._window_size <= 0:
            raise ValueError(f"Update freq should be greater than 0. Got {self._window_size}.")
        if not isinstance(self._window_size, int):
            raise TypeError(f"Smoothing window size should be of type int. Got {type(self._window_size)} instead.")

    def _log_integration_version(self) -> None:
        self.base_handler[INTEGRATION_VERSION_KEY] = detectron2.__version__

    def _log_config(self) -> None:
        if hasattr(self.trainer, "cfg") and isinstance(self.trainer.cfg, dict):
            self.base_handler["config"] = self.trainer.cfg

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
            self._run[neptune_model_path] = File.from_stream(fp)
        os.remove(checkpoint_path)

    def _log_metrics(self) -> None:
        storage = detectron2.utils.events.get_event_storage()
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            self.base_handler[f"metrics/{k}"].log(v)

    def _can_save_checkpoint(self) -> bool:
        return hasattr(self.trainer, "checkpointer") and isinstance(self.trainer.checkpointer, Checkpointer)

    def _should_perform_after_step(self) -> bool:
        return self.trainer.iter % self._window_size == 0

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
        """Optionally saves the final model checkpoint. Syncs the run and stops it."""
        if self.log_model:
            self._log_checkpoint(final=True)

        self._run.sync()
        self._run.stop()

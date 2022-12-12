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

__all__ = [
    "__version__",
    "NeptuneHook",
]

import warnings
from typing import (
    Any,
    Optional,
)

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
    from neptune.new.internal.utils import verify_type
    from neptune.new.internal.utils.compatibility import expect_not_an_experiment

except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type
    from neptune.internal.utils.compatibility import expect_not_an_experiment

import detectron2
from detectron2.engine import hooks
from neptune.new.metadata_containers import Run

from neptune_detectron2.impl.version import __version__

INTEGRATION_VERSION_KEY = "source_code/integrations/detectron2"


# TODO: Implementation of neptune-detectron2 here
class NeptuneHook(hooks.HookBase):
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

        if self._window_size <= 0:
            raise ValueError(f"Update freq should be greater than 0. Got {self._window_size}.")
        if not isinstance(self._window_size, int):
            raise TypeError(f"Smoothing window size should be of type int. Got {type(self._window_size)} instead.")

        self._run = neptune.init_run(**kwargs) if not isinstance(run, Run) else run

        if base_namespace.endswith("/"):
            self._base_namespace = base_namespace[:-1]
        self.base_handler = self._run[base_namespace]

        self.base_handler[INTEGRATION_VERSION_KEY] = detectron2.__version__
        verify_type("run", self._run, Run)
        expect_not_an_experiment(self._run)

    def _log_checkpoint(self, final: bool = False) -> None:
        if self.trainer.checkpointer is None:
            warnings.warn("Checkpointer not present for the current trainer.")
            return

        path = "model/checkpoints/checkpoint_{}"
        if final:
            path = path.format("final")
        else:
            path = path.format(f"iter_{self.trainer.iter}")

        if self.trainer.checkpointer.has_checkpoint():
            self._run[path].upload(self.trainer.checkpointer.get_checkpoint_file())
        else:
            warnings.warn(f"Checkpoints do not exist at target directory {self.trainer.checkpointer.save_dir}.")

    def before_train(self) -> None:
        self.base_handler["config"] = self.trainer.cfg
        self.base_handler["model/summary"] = str(self.trainer.model)

    def after_step(self) -> None:
        if self.trainer.iter % self._window_size != 0:
            return

        storage = detectron2.utils.events.get_event_storage()
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            self.base_handler[f"metrics/{k}"].log(v)

        if self.log_checkpoints:
            self._log_checkpoint()

    def after_train(self) -> None:
        self._log_checkpoint(final=True)
        self._run.stop()

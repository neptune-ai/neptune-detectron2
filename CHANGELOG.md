## neptune-detectron2 0.2.0

### Fixes
- The NeptuneHook doesn't stop the run after training, just syncs. ([#7](https://github.com/neptune-ai/neptune-detectron2/pull/7))
- Passing a run or handler object is now compulsory - the hook will not create one automatically. ([#7](https://github.com/neptune-ai/neptune-detectron2/pull/7))

### Changes
- Updated the integration for compatibility with `neptune-client` `1.0.0`. ([#6](https://github.com/neptune-ai/neptune-detectron2/pull/6))

## neptune_detectron2 0.1.0

### Fixes
- Removed `expect_not_an_experiment` that was causing errors ([#4](https://github.com/neptune-ai/neptune-detectron2/pull/4))

### Features
- Create `NeptuneHook` for logging metadata ([#1](https://github.com/neptune-ai/neptune-detectron2/pull/1))

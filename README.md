# Neptune - Detectron2 integration

Experiment tracking for Detectron2-trained models.

## What will you get with this integration?

* Log, organize, visualize, and compare ML experiments in a single place
* Monitor model training live
* Version and query production-ready models and associated metadata (e.g., datasets)
* Collaborate with the team and across the organization

## What will be logged to Neptune?

* Model configuration,
* Training code and Git information,
* System metrics and hardware consumption,
* [Other metadata](https://docs.neptune.ai/logging/what_you_can_log)

![image](https://neptune.ai/wp-content/uploads/2023/07/detectron2-integration-dashboard.jpg)

## Resources

* [Documentation](https://docs.neptune.ai/integrations/detectron2/)
* [Code example on GitHub](https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/detectron2)
* [Example project in the Neptune app](https://neptune.ai/resources/detectron2-integration-example)

## Example

In the following example, we set the Trainer to save model checkpoints every 10th epoch. Neptune will upload those checkpoints and metrics at the same interval.


```python
neptune_run = neptune.init_run(
    project="workspace-name/project-name",
    name="My detectron2 run",
    tags = ["validation"],
    capture_stdout=False,
)

neptune_hook = NeptuneHook(
    run=neptune_run,
    log_checkpoints=True,
    metrics_update_freq=10,
)

```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting_help).
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! In the Neptune app, click the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP).
* You can just shoot us an email at [support@neptune.ai](mailto:support@neptune.ai).

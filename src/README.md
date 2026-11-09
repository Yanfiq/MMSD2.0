# MMSD2.0 Training Source

This directory contains the training, evaluation, prediction, dataset, and model
code for the MMSD2.0 experiment.

## Recent training changes

The training workflow was updated to make experiment tracking more complete and
to prevent the test set from influencing model selection.

### Metrics logged after every epoch

`train.py` now calculates and sends the following metrics to Weights & Biases
(W&B) for both the training and development sets after every epoch:

- Loss
- Accuracy
- F1, precision, and recall for the positive class
- Macro F1, macro precision, and macro recall across classes `0` and `1`

Metrics use the `train/*` and `dev/*` namespaces, such as `train/loss`,
`train/macro_f1`, `dev/accuracy`, and `dev/macro_f1`. The corresponding epoch
number is included in the same W&B log operation. Console logging also reports
the main training and development results for each epoch.

Training metrics are collected from predictions made during normal training.
They are therefore monitoring metrics: dropout remains active, and batches near
the end of an epoch use a newer model state than batches near the beginning.

### Best-model checkpoint and W&B artifact

Development accuracy remains the model-selection metric. Whenever it exceeds
the best accuracy from earlier epochs, training performs the following actions:

1. Saves the model state to `<output_dir>/<model>/model.pt`.
2. Creates a W&B model artifact named `<model>-checkpoint`.
3. Adds the checkpoint to the artifact as `model.pt`.
4. Records the epoch and all development metrics in the artifact metadata.

Using the same artifact name lets W&B maintain successive improved checkpoints
as artifact versions.

### Test evaluation only after training

The test set is no longer evaluated whenever development accuracy improves.
After all epochs finish, the workflow reloads the best saved checkpoint and
evaluates the test set exactly once. All test metrics are then logged under the
`test/*` namespace.

The W&B run summary also records:

- `best_epoch`
- `best_dev_accuracy`

This separation keeps checkpoint selection based on development data and
reserves the test data for the final performance estimate.

### W&B run configuration

`main.py` defines `epoch` as the step metric for `train/*` and `dev/*`, allowing
their charts to share the correct epoch axis. It also calls `wandb.finish()`
after training to flush uploads and close the run cleanly.

The pinned W&B version in `requirements.txt` was updated from `0.13.3` to
`0.22.3`.

## Resulting training flow

```text
initialize model and W&B
        |
        v
train one epoch
        |
        v
calculate and log train + development metrics
        |
        v
development accuracy improved? -- yes --> save checkpoint + W&B artifact
        |
        no
        |
        v
repeat until all epochs finish
        |
        v
load best checkpoint
        |
        v
evaluate and log the test set once
        |
        v
finish the W&B run
```

import functools
import gc
import random

import gin
import inputs
import models
import numpy as np
import predictors
import trax
from trax import fastmath, optimizers
from trax.layers import base
from trax.supervised import callbacks as tc
from trax.supervised import lr_schedules, training
from trax import layers as tl
import time

@gin.configurable(module='code.trainer')
def num_devices(value=1):
  """Returns how many devices to use (if None, default, use all available)."""
  return value


@gin.configurable(module='code.trainer')
class SaveCheckpointCallback(tc.TrainingStepCallback):

    def __init__(
            self,
            loop,
            log_every=500,
            filename=None):
        super().__init__(loop)

        self._log_every = log_every
        self._filename = filename if filename else 'model'

    def call_at(self, step):
        return step > 0 and step % self._log_every == 0

    def on_step_begin(self, step):
        pass

    def on_step_end(self, step):
        self._loop.save_checkpoint(
            f'{self._filename}_{step}')


@gin.configurable(module='code.trainer')
def train(
    output_dir,
    inputs=inputs.CreateInputs,
    model_body=models.TransformerBody,
    predictor_class=predictors.SerialPredictor,
    optimizer=optimizers.Adam,
    lr_schedule=lr_schedules.multifactor,
    extra_callbacks=[],
    n_steps=10000,
    eval_every=500,
    n_eval_batches=None,
    seed=None,
    calc_eval_loss=True,
):
    """Trains a time series model.

    Args:
        output_dir: Directory where to put the logs and checkpoints.
        inputs: Is a callable that returns `ti.Inputs` (e.g., `CreateInputs`)
        model_body: Function mode -> model returning the body of a Trax
            sequence decoder model, e.g. code.models.TransformerBody, with
            input shape [batch_size, n_timesteps, d_in] and output shape
            [..., d_out].
        predictor: Time series predictor class, see code.predictors.
        optimizer: Optimizer class, see trax.optimizers.
        lr_schedule: Learning rate schedule, see trax.supervised.lr_schedules.
        extra_callbacks: List of training callbacks, see
            trax.supervised.callbacks.
        n_steps: Number of steps to train for.
        eval_every: How often to run supervised evaluation.
        n_eval_batches: Number of batches to run during supervised evaluation.
        seed: the random seed to use; time/os dependent if ``None`` (default).
    """

    base.N_WEIGHTS_SHARDS = 1
    n_devices = num_devices() or fastmath.local_device_count()
    # Set seed.
    np.random.seed(seed)
    random.seed(seed)

    print("\n1. Building Predictor...")
    start = time.perf_counter()
    predictor = predictor_class(
        model_body_fn=model_body,
        accelerate_predict_model=True,
    )

    # `inputs` is an callable returning an instance of `ti.Inputs`.
    predictor.before_training(inputs)
    pred_time = time.perf_counter() - start
    print(f" Predictor: {pred_time/100*1000:.2f}ms per call")
    
    inputs = inputs()

    callbacks = []
    eval_tasks = []

    print("\n2. Called Callback...")
    start = time.perf_counter()
    if calc_eval_loss:
        loss_fn = predictor.make_loss()
        eval_tasks.append(
            trax.supervised.training.EvalTask(
                inputs.eval_stream(n_devices),
                metrics=[loss_fn],
                metric_names=[loss_fn._name],
                n_eval_batches=n_eval_batches))
    pred_time = time.perf_counter() - start
    print(f" Called Callback: {pred_time/100*1000:.2f}ms per call")
    # Save checkpoint callback.
    
    start = time.perf_counter()    
    callbacks.append(SaveCheckpointCallback)
    for callback in extra_callbacks:
        # Process Neptune callback.
        if callback.__name__ == 'NeptuneCallback':
            callbacks.append(
                functools.partial(
                    callback,
                    n_steps=n_steps,
                )
            )
        # Process remaining extra callbacks.
        else:
            callbacks.append(callback)
    pred_time = time.perf_counter() - start
    print(f" Saved Callback: {pred_time/100*1000:.2f}ms per call")    
    # Setup the model.
    start = time.perf_counter()  
    model_train = tl.Accelerate(predictor.make_train_eval_model(mode='train'))
    model_predict_eval = tl.Accelerate(predictor.make_train_eval_model(mode='eval'))
    pred_time = time.perf_counter() - start
    print(f" Created Model: {pred_time/100*1000:.2f}ms per call")   
    
    start = time.perf_counter()     
    # Setup input stream
    train_stream_instance = inputs.train_stream(n_devices)
    optimizer_instance = optimizer()
    pred_time = time.perf_counter() - start
    print(f" Setup Stream: {pred_time/100*1000:.2f}ms per call") 
    # Parts of code below are copied from trax's `training.train_lib.py`.
    # Prepare the training task.
    start = time.perf_counter()  
    train_task = training.TrainTask(
        train_stream_instance,
        loss_layer=predictor.make_loss(),
        optimizer=optimizer_instance,
        lr_schedule=lr_schedule(),
        n_steps_per_checkpoint=eval_every,
        n_steps_per_permanent_checkpoint=None)
    pred_time = time.perf_counter() - start
    print(f" Setup Train Task: {pred_time/100*1000:.2f}ms per call") 
    
    start = time.perf_counter() 
    loop = trax.supervised.training.Loop(
        model=model_train,
        tasks=[train_task],
        eval_model=model_predict_eval,
        eval_tasks=eval_tasks,  # Do not set to None to be able to set eval_at.
        output_dir=output_dir,
        checkpoint_at=lambda step: False,  # Use callback for checkpointing.
        permanent_checkpoint_at=lambda step: False,  # Use callback for checkpointing.
        n_devices=n_devices,
        loss_chunk_size=0,  # Default value.
        use_memory_efficient_trainer=False,  # Default value.
        adasum=False,  # Default value.
        random_seed=np.random.randint(int(1e6)),
        callbacks=callbacks,
        eval_at=lambda step: (step == 1 or step % eval_every == 0),
    )
    pred_time = time.perf_counter() - start
    print(f" Setup Loop: {pred_time/100*1000:.2f}ms per call") 
    # Train and return the loop.
    start = time.perf_counter()
    loop.run(n_steps)
    pred_time = time.perf_counter() - start
    print(f" Ran loop for {n_steps} steps: {pred_time/100*1000:.2f}ms per call") 
    n = gc.collect()

    return loop, predictor

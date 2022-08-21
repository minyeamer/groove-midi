"""MusicVAE script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from magenta.models.music_vae import configs
from magenta.models.music_vae import data
from magenta.models.music_vae import TrainedModel
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim

logging = tf.logging


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, examples_path, output_dir):
  """Writes a tensorboard text summary of the trial."""

  examples_path_summary = tf.summary.text(
      'examples_path', tf.constant(examples_path, name='examples_path'),
      collections=[])

  hparams_dict = hparams.values()

  # Create a markdown table from hparams.
  header = '| Key | Value |\n| :--- | :--- |\n'
  keys = sorted(hparams_dict.keys())
  lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
  hparams_table = header + '\n'.join(lines) + '\n'

  hparam_summary = tf.summary.text(
      'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
    writer.add_summary(examples_path_summary.eval())
    writer.add_summary(hparam_summary.eval())
    writer.close()


def _get_input_tensors(dataset, config):
  """Get input tensors from dataset."""
  batch_size = config.hparams.batch_size
  iterator = tf.data.make_one_shot_iterator(dataset)
  (input_sequence, output_sequence, control_sequence,
   sequence_length) = iterator.get_next()
  input_sequence.set_shape(
      [batch_size, None, config.data_converter.input_depth])
  output_sequence.set_shape(
      [batch_size, None, config.data_converter.output_depth])
  if not config.data_converter.control_depth:
    control_sequence = None
  else:
    control_sequence.set_shape(
        [batch_size, None, config.data_converter.control_depth])
  sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())

  return {
      'input_sequence': input_sequence,
      'output_sequence': output_sequence,
      'control_sequence': control_sequence,
      'sequence_length': sequence_length
  }


def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def train(train_dir,
          config,
          dataset_fn,
          checkpoints_to_keep=5,
          keep_checkpoint_every_n_hours=1,
          num_steps=None,
          master='',
          num_sync_workers=0,
          num_ps_tasks=0,
          task=0):
  """Train loop."""
  tf.gfile.MakeDirs(train_dir)
  is_chief = (task == 0)
  if is_chief:
    _trial_summary(
        config.hparams, config.train_examples_path or config.tfds_name,
        train_dir)
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(
        num_ps_tasks, merge_devices=True)):

      model = config.model
      model.build(config.hparams,
                  config.data_converter.output_depth,
                  is_training=True)

      optimizer = model.train(**_get_input_tensors(dataset_fn(), config))

      hooks = []
      if num_sync_workers:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            num_sync_workers)
        hooks.append(optimizer.make_session_run_hook(is_chief))

      grads, var_list = list(zip(*optimizer.compute_gradients(model.loss)))
      global_norm = tf.global_norm(grads)
      tf.summary.scalar('global_norm', global_norm)

      if config.hparams.clip_mode == 'value':
        g = config.hparams.grad_clip
        clipped_grads = [tf.clip_by_value(grad, -g, g) for grad in grads]
      elif config.hparams.clip_mode == 'global_norm':
        clipped_grads = tf.cond(
            global_norm < config.hparams.grad_norm_clip_to_zero,
            lambda: tf.clip_by_global_norm(  # pylint:disable=g-long-lambda
                grads, config.hparams.grad_clip, use_norm=global_norm)[0],
            lambda: [tf.zeros(tf.shape(g)) for g in grads])
      else:
        raise ValueError(
            'Unknown clip_mode: {}'.format(config.hparams.clip_mode))
      train_op = optimizer.apply_gradients(
          list(zip(clipped_grads, var_list)),
          global_step=model.global_step,
          name='train_step')

      logging_dict = {'global_step': model.global_step,
                      'loss': model.loss}

      hooks.append(tf.train.LoggingTensorHook(logging_dict, every_n_iter=100))
      if num_steps:
        hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

      scaffold = tf.train.Scaffold(
          saver=tf.train.Saver(
              max_to_keep=checkpoints_to_keep,
              keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))
      tf_slim.training.train(
          train_op=train_op,
          logdir=train_dir,
          scaffold=scaffold,
          hooks=hooks,
          save_checkpoint_secs=60,
          master=master,
          is_chief=is_chief)


def evaluate(train_dir,
             eval_dir,
             config,
             dataset_fn,
             num_batches,
             master=''):
  """Evaluate the model repeatedly."""
  tf.gfile.MakeDirs(eval_dir)

  _trial_summary(
      config.hparams, config.eval_examples_path or config.tfds_name, eval_dir)
  with tf.Graph().as_default():
    model = config.model
    model.build(config.hparams,
                config.data_converter.output_depth,
                is_training=False)

    eval_op = model.eval(
        **_get_input_tensors(dataset_fn().take(num_batches), config))

    hooks = [
        tf_slim.evaluation.StopAfterNEvalsHook(num_batches),
        tf_slim.evaluation.SummaryAtEndHook(eval_dir)
    ]
    tf_slim.evaluation.evaluate_repeatedly(
        train_dir,
        eval_ops=eval_op,
        hooks=hooks,
        eval_interval_secs=60,
        master=master)


def generate(config,
             flags,
             master=''):
  """Generate outputs."""
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  config.data_converter.max_tensors_per_item = None

  if flags.mode == 'interpolate':
    if flags.input_midi_1 is None or flags.input_midi_2 is None:
      raise ValueError(
          '`--input_midi_1` and `--input_midi_2` must be specified in '
          '`interpolate` mode.')
    input_midi_1 = os.path.expanduser(flags.input_midi_1)
    input_midi_2 = os.path.expanduser(flags.input_midi_2)
    if not os.path.exists(input_midi_1):
      raise ValueError('Input MIDI 1 not found: %s' % flags.input_midi_1)
    if not os.path.exists(input_midi_2):
      raise ValueError('Input MIDI 2 not found: %s' % flags.input_midi_2)
    input_1 = note_seq.midi_file_to_note_sequence(input_midi_1)
    input_2 = note_seq.midi_file_to_note_sequence(input_midi_2)

    def _check_extract_examples(input_ns, path, input_number):
      """Make sure each input returns exactly one example from the converter."""
      tensors = config.data_converter.to_tensors(input_ns).outputs
      if not tensors:
        print(
            'MusicVAE configs have very specific input requirements. Could not '
            'extract any valid inputs from `%s`. Try another MIDI file.' % path)
        sys.exit()
      elif len(tensors) > 1:
        basename = os.path.join(
            flags.output_dir,
            '%s_input%d-extractions_%s-*-of-%03d.mid' %
            (flags.config, input_number, date_and_time, len(tensors)))
        for i, ns in enumerate(config.data_converter.from_tensors(tensors)):
          note_seq.sequence_proto_to_midi_file(
              ns, basename.replace('*', '%03d' % i))
        print(
            '%d valid inputs extracted from `%s`. Outputting these potential '
            'inputs as `%s`. Call script again with one of these instead.' %
            (len(tensors), path, basename))
        sys.exit()
    logging.info(
        'Attempting to extract examples from input MIDIs using config `%s`...',
        flags.config)
    _check_extract_examples(input_1, flags.input_midi_1, 1)
    _check_extract_examples(input_2, flags.input_midi_2, 2)

  logging.info('Loading model...')
  if flags.run_dir:
    checkpoint_dir_or_path = os.path.expanduser(
        os.path.join(flags.run_dir, 'train'))
  else:
    checkpoint_dir_or_path = os.path.expanduser(flags.checkpoint_file)
  model = TrainedModel(
      config, batch_size=min(flags.max_batch_size, flags.num_outputs),
      checkpoint_dir_or_path=checkpoint_dir_or_path)

  if flags.mode == 'interpolate':
    logging.info('Interpolating...')
    _, mu, _ = model.encode([input_1, input_2])
    z = np.array([
        _slerp(mu[0], mu[1], t) for t in np.linspace(0, 1, flags.num_outputs)])
    results = model.decode(
        length=config.hparams.max_seq_len,
        z=z,
        temperature=flags.temperature)
  elif flags.mode == 'sample':
    logging.info('Sampling...')
    results = model.sample(
        n=flags.num_outputs,
        length=config.hparams.max_seq_len,
        temperature=flags.temperature)

  basename = os.path.join(
      flags.output_dir,
      '%s_%s_%s-*-of-%03d.mid' %
      (flags.config, flags.mode, date_and_time, flags.num_outputs))
  logging.info('Outputting %d files as `%s`...', flags.num_outputs, basename)
  for i, ns in enumerate(results):
    note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

  logging.info('Done.')

import os
import json
from pathlib import Path
from collections import OrderedDict

from magenta.common import merge_hparams
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import configs
from magenta.models.music_vae import data
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE
from music_vae import train, evaluate, generate
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'master', '',
    'The TensorFlow master to use.')
flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of NoteSequence examples. Overrides the config.')
flags.DEFINE_string(
    'tfds_name', None,
    'TensorFlow Datasets dataset name to use. Overrides the config.')
flags.DEFINE_string(
    'run_dir', None,
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
flags.DEFINE_integer(
    'num_steps', 200000,
    'Number of training steps or `None` for infinite.')
flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches to use during evaluation or `None` for all batches '
    'in the data source.')
flags.DEFINE_integer(
    'checkpoints_to_keep', 100,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
flags.DEFINE_integer(
    'keep_checkpoint_every_n_hours', 1,
    'In addition to checkpoints_to_keep, keep a checkpoint every N hours.')
flags.DEFINE_string(
    'mode', 'train',
    'Which mode to use (`train`, `eval`, `sample`, `interpolate`).')
flags.DEFINE_string(
    'config', '',
    'The name of the config to use.')
flags.DEFINE_string(
    'config_file', '',
    'Path to a custom config file')
flags.DEFINE_string(
    'hparams', '',
    'A comma-separated list of `name=value` hyperparameter values to merge '
    'with those in the config.')
flags.DEFINE_bool(
    'cache_dataset', True,
    'Whether to cache the dataset in memory for improved training speed. May '
    'cause memory errors for very large datasets.')
flags.DEFINE_integer(
    'task', 0,
    'The task number for this worker.')
flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter server tasks.')
flags.DEFINE_integer(
    'num_sync_workers', 0,
    'The number of synchronized workers.')
flags.DEFINE_string(
    'eval_dir_suffix', '',
    'Suffix to add to eval output directory.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'input_midi_1', None,
    'Path of start MIDI file for interpolation.')
flags.DEFINE_string(
    'input_midi_2', None,
    'Path of end MIDI file for interpolation.')
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` mode, the number of samples to produce. In `interpolate` '
    'mode, the number of steps (including the endpoints).')
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
    'temperature', 0.5,
    'The randomness of the decoding process.')

HParams = contrib_training.HParams


def read_json(fname):
  fname = Path(fname)
  with fname.open('rt') as handle:
    return json.load(handle, object_hook=OrderedDict)


def get_custom_config(config_dict):
  for key, values in config_dict.items():
    values['model'] = eval(f"MusicVAE({','.join([''.join(scripts) for scripts in values['model'].values()])})")
    values['hparams'] = merge_hparams(
                    lstm_models.get_default_hparams(),
                    HParams(**values['hparams']))
    values['data_converter'] = eval(
      ''.join([f"{converter}({','.join([f'{key}={value}' for key,value in params.items()])})"
        for converter,params in values['data_converter'].items()]))
    config_dict[key] = configs.Config(**values)
  return config_dict


def run(config_map,
        tf_file_reader=tf.data.TFRecordDataset,
        file_reader=tf.python_io.tf_record_iterator):
  """Load model params, save config file and start trainer.

  Args:
    config_map: Dictionary mapping configuration name to Config object.
    tf_file_reader: The tf.data.Dataset class to use for reading files.
    file_reader: The Python reader to use for reading files.

  Raises:
    ValueError: if required flags are missing or invalid.
  """
  if FLAGS.mode not in ['train', 'eval', 'sample', 'interpolate']:
    raise ValueError('Invalid mode: %s' % FLAGS.mode)

  if FLAGS.config not in config_map:
    try:
      if Path(FLAGS.config_file).exists():
        custom_config = get_custom_config(read_json(FLAGS.config_file))
        config_map.update(custom_config)
      if FLAGS.config not in config_map:
        raise AttributeError()
    except AttributeError:
      raise ValueError('Invalid config: %s' % FLAGS.config)
  config = config_map[FLAGS.config]

  if FLAGS.mode in ['train', 'eval']:
    if not FLAGS.run_dir:
      raise ValueError('Invalid run directory: %s' % FLAGS.run_dir)
    run_dir = os.path.expanduser(FLAGS.run_dir)
    train_dir = os.path.join(run_dir, 'train')

    if FLAGS.hparams:
      config.hparams.parse(FLAGS.hparams)
    config_update_map = {}
    if FLAGS.examples_path:
      config_update_map['%s_examples_path' % FLAGS.mode] = os.path.expanduser(
          FLAGS.examples_path)
    if FLAGS.tfds_name:
      if FLAGS.examples_path:
        raise ValueError(
            'At most one of --examples_path and --tfds_name can be set.')
      config_update_map['tfds_name'] = FLAGS.tfds_name
      config_update_map['eval_examples_path'] = None
      config_update_map['train_examples_path'] = None
    config = configs.update_config(config, config_update_map)
    if FLAGS.num_sync_workers:
      config.hparams.batch_size //= FLAGS.num_sync_workers

    is_training = FLAGS.mode == 'train'

    def dataset_fn():
      return data.get_dataset(
        config,
        tf_file_reader=tf_file_reader,
        is_training=is_training,
        cache_dataset=FLAGS.cache_dataset)

    if is_training:
      train(
        train_dir,
        config=config,
        dataset_fn=dataset_fn,
        checkpoints_to_keep=FLAGS.checkpoints_to_keep,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        num_steps=FLAGS.num_steps,
        master=FLAGS.master,
        num_sync_workers=FLAGS.num_sync_workers,
        num_ps_tasks=FLAGS.num_ps_tasks,
        task=FLAGS.task)
    else:
      num_batches = FLAGS.eval_num_batches or data.count_examples(
        config.eval_examples_path,
        config.tfds_name,
        config.data_converter,
        file_reader) // config.hparams.batch_size
      eval_dir = os.path.join(run_dir, 'eval' + FLAGS.eval_dir_suffix)
      evaluate(
        train_dir,
        eval_dir,
        config=config,
        dataset_fn=dataset_fn,
        num_batches=num_batches,
        master=FLAGS.master)

  elif FLAGS.mode in ['sample', 'interpolate']:
    if FLAGS.run_dir is None == FLAGS.checkpoint_file is None:
      raise ValueError(
          'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
    if FLAGS.output_dir is None:
      raise ValueError('`--output_dir` is required.')
    tf.gfile.MakeDirs(FLAGS.output_dir)
    generate(config, FLAGS)

  else:
    raise ValueError('Invalid mode: {}'.format(FLAGS.mode))


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)
  run(configs.CONFIG_MAP)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()

import click
import glob
from gpt2_model import *
from data_pipeline import input_fn
import tensorflow as tf
import os
import json

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"
DATASET_PATH = _ROOT + "/data/tf_records/*.tfrecord"


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=32000, show_default=True, help="Vocab size")
@click.option('--optimizer', type=str, default="adam", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=8, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--distributed', type=bool, default=False, show_default=True, help="distributed training")
@click.option('--dataset-path', type=str, default=DATASET_PATH, show_default=True, help="dataset path")
@click.option('--model-dir', type=str, default=MODEL_DIR, show_default=True, help="model directory")
@click.option('--log-dir', type=str, default=LOG_DIR, show_default=True, help="log directory")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          optimizer="adam", batch_size=16, learning_rate=1e-3, distributed=False,
          dataset_path="", model_dir="", log_dir=""):

    par_map = {"num_layers": num_layers, "d_model": embedding_size,
               "num_heads": num_heads, "dff": dff,
               "max_seq_len": max_seq_len, "vocab_size": vocab_size}

    exp_name = "_".join(['{}_{}'.format(k, v) for k, v in par_map.items()])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(model_dir + '/model_par.json', 'w') as f:
        json.dump(par_map, f)

    tf_records = glob.glob((dataset_path))
    if distributed:
        dist_dataset = input_fn(tf_records, batch_size=batch_size)
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dist_dataset)
        with mirrored_strategy.scope():

            model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                         optimizer=optimizer, learning_rate=learning_rate, log_dir=log_dir)
            model.creat_optimizer()
            model.create_checkpoint_manager(model_dir)
            model.create_summary_writer(log_dir)

        model.mirrored_strategy = mirrored_strategy
        model.fit(dist_dataset)
    else:
        dataset = input_fn(tf_records, batch_size=batch_size)
        for d in dataset:
            print(d)
            break
            exit()
        model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                     optimizer=optimizer, learning_rate=learning_rate)
        model.creat_optimizer()
        model.create_checkpoint_manager(model_dir)
        model.create_summary_writer(log_dir)
        model.fit(dataset)
        print("Training Done................")


if __name__ == "__main__":
    train()

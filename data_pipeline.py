import collections
import tensorflow as tf

PAD_ID = 0
UNKNOWN_ID = 1
START_ID = 3
END_ID = 4


def load_vocab(vocab_path):
    vocab = collections.OrderedDict()
    index = 0
    for line in open(vocab_path, 'r').read().splitlines():
        vocab[line.split()[0]] = index
        index += 1
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def parse_example_(serialized_example):
    data_fields = {
        "inputs": tf.io.VarLenFeature(tf.int64),
        "targets": tf.io.VarLenFeature(tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed["inputs"])
    targets = tf.sparse.to_dense(parsed["targets"])

    inputs = tf.cast(inputs, tf.int32)
    targets = tf.cast(targets, tf.int32)

    return inputs, targets


def parse_example(serialized_example):
    data_fields = {
        "input_ids": tf.io.FixedLenFeature([10239 + 1], tf.int64),
        # "input_ids": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    seq_length = 515 + 1
    seq_length = 1024 + 0  # On TPU must be 1024 + 0
    inputs = parsed["input_ids"][0:seq_length]
    targets = parsed["input_ids"][0:seq_length]
    inputs = inputs[:-1]
    targets = targets[1:]
    # inputs = tf.sparse.to_dense(inputs)
    # targets = tf.sparse.to_dense(targets)

    inputs = tf.cast(inputs, tf.int32)
    targets = tf.cast(targets, tf.int32)

    return inputs, targets


def input_fn(tf_records, batch_size=32, padded_shapes=([-1], [-1]), epoch=10, buffer_size=100):
    input_files = tf.data.Dataset.list_files(tf_records)
    dataset = input_files.shuffle(buffer_size=512)
    dataset = tf.data.TFRecordDataset(dataset, buffer_size=100)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(parse_example, num_parallel_calls=AUTOTUNE)
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def input_fn_(tf_records, batch_size=32, padded_shapes=([-1], [-1]), epoch=10, buffer_size=10000):
    input_files = tf.data.Dataset.list_files(dataset_path)
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(lambda record: parse_example(record))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat()

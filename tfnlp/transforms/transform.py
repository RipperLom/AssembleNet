import tensorflow as tf

from tfnlp.utils.common import get_all_files


def load_batch_ops(example, batch_size, shuffle):
    """
    load batch ops
    """
    if not shuffle:
        return tf.train.batch([example], 
                              batch_size = batch_size,
                              num_threads = 1,
                              capacity = 10000 + 2 * batch_size)
    else:
        return tf.train.shuffle_batch([example],
                                      batch_size = batch_size,
                                      num_threads = 1,
                                      capacity = 10000 + 2 * batch_size,
                                      min_after_dequeue = 10000)


class TFPairwisePaddingData(object):
    """
    for pairwise padding data
    """
    def __init__(self, config):
        self.filelist = get_all_files(config["train_file"])
        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["num_epochs"])
        if int(config["shuffle"]) == 0:
            shuffle = False
        else:
            shuffle = True
        self.shuffle = shuffle
        self.reader = None
        self.file_queue = None
        self.left_slots = dict(config["left_slots"])
        self.right_slots = dict(config["right_slots"])
        
    def ops(self):
        """
        produce data
        """
        self.file_queue = tf.train.string_input_producer(self.filelist,
                                                         num_epochs=self.epochs)
        self.reader = tf.TFRecordReader()
        _, example = self.reader.read(self.file_queue)
        batch_examples = load_batch_ops(example, self.batch_size, self.shuffle)
        features_types = {}
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)}) 
                            for (u, v) in self.left_slots.iteritems()]
        [features_types.update({"pos_" + u: tf.FixedLenFeature([v], tf.int64)}) 
                            for (u, v) in self.right_slots.iteritems()]
        [features_types.update({"neg_" + u: tf.FixedLenFeature([v], tf.int64)}) 
                            for (u, v) in self.right_slots.iteritems()]
        features = tf.parse_example(batch_examples, features = features_types)
        return dict([(k, features[k]) for k in self.left_slots.keys()]),\
                dict([(k, features["pos_" + k]) for k in self.right_slots.keys()]),\
                    dict([(k, features["neg_" + k]) for k in self.right_slots.keys()])


class TFPointwisePaddingData(object):
    """
    for pointwise padding data
    """
    def __init__(self, config):
        self.filelist = get_all_files(config["train_file"])
        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["num_epochs"])
        if int(config["shuffle"]) == 0:
            shuffle = False
        else:
            shuffle = True
        self.shuffle = shuffle
        self.reader = None
        self.file_queue = None
        self.left_slots = dict(config["left_slots"])
        self.right_slots = dict(config["right_slots"])
    
    def ops(self):
        """
        gen data
        """
        self.file_queue = tf.train.string_input_producer(self.filelist, 
                                                         num_epochs=self.epochs)
        self.reader = tf.TFRecordReader()
        _, example = self.reader.read(self.file_queue)
        batch_examples = load_batch_ops(example, self.batch_size, self.shuffle)
        features_types = {"label": tf.FixedLenFeature([2], tf.int64)}
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)}) 
                            for (u, v) in self.left_slots.items()]
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)}) 
                            for (u, v) in self.right_slots.items()]
        features = tf.parse_example(batch_examples, features = features_types)
        return dict([(k, features[k]) for k in self.left_slots.keys()]),\
                dict([(k, features[k]) for k in self.right_slots.keys()]),\
                    features["label"]


class TFListwisePaddingData(object):
    """
    for listwise padding data
    """
    def __init__(self, config):
        self.filelist = get_all_files(config["train_file"])
        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["num_epochs"])
        if int(config["shuffle"]) == 0:
            shuffle = False
        else:
            shuffle = True
        self.shuffle = shuffle
        self.reader = None
        self.file_queue = None
        self.left_slots = dict(config["left_slots"])
        self.n_class = int(config['n_class'])

    def ops(self):
        """
        gen data
        """
        self.file_queue = tf.train.string_input_producer(self.filelist,
                                                         num_epochs=self.epochs)
        self.reader = tf.TFRecordReader()
        _, example = self.reader.read(self.file_queue)
        batch_examples = load_batch_ops(example, self.batch_size, self.shuffle)
        features_types = {"label": tf.FixedLenFeature([self.n_class], tf.int64)}
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)})
                            for (u, v) in self.left_slots.items()]
        features = tf.parse_example(batch_examples, features = features_types)
        return dict([(k, features[k]) for k in self.left_slots.keys()]),\
                    features["label"]


class Elmo4PairData(object):
    """
    for listwise padding data
    """
    def __init__(self, config):
        self.filelist = get_all_files(config["train_file"])
        self.batch_size = int(config["batch_size"])
        self.epochs = int(config["num_epochs"])
        if int(config["shuffle"]) == 0:
            shuffle = False
        else:
            shuffle = True
        self.shuffle = shuffle
        self.reader = None
        self.file_queue = None
        self.token_ids = dict(config["token_ids"])
        self.token_ids_reverse = dict(config["token_ids_reverse"])
        self.next_token_ids = dict(config["next_token_ids"])
        self.next_token_ids_reverse = dict(config["next_token_ids_reverse"])

    def ops(self):
        """
        gen data
        """
        self.file_queue = tf.train.string_input_producer(self.filelist,
                                                         num_epochs=self.epochs)
        self.reader = tf.TFRecordReader()
        _, example = self.reader.read(self.file_queue)
        batch_examples = load_batch_ops(example, self.batch_size, self.shuffle)
        features_types = {}
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)})
                            for (u, v) in self.token_ids.items()]
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)})
                            for (u, v) in self.token_ids_reverse.items()]
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)})
                            for (u, v) in self.next_token_ids.items()]
        [features_types.update({u: tf.FixedLenFeature([v], tf.int64)})
                            for (u, v) in self.next_token_ids_reverse.items()]
        features = tf.parse_example(batch_examples, features = features_types)
        return dict([(k, features[k]) for k in self.token_ids.keys()]), \
               dict([(k, features[k]) for k in self.token_ids_reverse.keys()]), \
               dict([(k, features[k]) for k in self.next_token_ids.keys()]), \
               dict([(k, features[k]) for k in self.next_token_ids_reverse.keys()])


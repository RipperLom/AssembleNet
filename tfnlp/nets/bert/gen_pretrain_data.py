# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import collections
import tensorflow as tf

from tfnlp.utils import tokenization
from tfnlp.utils import text_norm
from tfnlp.utils import common
from tfnlp.utils import lang_tool


# 全局分词对象
g_seg_obj = None
FLAGS = None


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


class TrainingInstance(object):
    """
    A single training instance (sentence pair).
        tokens:              sent_a + sent_b
        segment_ids:         sent_a 和 sent_b 中间的切分mask
        masked_lm_positions: 哪些位置被遮挡了
        masked_lm_labels:    遮挡的label是什么
        is_random_next:      sent_b是否随机得来
    """
    def __init__(self, tokens, segment_ids, masked_lm_positions, 
                masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.is_random_next = is_random_next

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([text_norm.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([text_norm.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s
    
    def __repr__(self):
        return self.__str__()



def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files=[]):
    """Create TF example files from TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.tokens2ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        # pad 0
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        
        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.tokens2ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1
        if instance.is_random_next:
            next_sentence_label = 0

        # 特征构建
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

        # just for debug
        if inst_index < 10:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % \
                " ".join([text_norm.printable_text(x) for x in instance.tokens]))
            
            for feat_name in features.keys():
                feature = features[feat_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feat_name, " ".join([str(x) for x in values])))
    
    # close file
    for writer in writers:
        writer.close()
    tf.logging.info("Wrote %d total instances", total_written)
    return True



# 从文本文件构造 masked LM/next sentence 
# 一个句子一行
# doc直接用空号分割，next sentence prediction 不跨doc 
def create_training_instances(input_files, tokenizer, max_seq_length,
                            dupe_factor, short_seq_prob, masked_lm_prob,
                            max_predictions_per_seq, rng):
    all_documents = [[]]
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = text_norm.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()
                # 空行作为doc 分隔符
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                # print("句子：", tokens)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents and shuffle
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []

    ## 数据重复次数
    for _ in range(dupe_factor):
        for doc_idx in range(len(all_documents)):
            instances.extend(
                    create_instances_from_document(
                            all_documents, doc_idx, max_seq_length, short_seq_prob,
                            masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents, document_index, \
        max_seq_length, short_seq_prob, \
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)

        # 多个句子凑够一个chunk
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                
                #截断
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                # sent_a + sent_b: 句子 + 分割标识
                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                # 生成 mask lm position的位置
                (tokens, masked_lm_positions, masked_lm_labels) \
                        = create_masked_lm_predictions(\
                            tokens, masked_lm_prob, \
                            max_predictions_per_seq, vocab_words, rng)
                
                instance = TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        is_random_next=is_random_next,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances



# 命名元组
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                max_predictions_per_seq, vocab_words, rng):
    """
        Creates the predictions for the masked LM objective.
        Whole Word Masking: 根据词查分的子串都mask

        1、 预测词个数：  min(最大预测词， len(tokens) * 0.15)
        2、 生成的mask中，80% mask  10% 原词 10% 随机替换
        3、 记录mask的位置 和 词
    """
    offset_dic = {}
    if FLAGS.seg_path:
        global g_seg_obj
        txt = "".join(tokens)
        txt_seg = g_seg_obj.seg(txt)
        # print("segword->", txt_seg)
        # offset -> token_id
        offset = 0
        i = 0
        for (i, word) in enumerate(txt_seg):
            for j in range(len(word)):
                offset_dic[offset + j] = i
            offset += len(word)

    offset = 0
    cand_indexes = []
    pre_token = ""
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            offset += len(token)
            continue
        # merge 中文分词信息
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1):
            if offset_dic.get(offset) == offset_dic.get(offset - len(pre_token)):
                cand_indexes[-1].append(i)
                offset += len(token)
                pre_token = token
                continue
        
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1) and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
        offset += len(token)
        pre_token = token


    # print(tokens)
    # print(cand_indexes)
    # tmp = []
    # for toks in cand_indexes:
    #     for i in toks:
    #         tmp.append(tokens[i])
    #     tmp.append(" ")
    # print("".join(tmp))

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)

    # mask词的个数
    num_to_predict = min(max_predictions_per_seq,  max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    #mask-LM： 位置 + label
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """
        Truncates a pair of sequences to a maximum sequence length.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        # 为了增加随机性：选择长句子 随机 删头 或者 删尾
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
    return True


def g_define_flag():
    flags = tf.flags
    flags.DEFINE_string( "input_file", None, "Input raw text file")
    flags.DEFINE_string("output_prefix", None, "Output TF example file prefix")
    flags.DEFINE_string("seg_path", None, "lang seg dict path")
    flags.DEFINE_integer("output_num", 1, "Output TF example file number.")
    flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
    flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")

    flags.DEFINE_bool("do_whole_word_mask", False, "Whether to use whole word masking rather than per-WordPiece masking.")
    flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
    flags.DEFINE_integer("max_predictions_per_seq", 20, "Maximum number of masked LM predictions per sequence.")
    flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
    flags.DEFINE_integer("dupe_factor", 10,  "Number of times to duplicate the input data (with different masks).")
    flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
    flags.DEFINE_float( "short_seq_prob", 0.1, "Probability of creating sequences which are shorter than the maximum length.")
    
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_prefix")
    return FLAGS


def main(_):
    global FLAGS
    tf.logging.set_verbosity(tf.logging.INFO)

    # 加载分词 
    if FLAGS.seg_path:
        tf.logging.info("*** load seg dict ***")
        seg = lang_tool.LangTool()
        if seg.load(FLAGS.seg_path):
            global g_seg_obj
            g_seg_obj = seg

    # token 
    tokenizer = tokenization.FullTokenizer( vocab_file = FLAGS.vocab_file, 
                            do_lower_case = FLAGS.do_lower_case)

    # 所有输入文件
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))
    
    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("    %s", input_file)

    # gen instances
    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
            input_files, tokenizer, FLAGS.max_seq_length, 
            FLAGS.dupe_factor, FLAGS.short_seq_prob, FLAGS.masked_lm_prob, 
            FLAGS.max_predictions_per_seq,  rng)
    
    # dump to tf_example
    output_prefix = FLAGS.output_prefix
    output_files = common.file_list(output_prefix, FLAGS.output_num, ".tfrecord")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("    %s", output_file)
    
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                FLAGS.max_predictions_per_seq, output_files)
    return True


if __name__ == "__main__":
    FLAGS = g_define_flag()
    tf.app.run()



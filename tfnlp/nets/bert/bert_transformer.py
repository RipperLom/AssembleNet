#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################

"""
File: bert_transformer.py
Author: wangyan 
Date: 2019/09/10 11:53:24
Brief: bert config
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re
import six
import json
import copy
import math

import numpy as np
import tensorflow as tf
from tfnlp.nets.bert import bert_util 


class Attention(object):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and 
    `to_tensor` into "key" and "value" tensors. 
    These are (effectively) a list of tensors of length `num_attention_heads`, 
    where each tensor is of shape [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length, from_width]
        
        to_tensor:   float Tensor of shape [batch_size, to_seq_length, to_width]

        attention_mask: (optional) int32 Tensor of shape [batch_size, from_seq_length, to_seq_length]. 
                    The values should be 1 or 0. 
                    The attention scores will effectively be set to 
                        infinity for any positions in the mask that are 0
                        and will be unchanged for positions that are 1
        
        num_attention_heads: int. Number of attention heads.
        
        size_per_head:       int. Size of each attention head.

        query_act: (optional) Activation function for the query transform.
        key_act:   (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.

        attention_probs_dropout_prob: (optional) float. Dropout probability of the attention probabilities.
        initializer_range:            float. Range of the weight initializer.
        
        do_return_2d_tensor: bool. 
                    If True, the output will be of shape [batch_size * from_seq_length, num_attention_heads * size_per_head]
                    If False, the output will be of shape [batch_size, from_seq_length, num_attention_heads * size_per_head].
        
        batch_size: (Optional) int. If the input is 2D, this might be the batch size of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length of the 3D version of the `from_tensor`.
        to_seq_length:   (Optional) If the input is 2D, this might be the seq length of the 3D version of the `to_tensor`.

    Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads * size_per_head]. 
                    (If `do_return_2d_tensor` is true, this will be of shape
                    [batch_size * from_seq_length, num_attention_heads * size_per_head]).

    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def __init__(self, name = "multi_headed_attention"):
        self.name = name

    @staticmethod
    def build(from_tensor,
            to_tensor,
            attention_mask = None,
            num_attention_heads = 1,
            size_per_head = 512,
            query_act = None,
            key_act = None,
            value_act = None,
            attention_probs_dropout_prob = 0.0,
            initializer_range = 0.02,
            do_return_2d_tensor = False,
            batch_size = None,
            from_seq_length = None,
            to_seq_length = None):

        # fmt shape
        from_shape = bert_util.get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = bert_util.get_shape_list(to_tensor, expected_rank=[2, 3])
        if len(from_shape) != len(to_shape):
            raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")
        
        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (batch_size is None or from_seq_length is None or to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values for `batch_size`, `from_seq_length`, and `to_seq_length` " + 
                    "must all be specified.")

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        from_tensor_2d = bert_util.reshape_to_matrix(from_tensor)
        to_tensor_2d = bert_util.reshape_to_matrix(to_tensor)

        # `query_layer` = [B*F, N*H]
        query_layer = tf.layers.dense(
            from_tensor_2d,
            num_attention_heads * size_per_head,
            activation = query_act,
            name = "query",
            kernel_initializer = tf.truncated_normal_initializer(initializer_range))

        # `key_layer` = [B*T, N*H]
        key_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation = key_act,
            name = "key",
            kernel_initializer = tf.truncated_normal_initializer(initializer_range))

        # `value_layer` = [B*T, N*H]
        value_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation = value_act,
            name = "value",
            kernel_initializer = tf.truncated_normal_initializer(initializer_range))

        # `query_layer` = [B, N, F, H]
        query_layer = Attention.transpose_for_scores(query_layer, batch_size,
                                            num_attention_heads, from_seq_length,
                                            size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = Attention.transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                            to_seq_length, size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = bert_util.dropout(attention_probs, attention_probs_dropout_prob)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape( value_layer, \
                [batch_size, to_seq_length, num_attention_heads, size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return context_layer


    @staticmethod
    def transpose_for_scores(input_tensor, batch_size, \
                            num_attention_heads, seq_length, width):
        output_tensor = tf.reshape( input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor


class Transformer(object):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.

    See the original paper: https://arxiv.org/abs/1706.03762
    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
        
        hidden_size: int. Hidden size of the Transformer.
        
        num_hidden_layers: int. Number of layers (blocks) in the Transformer.
        
        num_attention_heads: int. Number of attention heads in the Transformer.
        
        intermediate_size: int. The size of the "intermediate" (a.k.a., feed forward) layer.
        
        intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
        
        hidden_dropout_prob: float. Dropout probability for the hidden layers.
        attention_probs_dropout_prob: float. Dropout probability of the attention probabilities.
        
        initializer_range: float. Range of the initializer (stddev of truncated normal).
        
        do_return_all_layers: Whether to also return all layers or just the final layer.
    Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size], the final hidden layer of the Transformer.

    Raises:
        ValueError: A Tensor shape or parameter is invalid.
    """

    def __init__(self):
        pass
    
    def load(self):
        pass


    @staticmethod
    def build(input_tensor,
                attention_mask = None,
                hidden_size = 768,
                num_hidden_layers = 12,
                num_attention_heads = 12,
                intermediate_size = 3072,
                intermediate_act_fn = bert_util.gelu,
                hidden_dropout_prob = 0.1,
                attention_probs_dropout_prob = 0.1,
                initializer_range = 0.02,
                do_return_all_layers = False):
        # check param
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" \
                            % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
        input_shape = bert_util.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]
        
        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %  (input_width, hidden_size))

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = bert_util.reshape_to_matrix(input_tensor)

        all_layer_outputs = []
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output

                # attention
                with tf.variable_scope("attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = Attention.build(from_tensor = layer_input,
                                                to_tensor = layer_input,
                                                attention_mask = attention_mask,
                                                num_attention_heads = num_attention_heads,
                                                size_per_head = attention_head_size,
                                                attention_probs_dropout_prob = attention_probs_dropout_prob,
                                                initializer_range = initializer_range,
                                                do_return_2d_tensor = True,
                                                batch_size = batch_size,
                                                from_seq_length = seq_length,
                                                to_seq_length = seq_length)
                        attention_heads.append(attention_head)
                    
                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        # In the case where we have other sequences, we just concatenate
                        # them to the self-attention head before the projection.
                        attention_output = tf.concat(attention_heads, axis = -1)
                    
                    # Run a linear projection of `hidden_size` then add a residual
                    # with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(attention_output, hidden_size,
                                            kernel_initializer = tf.truncated_normal_initializer(initializer_range))
                        attention_output = bert_util.dropout(attention_output, hidden_dropout_prob)
                        attention_output = bert_util.layer_norm(attention_output + layer_input)
                
                # intermediate
                with tf.variable_scope("intermediate"):
                    # The activation is only applied to the "intermediate" hidden layer
                    intermediate_output = tf.layers.dense( attention_output, intermediate_size,
                                    activation=intermediate_act_fn,
                                    kernel_initializer = tf.truncated_normal_initializer(initializer_range))

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    layer_output = tf.layers.dense( intermediate_output, hidden_size,
                                    kernel_initializer = tf.truncated_normal_initializer(initializer_range))
                    layer_output = bert_util.dropout(layer_output, hidden_dropout_prob)
                    layer_output = bert_util.layer_norm(layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)
        
        # 是否返回所有层
        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = bert_util.reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = bert_util.reshape_from_matrix(prev_output, input_shape)
            return final_output



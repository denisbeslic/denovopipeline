# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from ..utils import misc_utils as utils


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size


def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  if tf.gfile.Exists(vocab_file):
    utils.print_out("# Vocab file %s exists" % vocab_file)
    vocab, vocab_size = load_vocab(vocab_file)
    if check_special_token:
      # Verify if the vocab starts with unk, sos, eos
      # If not, prepend those tokens & generate a new vocab file
      if not unk: unk = UNK
      if not sos: sos = SOS
      if not eos: eos = EOS
      assert len(vocab) >= 3
      if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
        utils.print_out("The first 3 vocab words [%s, %s, %s]"
                        " are not [%s, %s, %s]" %
                        (vocab[0], vocab[1], vocab[2], unk, sos, eos))
        vocab = [unk, sos, eos] + vocab
        vocab_size += 3
        new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
        vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  vocab_size = len(vocab)
  return vocab_size, vocab_file


def create_vocab_tables(tgt_vocab_file):
  """Creates vocab tables for tgt_vocab_file."""
  tgt_vocab_table = lookup_ops.index_table_from_file(
      tgt_vocab_file, default_value=UNK_ID)
  
  return tgt_vocab_table


def load_embed_txt(embed_file):
  """Load embed_file into a python dictionary.

  Note: the embed_file should be a Glove formated txt file. Assuming
  embed_size=5, for example:

  the -0.071549 0.093459 0.023738 -0.090339 0.056123
  to 0.57346 0.5417 -0.23477 -0.3624 0.4037
  and 0.20327 0.47348 0.050877 0.002103 0.060547

  Args:
    embed_file: file path to the embedding file.
  Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
  """
  emb_dict = dict()
  emb_size = None
  with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
    for line in f:
      tokens = line.strip().split(" ")
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        assert emb_size == len(vec), "All embedding size should be same."
      else:
        emb_size = len(vec)
  return emb_dict, emb_size

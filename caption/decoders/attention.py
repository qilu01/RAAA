import torch

import framework.configbase
import framework.ops


class DecoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.rnn_type = 'lstm'
    self.num_words = 0
    self.dim_word = 512
    self.hidden_size = 512
    self.num_layers = 1
    self.hidden2word = False
    self.tie_embed = False
    self.fix_word_embed = False
    self.max_words_in_sent = 20
    self.dropout = 0.5
    self.schedule_sampling = False
    self.ss_rate = 0.05
    self.ss_max_rate = 0.25
    self.ss_increase_rate = 0.05
    self.ss_increase_epoch = 5

    self.greedy_or_beam = False
    self.beam_width = 1
    self.sent_pool_size = 1

  def _assert(self):
    if self.tie_embed and not self.hidden2word:
      assert self.dim_word == self.hidden_size


class AttnDecoderConfig(DecoderConfig):

  def __init__(self):
    super().__init__()
    self.memory_same_key_value = True
    self.attn_input_size = 512
    self.attn_size = 512
    self.attn_type = 'mlp'

  def _assert(self):
    assert self.attn_type in ['dot', 'general', 'mlp'], ('Please select a valid attention type.')



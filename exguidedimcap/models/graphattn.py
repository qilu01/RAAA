import framework.configbase
import caption.encoders.vanilla
import caption.decoders.attention

import exguidedimcap.encoders.gcn

MPENCODER = 'mp_encoder'
ATTNENCODER = 'attn_encoder'
DECODER = 'decoder'


class GraphModelConfig(framework.configbase.ModelConfig):
  def __init__(self):
    super().__init__()
    self.subcfgs[MPENCODER] = caption.encoders.vanilla.EncoderConfig()
    self.subcfgs[ATTNENCODER] = exguidedimcap.encoders.gcn.RGCNEncoderConfig()
    self.subcfgs[DECODER] = caption.decoders.attention.AttnDecoderConfig()

  def _assert(self):
    assert self.subcfgs[MPENCODER].dim_embed == self.subcfgs[DECODER].hidden_size
    assert self.subcfgs[ATTNENCODER].dim_hidden == self.subcfgs[DECODER].attn_input_size





import torch
import numpy as np
import caption.encoders.vanilla
import caption.models.captionbase

import exguidedimcap.encoders.gcn
import exguidedimcap.decoders.memory
import exguidedimcap.models.graphattn

MPENCODER = 'mp_encoder'
ATTNENCODER = 'attn_encoder'
DECODER = 'decoder'


class RelationAwareGraphCovAndLangAttnModel(caption.models.captionbase.CaptionModelBase):
    def build_submods(self):
        submods = {}
        submods[MPENCODER] = caption.encoders.vanilla.Encoder(self.config.subcfgs[MPENCODER])
        submods[ATTNENCODER] = exguidedimcap.encoders.gcn.RoleRGCNEncoder(self.config.subcfgs[ATTNENCODER])
        submods[DECODER] = exguidedimcap.decoders.memory.LangAttnDecoder(self.config.subcfgs[DECODER])
        return submods

    def forward_loss(self, batch_data, step=None):
        input_batch = self.prepare_input_batch(batch_data, is_train=True)

        enc_outs = self.forward_encoder(input_batch)

        logits = self.submods[DECODER](input_batch['caption_ids'][:, :-1],
                                       enc_outs['init_states'], enc_outs['attn_fts'], input_batch['attn_masks'],
                                       input_batch['flow_edges'])
        cap_loss = self.criterion(logits, input_batch['caption_ids'],
                                  input_batch['caption_masks'])

        return cap_loss

    def validate_batch(self, batch_data, addition_outs=None):
        input_batch = self.prepare_input_batch(batch_data, is_train=False)
        enc_outs = self.forward_encoder(input_batch)

        batch_size = input_batch['attn_masks'].size(0)
        init_words = torch.zeros(batch_size, dtype=torch.int64).to(self.device)

        pred_sent, _ = self.submods[DECODER].sample_decode(init_words,
                                                           enc_outs['init_states'], enc_outs['attn_fts'],
                                                           input_batch['attn_masks'],
                                                           input_batch['flow_edges'], greedy=True)

        return pred_sent

    def test_batch(self, batch_data, greedy_or_beam):
        input_batch = self.prepare_input_batch(batch_data, is_train=False)
        enc_outs = self.forward_encoder(input_batch)

        batch_size = input_batch['attn_masks'].size(0)
        init_words = torch.zeros(batch_size, dtype=torch.int64).to(self.device)

        if greedy_or_beam:
            sent_pool = self.submods[DECODER].beam_search_decode(
                init_words, enc_outs['init_states'], enc_outs['attn_fts'],
                input_batch['attn_masks'], input_batch['flow_edges'])
            pred_sent = [pool[0][1] for pool in sent_pool]
        else:
            pred_sent, word_logprobs = self.submods[DECODER].sample_decode(
                init_words, enc_outs['init_states'], enc_outs['attn_fts'],
                input_batch['attn_masks'], input_batch['flow_edges'], greedy=True)
            sent_pool = []
            for sent, word_logprob in zip(pred_sent, word_logprobs):
                sent_pool.append([(word_logprob.sum().item(), sent, word_logprob)])

        return pred_sent, sent_pool

    def prepare_input_batch(self, batch_data, is_train=False):
        outs = {}
        outs['mp_fts'] = torch.FloatTensor(batch_data['mp_fts']).to(self.device)
        outs['attn_fts'] = torch.FloatTensor(batch_data['attn_fts']).to(self.device)
        outs['attn_masks'] = torch.FloatTensor(batch_data['attn_masks'].astype(np.float32)).to(self.device)
        # build rel_edges tensor
        batch_size, max_nodes, _ = outs['attn_fts'].size()
        num_rels = len(batch_data['edge_sparse_matrices'][0])
        rel_edges = np.zeros((batch_size, num_rels, max_nodes, max_nodes), dtype=np.float32)
        for i, edge_sparse_matrices in enumerate(
                batch_data['edge_sparse_matrices']):
            for j, edge_sparse_matrix in enumerate(
                    edge_sparse_matrices):
                rel_edges[i, j] = edge_sparse_matrix.todense()
        outs['rel_edges'] = torch.FloatTensor(rel_edges).to(self.device)
        if is_train:
            outs['caption_ids'] = torch.LongTensor(batch_data['caption_ids']).to(self.device)
            outs['caption_masks'] = torch.FloatTensor(batch_data['caption_masks'].astype(np.float32)).to(self.device)
            if 'gt_attns' in batch_data:
                outs['gt_attns'] = torch.FloatTensor(batch_data['gt_attns'].astype(np.float32)).to(self.device)

        flow_edges = [x.toarray() for x in batch_data[
            'flow_sparse_matrix']]
        flow_edges = np.stack(flow_edges, 0)
        outs['flow_edges'] = torch.FloatTensor(flow_edges).to(self.device)

        outs['node_types'] = torch.LongTensor(batch_data['node_types']).to(self.device)
        outs['attr_order_idxs'] = torch.LongTensor(batch_data['attr_order_idxs']).to(self.device)
        return outs

    def forward_encoder(self, input_batch):
        attn_embeds = self.submods[ATTNENCODER](input_batch['attn_fts'],
                                                input_batch['node_types'], input_batch['attr_order_idxs'],
                                                input_batch['rel_edges'])
        graph_embeds = torch.sum(attn_embeds * input_batch['attn_masks'].unsqueeze(2), 1)
        graph_embeds = graph_embeds / torch.sum(input_batch['attn_masks'], 1, keepdim=True)
        enc_states = self.submods[MPENCODER](
            torch.cat([input_batch['mp_fts'], graph_embeds], 1))
        return {'init_states': enc_states, 'attn_fts': attn_embeds}

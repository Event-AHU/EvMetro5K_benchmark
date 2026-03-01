# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from timm.models.helpers import named_apply

from strhub.data.utils import Tokenizer
from strhub.models.utils import init_weights

from .modules import Decoder, DecoderLayer, Encoder, TokenEmbedding

# from .hypergraph import feature_concat
from .hypergraph import build_H_and_G_from_tokens
# from .hypergraph import HypergraphAttentionEncoder
from .hypergraph import HGNN


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=(32, 128), patch_size=(2, 16), in_chans=3, embed_dim=384, norm_layer=False, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class PARSeq(nn.Module):

    def __init__(
        self,
        num_tokens: int,
        max_label_length: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        decode_ar: bool,
        refine_iters: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters
        
        # self.patch_embedding = PatchEmbed(img_size=(32, 128), patch_size=(2, 16), in_chans=3, embed_dim=384)
        
        self.encoder = Encoder(
            img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads, mlp_ratio=enc_mlp_ratio
        )
        
        self.patch_embedding = self.encoder.patch_embed
        
        # for name, param in self.encoder.named_parameters():                
        #     param.requires_grad = False
                
        self.build_G = build_H_and_G_from_tokens
        self.hypergraph_encoder = HGNN(in_ch=embed_dim*2, n_class=embed_dim, n_hid=embed_dim)
        
        
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, num_tokens - 2)
        self.text_embed = TokenEmbedding(num_tokens, embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    @property
    def _device(self) -> torch.device:
        return next(self.head.parameters(recurse=False)).device

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor, hg=None):
        return self.encoder(img, hg)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, tokenizer: Tokenizer, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        # breakpoint()
        event_image, rgb_image = images[0], images[1]
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = event_image.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        
        event_emb = self.patch_embedding(event_image)
        rgb_emb = self.patch_embedding(rgb_image)
        
        event_memory = self.encode(event_emb) # torch.Size([64, 3, 32, 128]) → torch.Size([64, 128, 384])
        # rgb_memory = self.encode(rgb_emb) # torch.Size([64, 3, 32, 128]) → torch.Size([64, 128, 384])
        
        rgbe_memory = torch.cat((event_memory, rgb_emb), dim=-1)
        _, G = self.build_G(rgbe_memory)
        hg_memory = self.hypergraph_encoder(rgbe_memory, G)
        
        memory = self.encode(rgb_emb, hg_memory)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = tokenizer.bos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                # Mask tokens beyond the first EOS token.
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]]
                )
                logits = self.head(tgt_out)

        return logits

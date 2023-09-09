# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,

    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.iou_head_depth  = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        self.cls_token = False

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, transformer_dim)

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def add_cls_token(self, num_classes: int, custom_hypers: bool = True, cls_token_only: bool = True):
        self.cls_token = True
        self.cls_token_only = cls_token_only
        self.num_classes = num_classes
        self.cls_iou_token = nn.Embedding(1, self.transformer_dim)
        self.cls_mask_tokens = nn.Embedding(self.num_classes, self.transformer_dim)
        self.custom_hypers = custom_hypers
        self.cls_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3)
                for i in range(self.num_classes)
            ]
        ) if custom_hypers else None
        self.cls_iou_prediction_head = MLP(
            self.transformer_dim, self.iou_head_hidden_dim, self.num_classes, self.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        context_embeddings: torch.Tensor=None,
        attn_sim=None,
        target_embedding=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched predicted masks for each class
          torch.Tensor: batched predictions of mask quality for each class
        """
        masks, iou_pred, cls_masks,cls_iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            context_embeddings=context_embeddings,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        if masks is not None:
            masks = masks[:, mask_slice, :, :]
            iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred, cls_masks, cls_iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor = None,
        attn_sim=None,
        target_embedding=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        b,*_ = sparse_prompt_embeddings.shape

        # Expand per-image data in batch direction to be per-mask
        # assert len(image_embeddings.shape) == 3,f"image_embeddings.shape: {image_embeddings.shape}, len(image_embeddings.shape): {len(image_embeddings.shape)}"
        src = torch.repeat_interleave(image_embeddings, b, dim=0)
        src = src + dense_prompt_embeddings # shape (b, c, h, w)
        b,c,h,w = src.shape

        pos_src = torch.repeat_interleave(image_pe, b, dim=0) # shape (b, c, h, w)
        b, c, h, w = src.shape


        if context_embeddings is None:
            ctx_src = torch.zeros((b,0,c,h,w)).to(src.device)
        else:
            assert len(context_embeddings.shape) == 3,f"context_embeddings.shape: {context_embeddings.shape}"
            n = context_embeddings.shape[0]
            # repeat context_embeddings to have shape (b, n, c, h, w)
            ctx_src = torch.repeat_interleave(context_embeddings, b, dim=0)

            # reshape pos_src to (b,c,(n+1)h,w)
            pos_src = pos_src.view(b, c, h, w).repeat(1, 1, n+1, 1).view(b, c, (n+1)*h, w)

        # Concatenate output tokens
        all_tokens = { }
        all_hyper_nets = { }
        all_iou_head = { }

        use_normal_token = not (self.cls_token and self.cls_token_only and self.training)

        if use_normal_token:
            all_tokens["main"] = [self.iou_token.weight, self.mask_tokens.weight]
            all_hyper_nets["main"] = self.output_hypernetworks_mlps
            all_iou_head["main"] = self.iou_prediction_head

        if self.cls_token:
            all_tokens["cls"] = [self.cls_iou_token.weight, self.cls_mask_tokens.weight,self.iou_token.weight, self.mask_tokens.weight]
            all_hyper_nets["cls"] = self.cls_hypernetworks_mlps
            all_iou_head["cls"] = self.cls_iou_prediction_head

        model_outputs = {}
        for k,tokens in all_tokens.items():

            num_mask_tokens = len(tokens[1])
            hyper_nets = all_hyper_nets[k]
            iou_head = all_iou_head[k]

            output_tokens = torch.cat(tokens, dim=0)
            output_tokens = output_tokens.unsqueeze(0).expand(b, -1, -1)
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

            # Run the transformer
            hs, low_res_embedding = self.transformer(src, pos_src, tokens, ctx_src, attn_sim, target_embedding)
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1 : (1 + num_mask_tokens), :]

            # Upscale mask embeddings and predict masks using the mask tokens
            low_res_embedding = low_res_embedding.transpose(1, 2).view(b, c, h, w)
            upscaled_embedding = self.output_upscaling(low_res_embedding)

            hyper_in_list: List[torch.Tensor] = []
            for i in range(num_mask_tokens):
                if hyper_nets is not None:
                    hyper_in_list.append(hyper_nets[i](mask_tokens_out[:, i, :]))
                else:
                    # this means we're making masks for a cls token, but custom_hypers=False
                    hyper_in_list.append(self.output_hypernetworks_mlps[0](mask_tokens_out[:, i, :]))
            hyper_in = torch.stack(hyper_in_list, dim=1)

            b, _c, _h, _w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, _c, _h * _w)).view(b, -1, _h, _w)


            # Generate mask quality predictions
            iou_pred = iou_head(iou_token_out)

            model_outputs[k] = (masks, iou_pred)

        masks,iou_pred = model_outputs["main"] if use_normal_token else (None,None)
        cls_masks,cls_iou_pred = model_outputs["cls"] if self.cls_token else (None,None)

        return masks,iou_pred, cls_masks,cls_iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
import diffusers
from typing import Union, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Timesteps, TimestepEmbedding, get_down_block, UNetMidBlock2DCrossAttn, linear_to_conv2d_map

class ControlNetConditioningEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels,
        conditioning_channels=3,
        block_out_channels=(16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetModel(diffusers.ControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_num_residuals(self):
        num_res = 2  # initial sample + mid block
        for down_block in self.down_blocks:
            num_res += len(down_block.resnets)
            if hasattr(down_block, "downsamplers") and down_block.downsamplers is not None:
                num_res += len(down_block.downsamplers)
        return num_res

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ):
        """
        The [`ControlNetModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if text_embeds is None:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )

                if time_ids is None:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )

                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        return down_block_res_samples, mid_block_res_sample


'''
class ControlNetModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels=4,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        only_cross_attention=False,
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=1280,
        transformer_layers_per_block=1,
        attention_head_dim=8,
        use_linear_projection=False,
        upcast_attention=False,
        resnet_time_scale_shift="default",
        conditioning_embedding_out_channels=(16, 32, 96, 256),
        **kwargs,
    ):
        super().__init__()

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
        )

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                transformer_layers_per_block=transformer_layers_per_block[i],
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

    def get_num_residuals(self):
        num_res = 2 # initial sample + mid block
        for down_block in self.down_blocks:
            num_res += len(down_block.resnets)
            if hasattr(down_block, "downsamplers") and down_block.downsamplers is not None:
                num_res += len(down_block.downsamplers)
        return num_res

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
    ):
        # 1. time
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        sample += controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        return down_block_res_samples, mid_block_res_sample
'''

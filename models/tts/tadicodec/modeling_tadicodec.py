import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from typing import Optional, Dict, Any

from models.tts.tadicodec.llama_nar_prefix import DiffLlamaPrefix
from models.codec.amphion_codec.quantize.bsq import (
    BinarySphericalQuantizer,
    SimpleQuantizer,
)
import torch.nn.functional as F


class TaDiCodec(nn.Module):
    """
    TaDiCodec: A diffusion-based codec model for Text-to-Speech (TTS)
    that uses a non-autoregressive Llama-style transformer architecture.

    It consists of:
    1. An Encoder that processes input features (e.g., mel-spectrograms or SSL features)
       into a latent representation.
    2. A Vector Quantizer (VQ) that discretizes the latent representation into codes.
    3. A Decoder that generates the output feature (e.g., mel-spectrogram) from the codes,
       optional text conditioning, and a prompt, using a flow-matching diffusion process.
    """

    def __init__(
        self,
        mel_dim: int = 128,
        in_dim: int = 128,
        hidden_size: int = 1024,
        encoder_num_layers: int = 8,
        decoder_num_layers: int = 16,
        num_heads: int = 16,
        cond_drop_p: float = 0.2,  # drop code for decoder
        context_drop_p: float = 0.2,  # drop context (mel) for decoder
        down_sample_factor: int = 8,  # down sample factor for vq
        vq_emb_dim: int = 14,  # codebook size 2^vq_emb_dim, 2^14 = 16384
        use_text_cond: bool = True,  # use text cond for decoder
        text_vocab_size: int = 32100,  # vocab size
        cond_dim: int = 1024,
        cond_scale_factor: int = 1,
        sigma: float = 1e-5,
        time_scheduler: str = "linear",
        use_vq: bool = True,
        vq_type: str = "bsq",
        use_repa_loss: bool = False,
        cfg: Optional[Any] = None,
    ):
        """
        Initializes the TaDiCodec model.

        Args:
            mel_dim (int): Dimension of the mel-spectrogram.
            in_dim (int): Dimension of the encoder's input features.
            hidden_size (int): Hidden size of the transformer models.
            encoder_num_layers (int): Number of layers in the encoder transformer.
            decoder_num_layers (int): Number of layers in the decoder transformer.
            num_heads (int): Number of attention heads in the transformers.
            cond_drop_p (float): Dropout probability for the VQ code condition in the decoder.
            context_drop_p (float): Dropout probability for the prompt context in the decoder.
            down_sample_factor (int): Factor by which to downsample the latent representation before VQ.
            vq_emb_dim (int): Dimension of the vector quantizer's embedding space.
            use_text_cond (bool): Whether to use text embeddings as a condition in the decoder.
            text_vocab_size (int): Size of the text vocabulary.
            cond_dim (int): Dimension of the conditional input.
            cond_scale_factor (int): Scaling factor for the condition.
            sigma (float): Small constant used in the flow matching formula.
            time_scheduler (str): Type of time scheduler for diffusion (e.g., 'linear').
            use_vq (bool): Whether to use vector quantization.
            vq_type (str): Type of vector quantizer ('bsq' or 'simple').
            use_repa_loss (bool): Whether to use the representational alignment loss.
            cfg (Optional[Any]): A configuration object that can override the default parameters.
        """
        super().__init__()

        # Override parameters with values from the config object if provided
        mel_dim = (
            cfg.mel_dim if cfg is not None and hasattr(cfg, "mel_dim") else mel_dim
        )
        in_dim = cfg.in_dim if cfg is not None and hasattr(cfg, "in_dim") else in_dim
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        encoder_num_layers = (
            cfg.encoder_num_layers
            if cfg is not None and hasattr(cfg, "encoder_num_layers")
            else encoder_num_layers
        )
        decoder_num_layers = (
            cfg.decoder_num_layers
            if cfg is not None and hasattr(cfg, "decoder_num_layers")
            else decoder_num_layers
        )
        num_heads = (
            cfg.num_heads
            if cfg is not None and hasattr(cfg, "num_heads")
            else num_heads
        )
        cond_drop_p = (
            cfg.cond_drop_p
            if cfg is not None and hasattr(cfg, "cond_drop_p")
            else cond_drop_p
        )
        context_drop_p = (
            cfg.context_drop_p
            if cfg is not None and hasattr(cfg, "context_drop_p")
            else context_drop_p
        )
        down_sample_factor = (
            cfg.down_sample_factor
            if cfg is not None and hasattr(cfg, "down_sample_factor")
            else down_sample_factor
        )
        vq_emb_dim = (
            cfg.vq_emb_dim
            if cfg is not None and hasattr(cfg, "vq_emb_dim")
            else vq_emb_dim
        )
        use_text_cond = (
            cfg.use_text_cond
            if cfg is not None and hasattr(cfg, "use_text_cond")
            else use_text_cond
        )
        text_vocab_size = (
            cfg.text_vocab_size
            if cfg is not None and hasattr(cfg, "text_vocab_size")
            else text_vocab_size
        )
        cond_dim = (
            cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        )
        cond_scale_factor = (
            cfg.cond_scale_factor
            if cfg is not None and hasattr(cfg, "cond_scale_factor")
            else cond_scale_factor
        )
        sigma = cfg.sigma if cfg is not None and hasattr(cfg, "sigma") else sigma
        time_scheduler = (
            cfg.time_scheduler
            if cfg is not None and hasattr(cfg, "time_scheduler")
            else time_scheduler
        )
        use_vq = cfg.use_vq if cfg is not None and hasattr(cfg, "use_vq") else use_vq
        vq_type = (
            cfg.vq_type if cfg is not None and hasattr(cfg, "vq_type") else vq_type
        )
        use_repa_loss = (
            cfg.use_repa_loss
            if cfg is not None and hasattr(cfg, "use_repa_loss")
            else use_repa_loss
        )

        self.mel_dim = mel_dim
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.num_heads = num_heads
        self.cond_drop_p = cond_drop_p
        self.context_drop_p = context_drop_p
        self.vq_emb_dim = vq_emb_dim
        self.down_sample_factor = down_sample_factor
        self.use_text_cond = use_text_cond
        self.text_vocab_size = text_vocab_size
        self.cond_dim = cond_dim
        self.cond_scale_factor = cond_scale_factor
        self.sigma = sigma
        self.time_scheduler = time_scheduler
        self.use_vq = use_vq
        self.vq_type = vq_type
        self.use_repa_loss = use_repa_loss

        # Text embedding layer
        if self.use_text_cond:
            self.text_emb = nn.Embedding(text_vocab_size, hidden_size)

        # VQ related layers
        self.vq_in_linear = nn.Linear(hidden_size, vq_emb_dim)
        if self.use_vq:
            if self.vq_type == "bsq":
                self.bsq = BinarySphericalQuantizer(embed_dim=vq_emb_dim)
            else:
                self.bsq = SimpleQuantizer(embed_dim=vq_emb_dim)
        self.vq_out_linear = nn.Linear(vq_emb_dim, hidden_size)

        # Repa (Representational Alignment) MLP for auxiliary loss
        if self.use_repa_loss:
            self.repa_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, 1024),
            )
            self.repa_layer_idx = (
                6  # The decoder layer from which to extract hidden states
            )

        self.reset_parameters()

        # Encoder: A non-autoregressive Llama-style model without time/text conditioning
        self.encoder = DiffLlamaPrefix(
            hidden_size=hidden_size,
            num_layers=encoder_num_layers,
            num_heads=num_heads,
            in_dim=self.in_dim,
            out_dim=None,  # Outputs hidden states for VQ
            use_text_emb=False,
            use_diff_step=False,
            use_cond=False,
        )

        # Decoder: A non-autoregressive Llama-style model with time, text, and code conditioning
        self.decoder = DiffLlamaPrefix(
            hidden_size=hidden_size,
            num_layers=decoder_num_layers,
            num_heads=num_heads,
            in_dim=self.mel_dim,
            out_dim=self.mel_dim,
            use_text_emb=use_text_cond,
            use_diff_step=True,
            use_cond=True,
        )

    @torch.no_grad()
    def forward_diffusion(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward diffusion process based on flow matching.
        It takes the clean data `x` and a timestep `t` to produce a noisy sample `xt`.
        It also creates a prompt/target mask.

        Args:
            x (torch.Tensor): The clean input data (e.g., mel-spectrogram), shape `(B, T, mel_dim)`.
            t (torch.Tensor): The diffusion timestep for each sample in the batch, shape `(B,)`.

        Returns:
            Tuple[torch.Tensor, ...]:
                - xt (torch.Tensor): The noisy sample at time `t`, shape `(B, T, mel_dim)`.
                - z (torch.Tensor): The noise vector used, drawn from N(0, I).
                - new_t (torch.Tensor): The original `t` tensor.
                - prompt_len (torch.Tensor): The length of the prompt for each sample.
                - mask (torch.Tensor): A mask where 1 indicates the target (noisy) region and 0 indicates the prompt (clean) region.
        """
        new_t = t
        t = t.unsqueeze(-1).unsqueeze(-1)  # Reshape for broadcasting
        z = torch.randn(
            x.shape, dtype=x.dtype, device=x.device, requires_grad=False
        )  # (B, T, mel_dim)

        context_drop_p = self.context_drop_p

        # Randomly decide the length of the prompt (un-noised context)
        if torch.rand(1) > context_drop_p:
            prompt_len = torch.randint(
                min(x.shape[1] // 4, 5), int(x.shape[1] * 0.4), (x.shape[0],)
            ).to(x.device)
        else:
            # Drop the context entirely by setting prompt length to 0
            prompt_len = torch.zeros(x.shape[0], device=x.device)

        # Create a mask to distinguish prompt from target
        is_prompt = torch.zeros_like(x[:, :, 0])  # (B, T)
        col_indices = torch.arange(is_prompt.shape[1], device=prompt_len.device).repeat(
            is_prompt.shape[0], 1
        )  # (B, T)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # 1 if it's a prompt frame

        mask = torch.ones_like(x[:, :, 0])  # Mask is 1 for target, 0 for prompt
        mask[is_prompt.bool()] = 0
        mask = mask.unsqueeze(-1)  # (B, T, 1)

        # Flow matching formula: xt = (1 - (1 - sigma) * t) * x0 + t * x
        # where x0 ~ N(0, 1) and x is the clean data sample.
        # The equation is applied only to the target region.
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, new_t, prompt_len, mask

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        The main training-time forward pass of the model.

        Args:
            x (torch.Tensor): The target mel-spectrogram, shape `(B, T, mel_dim)`.
            x_mask (torch.Tensor): Padding mask for `x`, shape `(B, T)`.
            x_in (Optional[torch.Tensor]): Optional input for the encoder (e.g., SSL features). If None, `x` is used.
            text_ids (Optional[torch.Tensor]): Input text token IDs for conditioning, shape `(B, text_len)`.
            text_mask (Optional[torch.Tensor]): Padding mask for `text_ids`.

        Returns:
            Dict[str, Optional[torch.Tensor]]: A dictionary containing various tensors for loss computation,
            such as the predicted flow, the final predicted mel, the noise target, etc.
        """
        # 1. Encoder pass
        if x_in is None:
            # Use target mel as encoder input
            vq_emb_pre = self.encoder(x=x, x_mask=x_mask)
        else:
            # Use provided SSL features as encoder input
            vq_emb_pre = self.encoder(x=x_in, x_mask=x_mask)

        # 2. Downsampling before VQ
        _, T, _ = vq_emb_pre.shape
        vq_emb_pre = vq_emb_pre.transpose(1, 2)
        vq_emb_pre = F.interpolate(
            vq_emb_pre, size=T // self.down_sample_factor, mode="linear"
        )
        vq_emb_pre = vq_emb_pre.transpose(1, 2)

        # 3. Vector Quantization
        vq_emb_pre = self.vq_in_linear(vq_emb_pre)
        vq_emb_pre = F.normalize(vq_emb_pre, dim=-1)  # L2 normalize before quantization

        if self.use_vq:
            vq_emb, vq_loss, info = self.bsq(vq_emb_pre)
            commit_loss = info["commit_loss"]
        else:
            vq_emb = vq_emb_pre
            vq_loss = torch.tensor(0.0, device=x.device)
            commit_loss = torch.tensor(0.0, device=x.device)
            info = None

        vq_emb_post = self.vq_out_linear(vq_emb)

        # 4. Upsampling after VQ
        vq_emb_post = vq_emb_post.transpose(1, 2)
        vq_emb_post = F.interpolate(vq_emb_post, size=T, mode="linear")
        vq_emb_post = vq_emb_post.transpose(1, 2)

        # 5. Decoder with flow matching
        # Sample a random timestep t for each item in the batch
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)  # Clamp to avoid numerical issues at boundaries

        # Perform forward diffusion to get the noisy input `xt` and the noise `z`
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(x, t)
        noise = z

        # 6. Prepare conditions for the decoder
        if self.use_text_cond:
            text_emb = self.text_emb(text_ids)
        else:
            text_emb = None

        # Use the upsampled VQ embedding as the primary condition
        cond_emb = vq_emb_post
        # Apply condition dropout for classifier-free guidance training
        if torch.rand(1) < self.cond_drop_p:
            cond_emb = torch.zeros_like(cond_emb)

        # 7. Decoder pass
        if self.use_repa_loss:
            # If using Repa loss, we need to output intermediate hidden states
            flow_pred, hidden_states = self.decoder(
                x=xt,
                x_mask=x_mask,
                text_embedding=text_emb,
                text_mask=text_mask,
                cond=cond_emb,
                diffusion_step=new_t,
                output_hidden_states=True,
            )
            ssl_feat_pred = self.repa_mlp(hidden_states[self.repa_layer_idx])
        else:
            flow_pred = self.decoder(
                x=xt,
                x_mask=x_mask,
                text_embedding=text_emb,
                text_mask=text_mask,
                cond=cond_emb,
                diffusion_step=new_t,
            )
            ssl_feat_pred = None

        # Predict the clean data `x0_pred` from the noisy input `xt` and the predicted flow
        # x_pred = xt + (1 - t) * flow_pred
        x_pred = xt + (1 - t.unsqueeze(-1).unsqueeze(-1)) * flow_pred

        # Final mask should consider both the prompt/target mask and the original padding mask
        final_mask = mask * x_mask.unsqueeze(-1)

        return {
            "noise": noise,
            "x": x,
            "flow_pred": flow_pred,
            "x_pred": x_pred,
            "final_mask": final_mask,
            "prompt_len": prompt_len,
            "vq_loss": vq_loss,
            "commit_loss": commit_loss,
            "ssl_feat_pred": ssl_feat_pred,
        }

    @torch.no_grad()
    def encode(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes an input feature `x` into discrete VQ codes. (Inference)

        Args:
            x (torch.Tensor): Input feature, shape `(B, T, in_dim)`.
            x_mask (torch.Tensor): Padding mask for `x`, shape `(B, T)`.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - vq_emb (torch.Tensor): The quantized continuous embeddings.
                - indices (Optional[torch.Tensor]): The discrete code indices, if VQ is used.
        """
        # Encoder pass
        vq_emb_pre = self.encoder(x=x, x_mask=x_mask)

        # Downsampling
        _, T, _ = vq_emb_pre.shape
        vq_emb_pre = vq_emb_pre.transpose(1, 2)
        vq_emb_pre = F.interpolate(
            vq_emb_pre, size=T // self.down_sample_factor, mode="linear"
        )
        vq_emb_pre = vq_emb_pre.transpose(1, 2)

        # VQ
        vq_emb_pre = self.vq_in_linear(vq_emb_pre)
        vq_emb_pre = F.normalize(vq_emb_pre, dim=-1)  # L2 norm

        if self.use_vq:
            vq_emb, _, info = self.bsq(vq_emb_pre)
            indices = info["indices"]
        else:
            vq_emb = vq_emb_pre
            indices = None

        return vq_emb, indices

    @torch.no_grad()
    def index2vq(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts VQ code indices back to continuous embeddings.

        Args:
            indices (torch.Tensor): The discrete code indices.

        Returns:
            torch.Tensor: The corresponding continuous codebook embeddings.
        """
        return self.bsq.get_codebook_entry(indices).float()

    @torch.no_grad()
    def reverse_diffusion(
        self,
        vq_emb: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        prompt_mel: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        n_timesteps: int = 32,
        cfg: float = 1.0,
        rescale_cfg: float = 0.75,
    ) -> torch.Tensor:
        """
        Performs the reverse diffusion process to generate mel-spectrograms from conditions. (Inference)

        Args:
            vq_emb (Optional[torch.Tensor]): Pre-quantized embeddings.
            indices (Optional[torch.Tensor]): Discrete VQ code indices. If provided, `vq_emb` is ignored.
            text_ids (Optional[torch.Tensor]): Text token IDs for conditioning.
            prompt_mel (Optional[torch.Tensor]): A mel-spectrogram prompt.
            x_mask (Optional[torch.Tensor]): Padding mask for the target generation length.
            prompt_mask (Optional[torch.Tensor]): Padding mask for the prompt.
            text_mask (Optional[torch.Tensor]): Padding mask for the text.
            n_timesteps (int): Number of steps in the reverse diffusion process.
            cfg (float): Classifier-Free Guidance scale.
            rescale_cfg (float): Rescaling factor for CFG to prevent saturation.

        Returns:
            torch.Tensor: The generated mel-spectrogram.
        """
        if vq_emb is None:
            assert indices is not None, "Either vq_emb or indices must be provided"
            vq_emb = self.index2vq(indices.long())

        # Upsample VQ embeddings to match the target mel length
        vq_emb_post = self.vq_out_linear(vq_emb)
        vq_emb_post = vq_emb_post.transpose(1, 2)
        vq_emb_post = F.interpolate(
            vq_emb_post, scale_factor=self.down_sample_factor, mode="linear"
        )
        vq_emb_post = vq_emb_post.transpose(1, 2)

        # Prepare text embeddings
        if self.use_text_cond:
            text_emb = self.text_emb(text_ids)
            if text_mask is None:
                text_mask = torch.ones_like(text_ids)
        else:
            text_emb, text_mask = None, None

        cond_emb = vq_emb_post

        # Handle prompt
        if prompt_mel is None:
            prompt_mel = torch.zeros(
                cond_emb.shape[0], 0, self.mel_dim, device=cond_emb.device
            )

        prompt_len = prompt_mel.shape[1]
        target_len = cond_emb.shape[1] - prompt_len

        # Prepare masks
        if x_mask is None:
            x_mask = torch.ones(cond_emb.shape[0], target_len, device=cond_emb.device)
        if prompt_mask is None:
            prompt_mask = torch.ones(
                cond_emb.shape[0], prompt_len, device=cond_emb.device
            )

        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)

        # Initialize with random noise
        z = torch.randn(
            (cond_emb.shape[0], target_len, self.mel_dim),
            dtype=cond_emb.dtype,
            device=cond_emb.device,
        )
        xt = z
        h = 1.0 / n_timesteps

        # Iterative denoising loop (Euler method)
        for i in range(n_timesteps):
            # Concatenate prompt and current noisy sample
            xt_input = torch.cat([prompt_mel, xt], dim=1)
            # Calculate current timestep
            t = (0 + (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )

            # Get conditional flow prediction
            flow_pred = self.decoder(
                x=xt_input,
                x_mask=xt_mask,
                text_embedding=text_emb,
                text_mask=text_mask,
                cond=cond_emb,
                diffusion_step=t,
            )
            flow_pred = flow_pred[
                :, prompt_len:, :
            ]  # Extract flow for the target region

            # Classifier-Free Guidance (CFG)
            if cfg > 0 and self.use_text_cond:
                # Get unconditional flow prediction by dropping conditions
                uncond_flow_pred = self.decoder(
                    x=xt,
                    x_mask=x_mask,
                    text_embedding=None,  # Drop text
                    text_mask=None,
                    cond=torch.zeros_like(cond_emb[:, : xt.shape[1], :]),  # Drop code
                    diffusion_step=t,
                )

                # Combine conditional and unconditional predictions
                flow_pred_cfg = uncond_flow_pred + cfg * (flow_pred - uncond_flow_pred)

                # Rescale to prevent saturation, as in Stable Diffusion
                if rescale_cfg > 0:
                    flow_pred_std = flow_pred.std()
                    cfg_std = flow_pred_cfg.std()
                    # Avoid division by zero
                    if cfg_std > 1e-6:
                        rescale_flow_pred = flow_pred_cfg * (flow_pred_std / cfg_std)
                        flow_pred = (
                            rescale_cfg * rescale_flow_pred
                            + (1 - rescale_cfg) * flow_pred_cfg
                        )
                    else:
                        flow_pred = flow_pred_cfg
                else:
                    flow_pred = flow_pred_cfg

            # Update the noisy sample
            dxt = flow_pred * h
            xt = xt + dxt

        return xt

    def reset_parameters(self):
        """
        Applies custom weight initialization to the model's submodules.
        """

        def _reset_parameters(m: nn.Module):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

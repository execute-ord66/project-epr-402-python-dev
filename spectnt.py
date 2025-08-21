import torch as th
import torch.nn as nn


class Res2DMaxPoolModule(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(tuple(pooling))

        # residual
        self.diff = False
        if in_channels != out_channels:
            self.conv_3 = nn.Conv2d(
                in_channels, out_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(out_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class ResFrontEnd(nn.Module):
    """
    Adapted from Minz Won ResNet implementation.
    
    Original code: https://github.com/minzwon/semi-supervised-music-tagging-transformer/blob/master/src/modules.py
    """
    def __init__(self, in_channels, out_channels, freq_pooling, time_pooling):
        super(ResFrontEnd, self).__init__()
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.layer1 = Res2DMaxPoolModule(
            in_channels, out_channels, pooling=(freq_pooling[0], time_pooling[0]))
        self.layer2 = Res2DMaxPoolModule(
            out_channels, out_channels, pooling=(freq_pooling[1], time_pooling[1]))
        self.layer3 = Res2DMaxPoolModule(
            out_channels, out_channels, pooling=(freq_pooling[2], time_pooling[2]))

    def forward(self, hcqt):
        """
        Inputs:
            hcqt: [B, F, K, T]

        Outputs:
            out: [B, ^F, ^K, ^T]
        """
        # batch normalization
        out = self.input_bn(hcqt)

        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out


class SpecTNTBlock(nn.Module):
    def __init__(
        self, n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, dropout, use_tct
    ):
        super().__init__()

        self.D = embed_dim
        self.F = n_frequencies
        self.K = n_channels
        self.T = n_times

        # TCT: Temporal Class Token
        if use_tct:
            self.T += 1

        # Shared frequency-time linear layers
        self.D_to_K = nn.Linear(self.D, self.K)
        self.K_to_D = nn.Linear(self.K, self.D)

        # Spectral Transformer Encoder
        self.spectral_linear_in = nn.Linear(self.F+1, spectral_dmodel)
        self.spectral_encoder_layer = nn.TransformerEncoderLayer(
            d_model=spectral_dmodel, nhead=spectral_nheads, dim_feedforward=spectral_dimff, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.spectral_linear_out = nn.Linear(spectral_dmodel, self.F+1)

        # Temporal Transformer Encoder
        self.temporal_linear_in = nn.Linear(self.T, temporal_dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dmodel, nhead=temporal_nheads, dim_feedforward=temporal_dimff, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal_linear_out = nn.Linear(temporal_dmodel, self.T)

    def forward(self, spec_in, temp_in):
        """
        Inputs:
            spec_in: spectral embedding input [B, T, F+1, K]
            temp_in: temporal embedding input [B, T, 1, D]

        Outputs:
            spec_out: spectral embedding output [B, T, F+1, K]
            temp_out: temporal embedding output [B, T, 1, D]
        """
        # Element-wise addition between TE and FCT
        spec_in = spec_in + \
            nn.functional.pad(self.D_to_K(temp_in), (0, 0, 0, self.F))

        # Spectral Transformer
        spec_in = spec_in.flatten(0, 1).transpose(1, 2)  # [B*T, K, F+1]
        emb = self.spectral_linear_in(spec_in)  # [B*T, K, spectral_dmodel]
        spec_enc_out = self.spectral_encoder_layer(
            emb)  # [B*T, K, spectral_dmodel]
        spec_out = self.spectral_linear_out(spec_enc_out)  # [B*T, K, F+1]
        spec_out = spec_out.view(-1, self.T, self.K,
                                 self.F+1).transpose(2, 3)  # [B, T, F+1, K]

        # FCT slicing (first raw) + back to D
        temp_in = temp_in + self.K_to_D(spec_out[:, :, :1, :])  # [B, T, 1, D]

        # Temporal Transformer
        temp_in = temp_in.permute(0, 2, 3, 1).flatten(0, 1)  # [B, D, T]
        emb = self.temporal_linear_in(temp_in)  # [B, D, temporal_dmodel]
        temp_enc_out = self.temporal_encoder_layer(
            emb)  # [B, D, temporal_dmodel]
        temp_out = self.temporal_linear_out(temp_enc_out)  # [B, D, T]
        temp_out = temp_out.unsqueeze(1).permute(0, 3, 1, 2)  # [B, T, 1, D]

        return spec_out, temp_out


class SpecTNTModule(nn.Module):
    def __init__(
        self, n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, n_blocks, dropout, use_tct
    ):
        super().__init__()

        D = embed_dim
        F = n_frequencies
        K = n_channels
        T = n_times

        # Frequency Class Token
        self.fct = nn.Parameter(th.zeros(1, T, 1, K))

        # Frequency Positional Encoding
        self.fpe = nn.Parameter(th.zeros(1, 1, F+1, K))

        # TCT: Temporal Class Token
        if use_tct:
            self.tct = nn.Parameter(th.zeros(1, 1, 1, D))
        else:
            self.tct = None

        # Temporal Embedding
        self.te = nn.Parameter(th.rand(1, T, 1, D))

        # SpecTNT blocks
        self.spectnt_blocks = nn.ModuleList([
            SpecTNTBlock(
                n_channels,
                n_frequencies,
                n_times,
                spectral_dmodel,
                spectral_nheads,
                spectral_dimff,
                temporal_dmodel,
                temporal_nheads,
                temporal_dimff,
                embed_dim,
                dropout,
                use_tct
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        """
        Input:
            x: [B, T, F, K]

        Output:
            spec_emb: [B, T, F+1, K]
            temp_emb: [B, T, 1, D]
        """
        batch_size = len(x)

        # Initialize spectral embedding - concat FCT (first raw) + add FPE
        fct = th.repeat_interleave(self.fct, batch_size, 0)  # [B, T, 1, K]
        spec_emb = th.cat([fct, x], dim=2)  # [B, T, F+1, K]
        spec_emb = spec_emb + self.fpe
        if self.tct is not None:
            spec_emb = nn.functional.pad(
                spec_emb, (0, 0, 0, 0, 1, 0))  # [B, T+1, F+1, K]

        # Initialize temporal embedding
        temp_emb = th.repeat_interleave(self.te, batch_size, 0)  # [B, T, 1, D]
        if self.tct is not None:
            tct = th.repeat_interleave(self.tct, batch_size, 0)  # [B, 1, 1, D]
            temp_emb = th.cat([tct, temp_emb], dim=1)  # [B, T+1, 1, D]

        # SpecTNT blocks inference
        for block in self.spectnt_blocks:
            spec_emb, temp_emb = block(spec_emb, temp_emb)

        return spec_emb, temp_emb


class SpecTNTForSalience(nn.Module):
    """
    Adapts the provided SpecTNT implementation for salience map prediction.
    This model uses the ResFrontEnd and SpecTNTModule as a powerful encoder,
    and adds a new convolutional decoder to reconstruct the salience map.
    """
    def __init__(self, config):
        super().__init__()
        model_params = config['model_params']
        data_params = config['data_params']

        # --- 1. ENCODER PART ---

        # A. Configure and create the ResNet Front-End
        fe_in_channels = model_params['input_channels']
        fe_out_channels = model_params.get('fe_out_channels', 64)
        # The pooling factors determine how much the input is downsampled.
        # This MUST be matched by the decoder's upsampling.
        self.fe_freq_pooling = model_params.get('fe_freq_pooling', (2, 2, 2))
        self.fe_time_pooling = model_params.get('fe_time_pooling', (2, 2, 1))

        self.frontend = ResFrontEnd(
            in_channels=fe_in_channels,
            out_channels=fe_out_channels,
            freq_pooling=self.fe_freq_pooling,
            time_pooling=self.fe_time_pooling
        )

        # B. Configure and create the SpecTNT Main Module
        # We calculate the downsampled dimensions to pass to the transformer
        n_bins = data_params['n_octaves'] * data_params['bins_per_octave']
        patch_width = config['training_params']['patch_width']

        downsampled_freqs = n_bins // (self.fe_freq_pooling[0] * self.fe_freq_pooling[1] * self.fe_freq_pooling[2])
        downsampled_times = patch_width // (self.fe_time_pooling[0] * self.fe_time_pooling[1] * self.fe_time_pooling[2])

        self.main_model = SpecTNTModule(
            n_channels=fe_out_channels,
            n_frequencies=downsampled_freqs,
            n_times=downsampled_times,
            spectral_dmodel=model_params.get('spectral_dmodel', 128),
            spectral_nheads=model_params.get('spectral_nheads', 4),
            spectral_dimff=model_params.get('spectral_dimff', 256),
            temporal_dmodel=model_params.get('temporal_dmodel', 128),
            temporal_nheads=model_params.get('temporal_nheads', 4),
            temporal_dimff=model_params.get('temporal_dimff', 256),
            embed_dim=model_params.get('embed_dim', 128),
            n_blocks=model_params.get('n_blocks', 4),
            dropout=model_params.get('dropout', 0.1),
            use_tct=False # We don't need a final classification token for this task
        )

        # --- 2. DECODER PART ---
        # This part upsamples the features back to the original input resolution.
        # The stride in ConvTranspose2d must match the pooling in ResFrontEnd.
        self.decoder = nn.Sequential(
            # Input is [B, fe_out_channels, downsampled_freqs, downsampled_times]
            nn.ConvTranspose2d(fe_out_channels, fe_out_channels, kernel_size=3,
                               stride=(self.fe_freq_pooling[2], self.fe_time_pooling[2]),
                               padding=1, output_padding=(1,0)),
            nn.BatchNorm2d(fe_out_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(fe_out_channels, fe_out_channels, kernel_size=3,
                               stride=(self.fe_freq_pooling[1], self.fe_time_pooling[1]),
                               padding=1, output_padding=1),
            nn.BatchNorm2d(fe_out_channels),
            nn.ReLU(),
            
            nn.ConvTranspose2d(fe_out_channels, fe_out_channels // 2, kernel_size=3,
                               stride=(self.fe_freq_pooling[0], self.fe_time_pooling[0]),
                               padding=1, output_padding=1),
            nn.BatchNorm2d(fe_out_channels // 2),
            nn.ReLU(),

            # Final layer to produce a 1-channel salience map
            nn.Conv2d(fe_out_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Input HCQT of shape (batch, channels, bins, time)
        """
        # 1. ENCODER: Pass through Front-End to downsample
        # Input: [B, C, F, T] -> Output: [B, C_out, F_down, T_down]
        fe_out = self.frontend(x)

        # 2. ENCODER: Reshape and pass through SpecTNT
        # Input: [B, C_out, F_down, T_down] -> Permute: [B, T_down, F_down, C_out]
        main_in = fe_out.permute(0, 3, 2, 1)
        
        # Output is a tuple of (spectral_embedding, temporal_embedding)
        # We need the spectral embedding as it contains the frequency information
        spec_emb, _ = self.main_model(main_in)

        # 3. DECODER: Prepare features for decoding
        # The spec_emb has extra "class tokens" for frequency and time. We must remove them.
        # Shape: [B, T_down+1, F_down+1, C_out] -> Slice: [B, T_down, F_down, C_out]
        spec_emb_sliced = spec_emb[:, :, 1:, :] # Correct: Removes only the FCT from the frequency axis
        
        # Reshape for convolutional decoder
        # Input: [B, T_down, F_down, C_out] -> Permute: [B, C_out, F_down, T_down]
        decoder_in = spec_emb_sliced.permute(0, 3, 2, 1)

        # 4. DECODER: Upsample features to full resolution
        # Input: [B, C_out, F_down, T_down] -> Output: [B, 1, F, T]
        salience_map = self.decoder(decoder_in)

        return salience_map
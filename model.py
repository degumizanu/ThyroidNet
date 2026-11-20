import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ThyroidNet(nn.Module):

    def __init__(
        self,
        height=256,
        width=256,
        pretrained=True,
        alpha_max=0.015,
        dz=1.0
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.alpha_max = alpha_max
        self.dz = dz

        # ----------------------------------
        # 1. GLAND PREPROCESSING
        # ----------------------------------

        self.avg_pool = nn.AvgPool2d(
            kernel_size=7,
            stride=1,
            padding=3
        )

        # ----------------------------------
        # 2. CONVNEXT ENCODER
        # ----------------------------------

        weights = (
            models.ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained else None
        )

        backbone = models.convnext_tiny(weights=weights)

        original_conv = backbone.features[0][0]

        new_conv = nn.Conv2d(
            4, 96,
            kernel_size=4,
            stride=4
        )

        with torch.no_grad():
            new_conv.weight[:, :3].copy_(original_conv.weight)

            new_conv.weight[:, 3:4].copy_(
                original_conv.weight.mean(
                    dim=1, keepdim=True
                )
            )

            new_conv.bias.copy_(original_conv.bias)

        backbone.features[0][0] = new_conv

        self.enc_stem = backbone.features[0]
        self.enc_s1 = backbone.features[1:4]
        self.enc_s2 = backbone.features[4:6]
        self.enc_s3 = backbone.features[6:8]

        # ----------------------------------
        # 3. DECODER
        # ----------------------------------

        self.dec3 = self.conv_block(
            768 + 384, 384
        )

        self.dec2 = self.conv_block(
            384 + 192, 192
        )

        self.dec1 = self.conv_block(
            192 + 96, 96
        )

        self.final_up = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.SiLU()
        )

        # ----------------------------------
        # 4. PHYSICS PARAMETER HEAD
        # ----------------------------------

        # Three outputs:
        # 0: Nakagami m
        # 1: Nakagami Omega
        # 2: attenuation alpha

        self.head_phys = nn.Conv2d(
            32, 3, kernel_size=1
        )

        # ----------------------------------
        # 5. DEPTH GAIN HEAD
        # ----------------------------------

        self.gain_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.SiLU(),
            nn.Linear(256, height)
        )

        # ----------------------------------
        # 6. PHYSICS GUIDED ATTENTION
        # ----------------------------------

        # Inputs:
        # m, log(Omega), S_mean, T

        self.physics_attention = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Tanh()
        )

        # Learnable bounded attention strength
        self.attention_strength = nn.Parameter(
            torch.tensor(0.0)
        )

        # ----------------------------------
        # 7. SEGMENTATION HEAD
        # ----------------------------------

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1)
        )

    def conv_block(self, in_c, out_c):

        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.SiLU(),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.SiLU()
        )

    # ----------------------------------
    # GLAND ATTENTION
    # ----------------------------------

    def calculate_gland_attention(self, img):

        local_mean = self.avg_pool(img)
        local_sq_mean = self.avg_pool(img ** 2)

        local_var = (
            local_sq_mean - local_mean ** 2
        ).clamp_min(0)

        is_skin = torch.sigmoid(
            50 * (local_mean - 0.6)
        )

        is_dark = torch.sigmoid(
            50 * (0.2 - local_mean)
        )

        prob_brightness = torch.exp(
            -10 * (local_mean - 0.4) ** 2
        )

        prob_texture = torch.exp(
            -50 * (local_var - 0.02) ** 2
        )

        gland_prob = (
            prob_brightness
            * prob_texture
            * (1 - is_skin)
            * (1 - is_dark)
        )

        return gland_prob

    # ----------------------------------
    # NAKAGAMI EXPECTED AMPLITUDE
    # ----------------------------------

    def nakagami_mean(self, m, omega):

        # E[S] =
        # Gamma(m+0.5)/Gamma(m)
        # * sqrt(Omega/m)

        log_gamma_ratio = (
            torch.lgamma(m + 0.5)
            - torch.lgamma(m)
        )

        S_mean = (
            torch.exp(log_gamma_ratio)
            * torch.sqrt(omega / m)
        )

        return S_mean

    # ----------------------------------
    # BEER-LAMBERT ATTENUATION
    # ----------------------------------

    def beer_lambert(self, alpha):

        # alpha: [B, 1, H, W]
        # Depth axis = 2

        cumulative_alpha = (
            2 * self.dz
            * torch.cumsum(alpha, dim=2)
        )

        transmission = torch.exp(
            -cumulative_alpha
        )

        return transmission

    # ----------------------------------
    # DEPTH GAIN
    # ----------------------------------

    def compute_gain(self, bottleneck, H, W):

        B = bottleneck.shape[0]

        log_gain = self.gain_head(bottleneck)

        # [B, 1, height]
        log_gain = log_gain.unsqueeze(1)

        # Support variable image height
        log_gain = F.interpolate(
            log_gain,
            size=H,
            mode="linear",
            align_corners=False
        )

        # Smooth the depth profile
        log_gain = F.avg_pool1d(
            F.pad(log_gain, (2, 2), mode="replicate"),
            kernel_size=5,
            stride=1
        )

        # Fix arbitrary global gain scale
        log_gain = (
            log_gain
            - log_gain.mean(dim=-1, keepdim=True)
        )

        # Bound log-gain for numerical stability
        log_gain = 2.0 * torch.tanh(
            log_gain / 2.0
        )

        gain_vec = torch.exp(
            log_gain.squeeze(1)
        )

        gain_map = gain_vec[:, None, :, None]

        gain_map = gain_map.expand(
            B, 1, H, W
        )

        return gain_vec, gain_map

    # ----------------------------------
    # FORWARD
    # ----------------------------------

    def forward(self, x, return_aux=False):

        H, W = x.shape[-2:]

        if x.size(1) == 1:
            img_gray = x
            x_rgb = x.repeat(1, 3, 1, 1)

        else:
            img_gray = x.mean(
                dim=1, keepdim=True
            )
            x_rgb = x

        # ----------------------------------
        # GLAND ATTENTION
        # ----------------------------------

        gland_map = self.calculate_gland_attention(
            img_gray
        )

        x_in = torch.cat(
            [x_rgb, gland_map], dim=1
        )

        # ----------------------------------
        # ENCODER
        # ----------------------------------

        e1 = self.enc_stem(x_in)
        e2 = self.enc_s1(e1)
        e3 = self.enc_s2(e2)
        b = self.enc_s3(e3)

        # ----------------------------------
        # DECODER
        # ----------------------------------

        b_up = F.interpolate(
            b,
            size=e3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = self.dec3(
            torch.cat([b_up, e3], dim=1)
        )

        d3_up = F.interpolate(
            d3,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = self.dec2(
            torch.cat([d3_up, e2], dim=1)
        )

        d2_up = F.interpolate(
            d2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = self.dec1(
            torch.cat([d2_up, e1], dim=1)
        )

        features = F.interpolate(
            d1,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        features = self.final_up(features)

        # ----------------------------------
        # PHYSICS PARAMETER ESTIMATION
        # ----------------------------------

        phys = self.head_phys(features)

        # Nakagami shape
        # m >= 0.5, bounded at 5

        m = 0.5 + 4.5 * torch.sigmoid(
            phys[:, 0:1]
        )

        # Nakagami spread
        # Omega > 0

        omega = (
            F.softplus(phys[:, 1:2])
            + 1e-4
        )

        # Positive, bounded attenuation
        # alpha is effective attenuation
        # per depth step

        alpha_map = self.alpha_max * torch.sigmoid(
            phys[:, 2:3]
        )

        # ----------------------------------
        # NAKAGAMI BACKSCATTER
        # ----------------------------------

        S_map = self.nakagami_mean(
            m, omega
        )

        # ----------------------------------
        # BEER-LAMBERT TRANSMISSION
        # ----------------------------------

        T_map = self.beer_lambert(
            alpha_map
        )

        # ----------------------------------
        # DEPTH GAIN
        # ----------------------------------

        gain_vec, gain_map = self.compute_gain(
            b, H, W
        )

        # ----------------------------------
        # EXPECTED IMAGE RECONSTRUCTION
        # ----------------------------------

        I_rec = (
            S_map
            * T_map
            * gain_map
        )

        # ----------------------------------
        # PHYSICS-GUIDED ATTENTION
        # ----------------------------------

        physics_features = torch.cat(
            [
                m / 5.0,
                torch.log1p(omega),
                S_map,
                T_map
            ],
            dim=1
        )

        attention = self.physics_attention(
            physics_features
        )

        strength = (
            0.5 * torch.tanh(
                self.attention_strength
            )
        )

        guided_features = (
            features
            * (1 + strength * attention)
        )

        # ----------------------------------
        # SEGMENTATION
        # ----------------------------------

        logits = self.seg_head(
            guided_features
        )

        if return_aux:

            aux = {
                "m": m,
                "omega": omega,
                "T": T_map,
                "I_rec": I_rec,
                "gain_map": gain_map,
                "attention": attention
            }

            return (
                S_map,
                alpha_map,
                gain_vec,
                logits,
                gland_map,
                aux
            )

        # Compatible with the previous
        # five-output interface

        return (
            S_map,
            alpha_map,
            gain_vec,
            logits,
            gland_map
        )
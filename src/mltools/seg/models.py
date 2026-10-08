import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import (
    ConvNextBackbone,
    DINOv3ConvNextBackbone,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)
from segmentation_models_pytorch.decoders.manet.model import MAnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder

load_dotenv()


def _hf_token() -> str:
    """The facebook/dinov3-* checkpoints are gated on HuggingFace — loading
    them without a token that's been granted access fails with an opaque
    401 deep inside from_pretrained. Fail fast with a clear message instead,
    and don't rely on some other module having called load_dotenv() first
    (e.g. run.py's process never imports mltools.common.hf, so .env
    wouldn't otherwise get loaded before training starts)."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set (checked .env and the environment) — required to "
            "download the gated facebook/dinov3-* checkpoints."
        )
    return token


class DINOv3FPN(nn.Module):
    def __init__(self, out_channels=256, num_classes=1):
        super().__init__()

        self.encoder = DINOv3ConvNextBackbone.from_pretrained(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            out_indices=[1, 2, 3, 4],
            token=_hf_token(),
        )

        # ConvNeXt-Large channel counts for stages 0, 1, 2, 3
        stage_channels = [192, 384, 768, 1536]

        # Create lateral layers that map EACH stage to out_channels
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) for ch in stage_channels
        ])

        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in stage_channels
        ])

        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def get_decoder_parameters(self):
        params = []
        params.extend(list(self.lateral_layers.parameters()))
        params.extend(list(self.smooth_layers.parameters()))
        params.extend(list(self.head.parameters()))
        return params

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.encoder(x).feature_maps  # list of 4 tensors

        # 1. Start from the deepest feature map (stage 3, 1536 channels)
        p = self.lateral_layers[3](features[3])

        # 2. Iterate backwards from stage 2 down to 0
        for i in range(2, -1, -1):
            # Upsample the path from above
            p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)

            # Add lateral connection from current stage 'i'
            p = p + self.lateral_layers[i](features[i])

            # Apply smoothing
            p = self.smooth_layers[i](p)

        # 3. Final Head
        out = F.interpolate(p, size=input_size, mode='bilinear', align_corners=False)
        return self.head(out)


class DINOv3MAnet(nn.Module):
    # ConvNeXt-Large stage output channels, at strides 4/8/16/32.
    ENCODER_CHANNELS = [192, 384, 768, 1536]

    def _build_encoder(self):
        return DINOv3ConvNextBackbone.from_pretrained(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            out_indices=[1, 2, 3, 4],
            token=_hf_token(),
        )

    def _build_decoder(self, encoder_channels):
        """(decoder, channels it outputs). Overridden by subclasses that keep
        this encoder but swap the decoder.

        Note the block count: every smp decoder drops the first encoder entry
        ("same spatial resolution as the input") and pairs each block with a
        skip, so with 4 encoder channels there are only 3 skips left and zip()
        builds 3 blocks, not the 4 decoder_channels suggest. The stride-4
        feature is consumed as a skip, and the output is stride 8 with
        decoder_channels[2] channels — forward() then upsamples it."""
        decoder_channels = (256, 128, 64, 32)
        decoder = MAnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=4,  # Matches our 4 stages
            pab_channels=64,
        )
        return decoder, decoder_channels[2]

    def __init__(self, num_classes=1):
        super().__init__()

        # 1. DINOv3 backbone
        self.encoder = self._build_encoder()
        encoder_channels = self.ENCODER_CHANNELS

        # 2. SMP's internal decoder
        self.decoder, head_channels = self._build_decoder(encoder_channels)

        # 3. Final Head
        self.segmentation_head = nn.Sequential(
            # This 3x3 conv cleans up the interpolation artifacts
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            # This 1x1 conv does the final classification
            nn.Conv2d(head_channels, num_classes, kernel_size=1)
        )

    def get_encoder_parameters(self, stages=[2, 3]):
        params = []
        for name, param in self.encoder.named_parameters():
            if any(f"stages.{stage}" in name for stage in stages):
                params.append(param)
        return params

    def unfreeze_encoder_stages(self, stages=[2, 3]):
        params = self.get_encoder_parameters(stages)
        for param in params:
            param.requires_grad = True

    def get_decoder_parameters(self):
        params = []
        params.extend(list(self.decoder.parameters()))
        params.extend(list(self.segmentation_head.parameters()))
        return params

    def forward(self, x):
        features = self.encoder(x).feature_maps

        # Pass features through the SMP decoder
        decoder_output = self.decoder(features)

        upsampled_output = F.interpolate(decoder_output, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return self.segmentation_head(upsampled_output)


class ConvNeXtMAnet(DINOv3MAnet):
    """DINOv3MAnet with supervised ImageNet weights instead of DINOv3's
    self-supervised LVD-1689M ones.

    Same ConvNeXt-Large architecture and stage widths, same decoder, same head
    — the pretraining is the only thing that differs, which is what makes this
    a control for "does DINOv3 pretraining actually buy anything here". Not a
    gated repo, so no HF token is needed.
    """

    def _build_encoder(self):
        encoder = ConvNextBackbone.from_pretrained(
            "facebook/convnext-large-224-22k-1k",
            out_indices=[1, 2, 3, 4],
        )
        channels = list(encoder.channels)
        if channels != self.ENCODER_CHANNELS:
            raise ValueError(
                f"encoder stage widths {channels} do not match the decoder's "
                f"{self.ENCODER_CHANNELS}; pick a ConvNeXt-Large checkpoint."
            )
        return encoder


class DINOv3BaseMAnet(DINOv3MAnet):
    """DINOv3MAnet with the ConvNeXt-Base DINOv3 encoder (~88M) instead of
    Large (~198M): same pretraining, decoder and head, only the encoder size
    differs."""

    ENCODER_CHANNELS = [128, 256, 512, 1024]

    def _build_encoder(self):
        encoder = DINOv3ConvNextBackbone.from_pretrained(
            "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            out_indices=[1, 2, 3, 4],
            token=_hf_token(),
        )
        channels = list(encoder.channels)
        if channels != self.ENCODER_CHANNELS:
            raise ValueError(
                f"encoder stage widths {channels} do not match the decoder's {self.ENCODER_CHANNELS}."
            )
        return encoder


class SMPEncoderMAnet(DINOv3MAnet):
    """DINOv3MAnet's decoder and head on any segmentation_models_pytorch
    encoder (resnet50, mit_b5, ...), so the backbone is the only thing that
    differs from exp_reference.

    smp encoders emit 6 features (strides 1, 2, 4, 8, 16, 32) where the DINOv3
    backbone emits 4 (strides 4..32). Feeding all 6 to MAnetDecoder would give
    it 5 blocks decoding to full resolution instead of the reference's 3
    blocks ending at stride 8 — a better model, but then a win couldn't be
    attributed to the backbone. So the two finest features are dropped and the
    decoder is built exactly as in DINOv3MAnet.

    unfreeze_stages is not supported here: it matches ConvNeXt's `stages.N`
    parameter names, which resnet (layer1..4) and mit (block1..4) don't use.
    """

    def __init__(self, encoder_name, encoder_weights="imagenet", num_classes=1):
        # Read by _build_encoder, which super().__init__ calls. Plain strings,
        # so assigning them before nn.Module.__init__ is fine.
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        super().__init__(num_classes=num_classes)

    def _build_encoder(self):
        encoder = get_encoder(
            self.encoder_name,
            in_channels=3,
            depth=5,
            weights=self.encoder_weights,
        )
        # Keep strides 4..32 only, matching DINOv3MAnet's 4-entry convention.
        self.ENCODER_CHANNELS = list(encoder.out_channels)[2:]
        return encoder

    def get_encoder_parameters(self, stages=[2, 3]):
        raise NotImplementedError(
            f"unfreeze_stages is not supported for the smp encoder "
            f"'{self.encoder_name}': its parameters aren't named `stages.N`. "
            f"Use freeze_encoder: true / false instead."
        )

    def unfreeze_encoder_stages(self, stages=[2, 3]):
        self.get_encoder_parameters(stages)

    def forward(self, x):
        # smp encoders return a plain list, not a BackboneOutput.
        features = self.encoder(x)[2:]

        decoder_output = self.decoder(features)

        upsampled_output = F.interpolate(
            decoder_output, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return self.segmentation_head(upsampled_output)


class DINOv3Unet(DINOv3MAnet):
    """Same DINOv3 ConvNeXt-Large backbone as DINOv3MAnet, with a plain U-Net
    decoder instead of MAnet's.

    MAnet is a U-Net with attention added (PAB at the bottleneck, MFAB in
    place of the plain skip blocks), so this is the ablation that says whether
    that attention earns its cost. Chosen over U-Net++: its nested dense skips
    mainly pay off when the encoder is trained with them, and this encoder is
    frozen in the reference setup — while every extra nested block is
    decoder-side memory at 512x512.

    Two differences from DINOv3MAnet beyond the decoder blocks themselves,
    both from padding the encoder channels with a placeholder entry (see
    _build_decoder): all 4 encoder features are used as skips (MAnet loses the
    stride-4 one), and the last block decodes to the input resolution instead
    of the head upsampling stride-8 logits.
    """

    DECODER_CHANNELS = (256, 128, 64, 32)

    def _build_decoder(self, encoder_channels):
        # UnetDecoder drops encoder_channels[0] as "the input-resolution
        # feature", so pad the list to get one block per stage (4 blocks,
        # 4 skips) rather than the 3 the unpadded list would produce. The
        # padded value is never read.
        decoder = UnetDecoder(
            encoder_channels=[0, *encoder_channels],
            decoder_channels=self.DECODER_CHANNELS,
            n_blocks=len(self.DECODER_CHANNELS),
        )
        return decoder, self.DECODER_CHANNELS[-1]

    def forward(self, x):
        features = self.encoder(x).feature_maps
        # The placeholder that matches the padded channel list. UnetDecoder
        # drops features[0]'s channels but reads its spatial size, using it as
        # the target the last (skipless) block upsamples to — so passing the
        # input image makes the decoder output full resolution, and no
        # interpolation is needed before the head.
        decoder_output = self.decoder([x, *features])
        return self.segmentation_head(decoder_output)


class DINOv3UPerNet(nn.Module):
    """Same DINOv3 ConvNeXt-Large backbone as DINOv3MAnet, with UPerNet's
    PSP + FPN decoder instead of MAnet's. Drop-in alternative for
    train.model._target_."""

    def __init__(self, num_classes=1, decoder_channels=256):
        super().__init__()

        self.encoder = DINOv3ConvNextBackbone.from_pretrained(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            out_indices=[1, 2, 3, 4],
            token=_hf_token(),
        )
        # ConvNeXt-Large stage output channels, at strides 4/8/16/32
        encoder_channels = [192, 384, 768, 1536]

        # UPerNetDecoder is written against smp's encoder convention, where
        # features run [1/1, 1/2, 1/4, ...] and it discards the first two
        # itself. This backbone only emits 1/4 downwards, so pad both the
        # channel list and (in forward) the feature list with placeholders
        # that land in the slice the decoder throws away.
        self.decoder = UPerNetDecoder(
            encoder_channels=[0, 0, *encoder_channels],
            encoder_depth=5,
            decoder_channels=decoder_channels,
        )

        # UPerNet's decoder output is 1/4 resolution with decoder_channels
        # channels, so classify first and upsample the logits — upsampling
        # 256 channels to full resolution before the head would cost ~16x
        # the activation memory for no benefit.
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def get_encoder_parameters(self, stages=[2, 3]):
        params = []
        for name, param in self.encoder.named_parameters():
            if any(f"stages.{stage}" in name for stage in stages):
                params.append(param)
        return params

    def unfreeze_encoder_stages(self, stages=[2, 3]):
        params = self.get_encoder_parameters(stages)
        for param in params:
            param.requires_grad = True

    def get_decoder_parameters(self):
        params = []
        params.extend(list(self.decoder.parameters()))
        params.extend(list(self.segmentation_head.parameters()))
        return params

    def forward(self, x):
        features = self.encoder(x).feature_maps
        decoder_output = self.decoder([None, None, *features])
        logits = self.segmentation_head(decoder_output)
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class DINOv3Mask2Former(nn.Module):
    """DINOv3 ConvNeXt-Large backbone with Mask2Former's query-based mask
    decoder.

    Unlike the dense decoders above, Mask2Former is a set-prediction model:
    it computes its own Hungarian-matched loss from per-instance mask/class
    targets rather than a per-pixel one. It therefore opts out of SegLoss via
    `computes_own_loss`, and SegModule calls compute_loss() instead. forward()
    still returns dense per-pixel logits, so the metrics, sliding-window
    inference and predict() all keep working unchanged.

    Targets are thresholded at mask_threshold and split into objects with
    connected components (see _build_targets) — the soft confidence values
    (0.2/0.6/1.0) the dense decoders train against are not representable in
    this formulation.
    """

    computes_own_loss = True

    def __init__(
        self,
        num_classes=1,
        num_queries=100,
        mask_threshold=0.5,
        hidden_dim=256,
        decoder_layers=10,
        encoder_layers=6,
        dim_feedforward=2048,
        encoder_feedforward_dim=1024,
        feature_size=256,
        mask_feature_size=256,
        no_object_weight=0.1,
    ):
        """Capacity knobs are exposed because the defaults are Mask2Former's,
        tuned for COCO/ADE20K-scale data; on a few hundred training images the
        decoder overfits badly. hidden_dim, feature_size and mask_feature_size
        may differ from one another — the model inserts an input projection
        when feature_size != hidden_dim and an MLP for hidden_dim ->
        mask_feature_size — but hidden_dim and feature_size must stay
        divisible by num_attention_heads (8).

        config.dropout is deliberately left at 0 and not exposed: transformers
        wires it to LayerDrop as well as to ordinary dropout, and the LayerDrop
        path is broken for this architecture. Skipping a decoder layer bypasses
        the regeneration of the cross-attention mask while level_index keeps
        advancing, so the next layer attends a feature level whose resolution
        no longer matches its mask and dies on a shape mismatch.
        """
        super().__init__()

        backbone = DINOv3ConvNextBackbone.from_pretrained(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            out_indices=[1, 2, 3, 4],
            token=_hf_token(),
        )
        config = Mask2FormerConfig(
            backbone_config=backbone.config,
            num_labels=num_classes,
            num_queries=num_queries,
            feature_strides=(4, 8, 16, 32),
            hidden_dim=hidden_dim,
            decoder_layers=decoder_layers,
            encoder_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            encoder_feedforward_dim=encoder_feedforward_dim,
            feature_size=feature_size,
            mask_feature_size=mask_feature_size,
            # Weight of the "no object" class in the classification loss.
            # The 0.1 default is tuned for COCO/ADE20K, where nearly every
            # image contains objects; here 31% of images have no knots at all
            # and the model fires spurious queries on more than half of them,
            # so raising this is the direct lever on false positives.
            no_object_weight=no_object_weight,
        )
        self.model = Mask2FormerForUniversalSegmentation(config)
        # Mask2FormerPixelLevelModule builds its backbone via
        # load_backbone(config), which initialises it randomly from the config
        # — swap in the pretrained DINOv3 weights. Safe because the pixel
        # decoder is sized from encoder.channels, identical either way since
        # both come from the same config.
        self.model.model.pixel_level_module.encoder = backbone

        self.num_classes = num_classes
        self.mask_threshold = mask_threshold

    @property
    def encoder(self):
        """SegModule.freeze_encoder()/unfreeze_encoder() reach for .encoder."""
        return self.model.model.pixel_level_module.encoder

    def get_encoder_parameters(self, stages=[2, 3]):
        params = []
        for name, param in self.encoder.named_parameters():
            if any(f"stages.{stage}" in name for stage in stages):
                params.append(param)
        return params

    def unfreeze_encoder_stages(self, stages=[2, 3]):
        for param in self.get_encoder_parameters(stages):
            param.requires_grad = True

    def get_decoder_parameters(self):
        encoder_params = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.model.parameters() if id(p) not in encoder_params]

    def _build_targets(self, targets):
        """(B, C, H, W) soft masks -> the per-image mask/class lists
        Mask2Former's matcher expects — one target per *object*, which is
        what makes the set-prediction formulation worth anything.

        The formatted dataset max-merges every annotation of a class into a
        single channel, so instances are recovered with connected components
        on the thresholded mask. That splits any objects which don't touch:
        measured against the raw COCO annotations, only ~7% of wood knots
        have even a bbox-overlapping neighbour (and overlapping boxes often
        still have disjoint masks), so the large majority come back separate.
        Objects that genuinely touch stay merged, as they already do for the
        connected-components GMS metric.

        Images with no foreground contribute empty tensors, which the matcher
        handles as "every query should predict no-object".
        """
        binary = (targets > self.mask_threshold).detach().cpu().numpy().astype(np.uint8)
        device = targets.device
        spatial = binary.shape[-2:]

        mask_labels, class_labels = [], []
        for sample in binary:
            masks, labels = [], []
            for class_idx in range(self.num_classes):
                count, components = cv2.connectedComponents(
                    sample[class_idx], connectivity=8
                )
                for instance in range(1, count):  # component 0 is background
                    masks.append(components == instance)
                    labels.append(class_idx)
            if masks:
                mask_labels.append(
                    torch.from_numpy(np.stack(masks)).float().to(device)
                )
                class_labels.append(
                    torch.tensor(labels, dtype=torch.long, device=device)
                )
            else:
                mask_labels.append(
                    torch.zeros((0, *spatial), dtype=torch.float, device=device)
                )
                class_labels.append(
                    torch.zeros((0,), dtype=torch.long, device=device)
                )
        return mask_labels, class_labels

    def _dense_logits(self, outputs, size):
        """Collapse the query predictions into a dense per-pixel map (the
        standard Mask2Former semantic-inference rule), then convert back to
        logits — everything downstream (SegModule.forward's sigmoid, the
        metrics' `> 0.0` threshold) assumes logits, not probabilities."""
        masks = F.interpolate(
            outputs.masks_queries_logits, size=size, mode="bilinear", align_corners=False
        )
        mask_probs = masks.sigmoid()
        # drop the trailing no-object logit
        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        semseg = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        eps = torch.finfo(semseg.dtype).eps
        return torch.logit(semseg.clamp(eps, 1 - eps))

    def compute_loss(self, images, targets):
        """Returns (loss, dense_logits) so the caller gets both from a single
        forward pass — the loss needs the query outputs, the metrics need the
        dense map."""
        mask_labels, class_labels = self._build_targets(targets)
        outputs = self.model(
            pixel_values=images, mask_labels=mask_labels, class_labels=class_labels
        )
        # The dense map only feeds the metrics, which detach anyway — building
        # it inside the graph would retain a (B, Q, H, W) interpolation and its
        # sigmoid for a backward pass that never traverses them.
        with torch.no_grad():
            logits = self._dense_logits(outputs, images.shape[-2:])
        return outputs.loss, logits

    def forward(self, x):
        return self._dense_logits(self.model(pixel_values=x), x.shape[-2:])

    # Keep threshold of transformers' Mask2FormerImageProcessor
    # .post_process_instance_segmentation (its default), not tuned on our data.
    INSTANCE_SCORE_THRESHOLD = 0.5

    @torch.no_grad()
    def predict_instances(self, x):
        """Mask2Former's own instances, for final predictions only (seg's
        predict() calls this when a model has it; training and the logged
        metrics use forward()).

        One pass over the whole image, since instances can't be blended across
        sliding windows the way dense logits are; the input is padded to a
        multiple of the backbone stride and the masks cropped back. The
        post-processing is transformers' post_process_instance_segmentation
        rule: each query's best class (no-object excluded) scored by class
        probability x mean mask probability inside its mask, masks = logit > 0,
        kept if the score reaches INSTANCE_SCORE_THRESHOLD and the mask is
        non-empty. Unlike transformers, masks are binarised at the image
        resolution rather than at a fixed 384x384 and then upsampled.

        x: (B, 3, H, W) normalised images. Returns one (masks (N, H, W) bool,
        labels (N,) in 1..num_classes, scores (N,)) tuple per image, on CPU.
        Overlapping instances stay separate."""
        h, w = x.shape[-2:]
        stride = 32
        x = F.pad(x, (0, (-w) % stride, 0, (-h) % stride))
        outputs = self.model(pixel_values=x)
        mask_logits = F.interpolate(
            outputs.masks_queries_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )[..., :h, :w]
        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]  # drop no-object

        results = []
        for probs, logits in zip(class_probs, mask_logits):
            num_queries, num_classes = probs.shape
            scores, flat = probs.flatten().topk(num_queries)
            query = torch.div(flat, num_classes, rounding_mode="floor")
            labels = flat % num_classes + 1
            logits = logits[query]
            masks = logits > 0
            area = masks.flatten(1).sum(1)
            mask_scores = (logits.sigmoid() * masks).flatten(1).sum(1) / (area + 1e-6)
            scores = scores * mask_scores
            keep = (scores >= self.INSTANCE_SCORE_THRESHOLD) & (area > 0)
            results.append((masks[keep].cpu(), labels[keep].cpu(), scores[keep].cpu()))
        return results

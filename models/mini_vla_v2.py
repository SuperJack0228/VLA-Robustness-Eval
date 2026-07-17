"""Causally grounded, interaction-aware MiniVLA release policy."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer


DEFAULT_LANGUAGE_MODEL = "distilbert-base-uncased"
DEFAULT_HIDDEN_DIM = 512
DEFAULT_CHUNK_SIZE = 20
DEFAULT_HISTORY_LENGTH = 5
DEFAULT_STATE_DIM = 17
NUM_EXPERT_PHASES = 10
ARCHITECTURE_VERSION = 4


def replace_batch_norm_with_group_norm(
    module: nn.Module,
    preferred_groups: int = 32,
) -> nn.Module:
    """Replace every BatchNorm2d without retaining batch-dependent state."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = min(preferred_groups, child.num_features)
            while groups > 1 and child.num_features % groups != 0:
                groups //= 2
            replacement = nn.GroupNorm(
                num_groups=groups,
                num_channels=child.num_features,
                eps=child.eps,
                affine=True,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batch_norm_with_group_norm(child, preferred_groups)
    return module


class SpatialVisionEncoder(nn.Module):
    """Shared ResNet50 that preserves a 7x7 feature map for 112px images."""

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        replace_batch_norm_with_group_norm(resnet)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        for frozen_stage in (self.stem, self.layer1, self.layer2):
            frozen_stage.requires_grad_(False)

        self.projection = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        self.spatial_position = nn.Parameter(
            torch.randn(1, hidden_dim, 7, 7) * 0.02
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.stem(image)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.projection(features)
        position = F.interpolate(
            self.spatial_position,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = features + position
        tokens = features.flatten(2).transpose(1, 2)
        return self.output_norm(tokens)


class MiniVLAV2(nn.Module):
    """ACT-style decoder over language, spatial vision, and proprioception tokens."""

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        language_model_name: str = DEFAULT_LANGUAGE_MODEL,
        language_max_length: int = 16,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.15,
        architecture_version: int = ARCHITECTURE_VERSION,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"MiniVLA V2 requires architecture_version={ARCHITECTURE_VERSION}"
            )
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.history_length = history_length
        self.language_model_name = language_model_name
        self.language_max_length = language_max_length
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.nhead = nhead
        self.dropout = dropout
        self.architecture_version = architecture_version

        self.language_tokenizer = AutoTokenizer.from_pretrained(
            language_model_name,
            local_files_only=local_files_only,
        )
        self.language_model = AutoModel.from_pretrained(
            language_model_name,
            local_files_only=local_files_only,
        )
        self.language_model.requires_grad_(False)
        self.language_model.eval()
        self._language_cache: dict[str, torch.Tensor] = {}
        self.language_projection = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.vision_encoder = SpatialVisionEncoder(hidden_dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_position = nn.Parameter(
            torch.randn(history_length, hidden_dim) * 0.02
        )
        # Fusion, language, agentview, wrist, proprioception, and target markers.
        self.modality_embedding = nn.Parameter(torch.randn(6, hidden_dim) * 0.02)
        self.fusion_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.multimodal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

        self.action_queries = nn.Parameter(
            torch.randn(chunk_size, hidden_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.interaction_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.interaction_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )
        self.phase_embedding = nn.Parameter(
            torch.randn(NUM_EXPERT_PHASES, hidden_dim) * 0.02
        )
        self.pose_head = nn.Linear(hidden_dim, 6)
        self.gripper_head = nn.Linear(hidden_dim, 1)
        self.phase_head = nn.Linear(hidden_dim, NUM_EXPERT_PHASES)
        grounding_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.grounding_decoder = nn.TransformerDecoder(
            grounding_layer,
            num_layers=2,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.grounding_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )
        self.target_position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.target_class_head = nn.Linear(hidden_dim, 3)

    def train(self, mode: bool = True):
        super().train(mode)
        self.language_model.eval()
        return self

    def encode_language(
        self,
        instructions: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not instructions or any(not text.strip() for text in instructions):
            raise ValueError("Instructions must contain non-empty strings")
        unique = list(dict.fromkeys(instructions))
        missing = [text for text in unique if text not in self._language_cache]
        if missing:
            encoded = self.language_tokenizer(
                missing,
                padding=True,
                truncation=True,
                max_length=self.language_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = self.language_model(**encoded).last_hidden_state
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for index, (text, length) in enumerate(zip(missing, lengths)):
                self._language_cache[text] = hidden[
                    index, : int(length)
                ].detach().cpu()

        cached = [self._language_cache[text] for text in instructions]
        max_length = max(tokens.shape[0] for tokens in cached)
        output = torch.zeros(
            len(cached),
            max_length,
            self.language_model.config.hidden_size,
            dtype=cached[0].dtype,
            device=device,
        )
        padding_mask = torch.ones(
            len(cached),
            max_length,
            dtype=torch.bool,
            device=device,
        )
        for index, tokens in enumerate(cached):
            length = tokens.shape[0]
            output[index, :length] = tokens.to(device)
            padding_mask[index, :length] = False
        return self.language_projection(output), padding_mask

    def forward(
        self,
        image_agentview: torch.Tensor,
        image_wrist: torch.Tensor,
        state_history: torch.Tensor,
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        batch_size = image_agentview.shape[0]
        expected_image_shape = (batch_size, 3)
        if image_agentview.shape[:2] != expected_image_shape or image_agentview.ndim != 4:
            raise ValueError(
                "image_agentview must be a single RGB frame per sample with "
                f"shape (B, 3, H, W), got {tuple(image_agentview.shape)}"
            )
        if image_wrist.shape != image_agentview.shape:
            raise ValueError("image_wrist must match image_agentview")
        expected_state = (batch_size, self.history_length, self.state_dim)
        if tuple(state_history.shape) != expected_state:
            raise ValueError(
                f"state_history must be {expected_state}, got {tuple(state_history.shape)}"
            )
        if len(instructions) != batch_size:
            raise ValueError(
                f"Expected {batch_size} instructions, got {len(instructions)}"
            )

        combined_images = torch.cat([image_agentview, image_wrist], dim=0)
        combined_tokens = self.vision_encoder(combined_images)
        agent_tokens, wrist_tokens = combined_tokens.chunk(2, dim=0)
        agent_tokens = agent_tokens + self.modality_embedding[2]
        wrist_tokens = wrist_tokens + self.modality_embedding[3]
        grounding_agent_tokens = agent_tokens

        proprio_tokens = self.proprio_encoder(state_history)
        proprio_tokens = (
            proprio_tokens
            + self.temporal_position.unsqueeze(0)
            + self.modality_embedding[4]
        )
        language_tokens, language_padding = self.encode_language(
            instructions,
            image_agentview.device,
        )
        language_tokens = language_tokens + self.modality_embedding[1]

        # Grounding is intentionally isolated from wrist and proprioceptive tokens.
        # This forces target localization to depend on language and the fixed camera.
        grounding_memory = torch.cat(
            [language_tokens, grounding_agent_tokens], dim=1
        )
        grounding_suffix = torch.zeros(
            batch_size,
            grounding_agent_tokens.shape[1],
            dtype=torch.bool,
            device=image_agentview.device,
        )
        grounding_padding = torch.cat(
            [language_padding, grounding_suffix],
            dim=1,
        )
        grounding_token = self.grounding_decoder(
            tgt=self.grounding_query.expand(batch_size, -1, -1),
            memory=grounding_memory,
            memory_key_padding_mask=grounding_padding,
        )[:, 0]
        policy_target_token = (
            grounding_token.unsqueeze(1) + self.modality_embedding[5]
        )

        fusion = self.fusion_token.expand(batch_size, -1, -1)
        fusion = fusion + self.modality_embedding[0]

        memory = torch.cat(
            [
                fusion,
                language_tokens,
                policy_target_token,
                agent_tokens,
                wrist_tokens,
                proprio_tokens,
            ],
            dim=1,
        )
        prefix_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=image_agentview.device,
        )
        suffix_mask = torch.zeros(
            batch_size,
            1
            + agent_tokens.shape[1]
            + wrist_tokens.shape[1]
            + self.history_length,
            dtype=torch.bool,
            device=image_agentview.device,
        )
        memory_padding_mask = torch.cat(
            [prefix_mask, language_padding, suffix_mask],
            dim=1,
        )
        memory = self.multimodal_encoder(
            memory,
            src_key_padding_mask=memory_padding_mask,
        )

        interaction_logits = self.interaction_head(memory[:, 0])
        interaction_token = self.interaction_projection(
            torch.sigmoid(interaction_logits)
        ).unsqueeze(1)
        interaction_token = interaction_token + self.interaction_embedding
        action_memory = torch.cat([memory, interaction_token], dim=1)
        interaction_padding = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=image_agentview.device,
        )
        action_memory_padding = torch.cat(
            [memory_padding_mask, interaction_padding],
            dim=1,
        )

        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.action_decoder(
            tgt=queries,
            memory=action_memory,
            memory_key_padding_mask=action_memory_padding,
        )
        phase_logits = self.phase_head(decoded)
        phase_context = torch.softmax(phase_logits, dim=-1) @ self.phase_embedding
        conditioned = decoded + phase_context
        return {
            "pose": self.pose_head(conditioned),
            "gripper_logits": self.gripper_head(conditioned).squeeze(-1),
            "phase_logits": phase_logits,
            "interaction_logits": interaction_logits,
            "target_position": self.target_position_head(grounding_token),
            "target_class_logits": self.target_class_head(grounding_token),
        }

    def model_config(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "chunk_size": self.chunk_size,
            "history_length": self.history_length,
            "language_model_name": self.language_model_name,
            "language_max_length": self.language_max_length,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "nhead": self.nhead,
            "dropout": self.dropout,
            "architecture_version": self.architecture_version,
        }

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("language_model.")
        }

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key for key in missing if not key.startswith("language_model.")
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing={invalid_missing}, "
                f"unexpected={unexpected}"
            )


if __name__ == "__main__":
    model = MiniVLAV2(local_files_only=True)
    output = model(
        image_agentview=torch.randn(2, 3, 112, 112),
        image_wrist=torch.randn(2, 3, 112, 112),
        state_history=torch.randn(2, 5, 17),
        instructions=[
            "Pick up the red cube",
            "Push the blue ball away from the robot",
        ],
    )
    for name, value in output.items():
        print(f"{name}: {tuple(value.shape)}", flush=True)

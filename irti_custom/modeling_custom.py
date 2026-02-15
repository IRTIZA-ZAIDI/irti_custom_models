import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput


def _assert_encoder_only_config(base_cfg: Any) -> None:
    """
    Enforce "encoder-only" using REAL, commonly-present HF config fields.
    - Encoder-decoder models (T5/BART) have is_encoder_decoder=True.
    - Decoder-only models (GPT-like) typically have is_decoder=True.
    These are standard HF config attributes (when applicable).
    """
    if getattr(base_cfg, "is_encoder_decoder", False):
        raise ValueError(
            "Unsupported base model: encoder-decoder architecture detected "
            "(config.is_encoder_decoder=True). Please use an encoder-only model."
        )

    # For decoder-only models, HF configs often set is_decoder=True (and is_encoder_decoder=False)
    if getattr(base_cfg, "is_decoder", False) and not getattr(
        base_cfg, "is_encoder_decoder", False
    ):
        raise ValueError(
            "Unsupported base model: decoder-only architecture detected "
            "(config.is_decoder=True). Please use an encoder-only model."
        )


@torch.no_grad()
def _assert_encoder_outputs(model: nn.Module, base_model_name_or_path: str) -> None:
    """
    Runtime capability check: encoder models loaded via AutoModel should return last_hidden_state.
    We do a tiny dry-run to ensure we’re not loading something incompatible.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([[0, 1]], device=device)  # tiny dummy
    attention_mask = torch.tensor([[1, 1]], device=device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    if not hasattr(out, "last_hidden_state"):
        raise ValueError(
            f"Base model '{base_model_name_or_path}' did not return last_hidden_state. "
            "This package requires an encoder model that returns last_hidden_state."
        )


def _optional_strict_mlm_check(
    base_model_name_or_path: str,
    trust_remote_code: bool,
    **kwargs,
) -> None:
    """
    Optional: user-requested style check.
    This ensures the identifier supports AutoModelForMaskedLM.
    Note: This can reject valid encoders that don’t ship an MLM mapping.
    """
    try:
        _ = AutoModelForMaskedLM.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(
            f"Strict MLM check failed for '{base_model_name_or_path}'. "
            "This likely means the model is not an MLM-capable encoder in transformers. "
            f"Original error: {repr(e)}"
        )


class MaskedSeqModelingConfig(PretrainedConfig):
    model_type = "irti_masked_seq_modeling"

    def __init__(
        self,
        num_labels: int = 2,
        base_model_name_or_path: Optional[str] = None,
        base_model_config: Optional[Dict[str, Any]] = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        trust_remote_code: bool = False,
        strict_mlm_check: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_labels = int(num_labels)
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_config = base_model_config or {}
        self.trust_remote_code = bool(trust_remote_code)

        # optional extra strict behavior
        self.strict_mlm_check = bool(strict_mlm_check)

        # label maps (HF convention)
        if id2label is None:
            id2label = {i: str(i) for i in range(self.num_labels)}
        if label2id is None:
            label2id = {v: int(k) for k, v in id2label.items()}

        self.id2label = {int(k): str(v) for k, v in id2label.items()}
        self.label2id = {str(k): int(v) for k, v in label2id.items()}


class AutoModelForMaskedSeqModeling(PreTrainedModel):
    """
    Encoder-only base + (mask_indices -> gather -> multilabel head)

    Matches your desired print style:
      AutoModelForMaskedSeqModeling(
        (bert): ...
        (head): ...
      )
    """

    config_class = MaskedSeqModelingConfig

    def __init__(self, config: MaskedSeqModelingConfig):
        super().__init__(config)

        # 1) Rebuild base config
        base_config = None
        if (
            isinstance(config.base_model_config, dict)
            and len(config.base_model_config) > 0
        ):
            try:
                base_config = AutoConfig.from_dict(config.base_model_config)
            except Exception:
                base_config = None

        if base_config is None:
            if not config.base_model_name_or_path:
                raise ValueError(
                    "Missing base model information. Provide `base_model_name_or_path` or `base_model_config`."
                )
            base_config = AutoConfig.from_pretrained(
                config.base_model_name_or_path,
                trust_remote_code=config.trust_remote_code,
            )

        # 2) Hard enforce encoder-only based on REAL config fields
        _assert_encoder_only_config(base_config)

        # 3) Build encoder skeleton
        self.bert = AutoModel.from_config(
            base_config,
            trust_remote_code=config.trust_remote_code,
        )

        # 4) Hidden size inference
        hidden_size = getattr(base_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(base_config, "d_model", None)
        if hidden_size is None:
            raise ValueError(
                "Could not infer hidden size from base model config (hidden_size or d_model)."
            )

        self.num_labels = config.num_labels

        # head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.num_labels),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_indices: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        if mask_indices is None:
            raise ValueError("mask_indices is required. Shape: [batch, num_masks].")

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state  # [B, T, H]

        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(
            -1, -1, sequence_output.size(-1)
        )
        mask_embeddings = torch.gather(
            sequence_output, 1, mask_indices_expanded
        )  # [B, M, H]

        logits = self.head(mask_embeddings)  # [B, M, L]

        loss = None
        if labels is not None:
            active_loss_mask = labels[:, :, 0] != -100
            active_logits = logits[active_loss_mask]
            active_labels = labels[active_loss_mask].float()
            loss = self.loss_fn(active_logits, active_labels)

        return MaskedLMOutput(loss=loss, logits=logits)

    @classmethod
    def from_encoder(
        cls,
        base_model_name_or_path: str,
        num_labels: int,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        trust_remote_code: bool = False,
        strict_mlm_check: bool = False,
        **kwargs,
    ) -> "AutoModelForMaskedSeqModeling":
        """
        Create a NEW model from an encoder checkpoint (hub id or local path).
        Initializes a fresh head (random weights) + loads encoder weights.
        """

        # Optional extra strict check: must support MLM loader
        if strict_mlm_check:
            _optional_strict_mlm_check(
                base_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # Load config first to enforce encoder-only before loading full weights
        base_cfg = AutoConfig.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        _assert_encoder_only_config(base_cfg)

        # Load base encoder weights
        base_model = AutoModel.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # Runtime capability check: must return last_hidden_state
        _assert_encoder_outputs(base_model, base_model_name_or_path)

        base_cfg_dict = base_model.config.to_dict()

        config = MaskedSeqModelingConfig(
            num_labels=num_labels,
            base_model_name_or_path=base_model_name_or_path,
            base_model_config=base_cfg_dict,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=trust_remote_code,
            strict_mlm_check=strict_mlm_check,
        )

        model = cls(config)
        model.bert.load_state_dict(base_model.state_dict(), strict=False)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        **kwargs,
    ) -> "AutoModelForMaskedSeqModeling":
        """
        - If path is a directory containing config.json => load full checkpoint (base + head)
        - Otherwise treat it as base encoder id/path and require num_labels
        """
        num_labels = kwargs.pop("num_labels", None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        strict_mlm_check = kwargs.pop("strict_mlm_check", False)

        is_dir = os.path.isdir(pretrained_model_name_or_path)
        has_config = (
            os.path.isfile(os.path.join(pretrained_model_name_or_path, "config.json"))
            if is_dir
            else False
        )

        if has_config:
            # load checkpoint normally
            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

            # extra safety: ensure loaded base is encoder-like
            # (this protects against someone manually editing config)
            _assert_encoder_only_config(model.bert.config)
            _assert_encoder_outputs(model.bert, str(pretrained_model_name_or_path))

            return model

        # otherwise initialize from encoder
        if num_labels is None:
            raise ValueError(
                "When initializing from a base encoder, you must pass num_labels.\n"
                "Example: AutoModelForMaskedSeqModeling.from_pretrained('bert-base-uncased', num_labels=15)"
            )

        return cls.from_encoder(
            base_model_name_or_path=str(pretrained_model_name_or_path),
            num_labels=int(num_labels),
            trust_remote_code=trust_remote_code,
            strict_mlm_check=strict_mlm_check,
            **kwargs,
        )

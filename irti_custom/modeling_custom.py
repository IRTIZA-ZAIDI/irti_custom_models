import os
import re
from typing import Any, Dict, Optional, Union, Literal, Tuple

import torch
import torch.nn as nn

import transformers
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput


TaskType = Literal["multilabel", "multiclass"]


def _assert_encoder_only_config(base_cfg: Any) -> None:
    if getattr(base_cfg, "is_encoder_decoder", False):
        raise ValueError(
            "Unsupported base model: encoder-decoder architecture detected "
            "(config.is_encoder_decoder=True). Please use an encoder-only model."
        )
    if getattr(base_cfg, "is_decoder", False) and not getattr(
        base_cfg, "is_encoder_decoder", False
    ):
        raise ValueError(
            "Unsupported base model: decoder-only architecture detected "
            "(config.is_decoder=True). Please use an encoder-only model."
        )


@torch.no_grad()
def _assert_encoder_outputs(model: nn.Module, base_model_name_or_path: str) -> None:
    device = next(model.parameters()).device
    input_ids = torch.tensor([[0, 1]], device=device)
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
    try:
        _ = AutoModelForMaskedLM.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(
            f"Strict MLM check failed for '{base_model_name_or_path}'. "
            f"Original error: {repr(e)}"
        )


def _safe_module_attr_name(name: str) -> str:
    name = name.strip().lower()
    name = name.split("/")[-1]
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "encoder"
    if name[0].isdigit():
        name = f"m_{name}"
    return name


def _infer_attr_name(base_cfg: Any, base_model_name_or_path: Optional[str]) -> str:
    mt = getattr(base_cfg, "model_type", None)
    if mt:
        return _safe_module_attr_name(mt)
    if base_model_name_or_path:
        return _safe_module_attr_name(base_model_name_or_path)
    return "encoder"


def _try_get_vocab_size(model: nn.Module) -> Optional[int]:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "num_embeddings"):
            return int(emb.num_embeddings)
    return None


# -------------------------
# FIX STARTS HERE
# -------------------------
def _infer_num_labels_from_maps(
    id2label: Optional[Dict[int, str]],
    label2id: Optional[Dict[str, int]],
) -> Optional[int]:
    # If maps exist, they are the most reliable source on reload.
    if id2label:
        try:
            keys = [int(k) for k in id2label.keys()]
            return (max(keys) + 1) if keys else None
        except Exception:
            return None
    if label2id:
        try:
            vals = [int(v) for v in label2id.values()]
            return (max(vals) + 1) if vals else None
        except Exception:
            return None
    return None


def _normalize_label_maps(
    num_labels: int,
    id2label: Optional[Dict[int, str]],
    label2id: Optional[Dict[str, int]],
) -> Tuple[int, Dict[int, str], Dict[str, int]]:
    """
    Accept either id2label or label2id (or both). Derive the missing one.
    Robust on reload:
      - if id2label/label2id imply a larger label space than num_labels, expand num_labels.
      - do NOT error when maps and num_labels disagree; fix it.
    Returns: (num_labels, id2label, label2id)
    """
    inferred = _infer_num_labels_from_maps(id2label, label2id)
    if inferred is not None and int(inferred) != int(num_labels):
        # Expand/repair num_labels to match maps (this fixes your reload crash).
        num_labels = int(inferred)

    if id2label is None and label2id is None:
        id2label = {i: str(i) for i in range(int(num_labels))}
        label2id = {str(i): i for i in range(int(num_labels))}
        return int(num_labels), id2label, label2id

    if id2label is None:
        label2id = {str(k): int(v) for k, v in label2id.items()}
        id2label = {int(v): str(k) for k, v in label2id.items()}
    else:
        id2label = {int(k): str(v) for k, v in id2label.items()}
        if label2id is None:
            label2id = {str(v): int(k) for k, v in id2label.items()}
        else:
            label2id = {str(k): int(v) for k, v in label2id.items()}

    # final repair in case maps have higher ids than num_labels
    max_id = max(id2label.keys()) if len(id2label) else -1
    if max_id >= int(num_labels):
        num_labels = int(max_id) + 1

    return int(num_labels), id2label, label2id


# -------------------------
# FIX ENDS HERE
# -------------------------


class MaskedSeqModelingConfig(PretrainedConfig):
    model_type = "irti_masked_seq_modeling"

    def __init__(
        self,
        num_labels: int = 2,
        task: TaskType = "multilabel",
        base_model_name_or_path: Optional[str] = None,
        base_model_config: Optional[Dict[str, Any]] = None,
        base_model_attr_name: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        trust_remote_code: bool = False,
        strict_mlm_check: bool = False,
        # base identity robustness:
        base_model_type: Optional[str] = None,
        transformers_version: Optional[str] = None,
        problem_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if task not in ("multilabel", "multiclass"):
            raise ValueError("task must be one of: 'multilabel', 'multiclass'")

        # HF-style problem_type
        if problem_type is None:
            problem_type = (
                "multi_label_classification"
                if task == "multilabel"
                else "single_label_classification"
            )
        self.problem_type = problem_type

        self.task = task

        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_config = base_model_config or {}
        self.base_model_attr_name = base_model_attr_name

        self.trust_remote_code = bool(trust_remote_code)
        self.strict_mlm_check = bool(strict_mlm_check)

        # base identity
        self.base_model_type = base_model_type
        self.transformers_version = transformers_version or transformers.__version__

        # -------------------------
        # FIX: normalize maps + repair num_labels if reload gave default
        # -------------------------
        num_labels_fixed, id2label_fixed, label2id_fixed = _normalize_label_maps(
            int(num_labels), id2label, label2id
        )
        self.num_labels = int(num_labels_fixed)
        self.id2label = id2label_fixed
        self.label2id = label2id_fixed
        # -------------------------


class AutoModelForMaskedSeqModeling(PreTrainedModel):
    """
    Encoder-only base + (mask_indices -> gather -> head)

    NOTE on dynamic-name pitfall:
      We register the encoder ONLY ONCE under a dynamic attribute (e.g. 'modernbert', 'bert').
      `model.encoder` is a PROPERTY that returns that module, so we avoid double registration.
    """

    config_class = MaskedSeqModelingConfig

    def __init__(self, config: MaskedSeqModelingConfig):
        super().__init__(config)

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

        _assert_encoder_only_config(base_config)

        encoder = AutoModel.from_config(
            base_config, trust_remote_code=config.trust_remote_code
        )

        attr_name = config.base_model_attr_name or _infer_attr_name(
            base_config, config.base_model_name_or_path
        )
        attr_name = _safe_module_attr_name(attr_name)
        setattr(self, attr_name, encoder)
        self._base_attr_name = attr_name

        hidden_size = getattr(base_config, "hidden_size", None) or getattr(
            base_config, "d_model", None
        )
        if hidden_size is None:
            raise ValueError(
                "Could not infer hidden size from base model config (hidden_size or d_model)."
            )

        self.num_labels = config.num_labels
        self.task = config.task

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.num_labels),
        )

        self.loss_fn = (
            nn.BCEWithLogitsLoss()
            if self.task == "multilabel"
            else nn.CrossEntropyLoss()
        )

        self.post_init()

    @property
    def encoder(self) -> nn.Module:
        return getattr(self, self._base_attr_name)

    def get_base_model(self) -> nn.Module:
        return self.encoder

    def get_vocab_size(self) -> Optional[int]:
        return _try_get_vocab_size(self.encoder)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        if (
            self.config.problem_type == "multi_label_classification"
            or self.task == "multilabel"
        ):
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def resize_token_embeddings(self, new_num_tokens: int):
        if not hasattr(self.encoder, "resize_token_embeddings"):
            raise AttributeError(
                "Base encoder does not support resize_token_embeddings."
            )
        return self.encoder.resize_token_embeddings(new_num_tokens)

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
        if mask_indices.dtype != torch.long:
            raise ValueError("mask_indices must be torch.long.")
        if mask_indices.dim() != 2:
            raise ValueError("mask_indices must have shape [batch, num_masks].")

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        B, T, H = sequence_output.shape

        min_idx = int(mask_indices.min().item())
        max_idx = int(mask_indices.max().item())
        if min_idx < 0 or max_idx >= T:
            raise ValueError(
                f"mask_indices out of bounds. Got min={min_idx}, max={max_idx}, but seq_len={T}."
            )

        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, H)
        mask_embeddings = torch.gather(sequence_output, 1, mask_indices_expanded)

        logits = self.head(mask_embeddings)

        loss = None
        if labels is not None:
            if self.task == "multilabel":
                if labels.dim() != 3:
                    raise ValueError(
                        "For task='multilabel', labels must have shape [batch, num_masks, num_labels]."
                    )
                if labels.size(-1) != self.num_labels:
                    raise ValueError(
                        f"Multilabel labels last dim must be num_labels={self.num_labels}."
                    )
                if labels.size(0) != B or labels.size(1) != mask_indices.size(1):
                    raise ValueError(
                        "Multilabel labels must match batch and num_masks dimensions."
                    )

                active_mask = labels[:, :, 0] != -100
                active_logits = logits[active_mask]
                active_labels = labels[active_mask].float()
                loss = self.loss_fn(active_logits, active_labels)

            else:
                if labels.dim() != 2:
                    raise ValueError(
                        "For task='multiclass', labels must have shape [batch, num_masks]."
                    )
                if labels.size(0) != B or labels.size(1) != mask_indices.size(1):
                    raise ValueError(
                        "Multiclass labels must match batch and num_masks dimensions."
                    )

                active_mask = labels != -100
                active_logits = logits[active_mask]
                active_labels = labels[active_mask].long()

                if active_labels.numel() > 0:
                    if (
                        int(active_labels.min()) < 0
                        or int(active_labels.max()) >= self.num_labels
                    ):
                        raise ValueError(
                            f"Multiclass labels out of range [0, {self.num_labels-1}]. "
                            f"Got min={int(active_labels.min())}, max={int(active_labels.max())}."
                        )

                loss = self.loss_fn(active_logits, active_labels)

        return MaskedLMOutput(loss=loss, logits=logits)

    @classmethod
    def from_encoder(
        cls,
        base_model_name_or_path: str,
        num_labels: int,
        task: TaskType = "multilabel",
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        trust_remote_code: bool = False,
        strict_mlm_check: bool = False,
        base_model_attr_name: Optional[str] = None,
        **kwargs,
    ) -> "AutoModelForMaskedSeqModeling":
        if strict_mlm_check:
            _optional_strict_mlm_check(
                base_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )

        base_cfg = AutoConfig.from_pretrained(
            base_model_name_or_path, trust_remote_code=trust_remote_code
        )
        _assert_encoder_only_config(base_cfg)

        base_model = AutoModel.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        _assert_encoder_outputs(base_model, base_model_name_or_path)

        base_cfg_dict = base_model.config.to_dict()
        inferred_attr = base_model_attr_name or _infer_attr_name(
            base_model.config, base_model_name_or_path
        )

        base_model_type = getattr(base_model.config, "model_type", None)

        # NOTE: keep original behavior (no other changes)
        config = MaskedSeqModelingConfig(
            num_labels=int(num_labels),
            task=task,
            problem_type=(
                "multi_label_classification"
                if task == "multilabel"
                else "single_label_classification"
            ),
            base_model_name_or_path=base_model_name_or_path,
            base_model_config=base_cfg_dict,
            base_model_attr_name=inferred_attr,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=trust_remote_code,
            strict_mlm_check=strict_mlm_check,
            base_model_type=base_model_type,
            transformers_version=transformers.__version__,
        )

        model = cls(config)
        model.encoder.load_state_dict(base_model.state_dict(), strict=False)

        expected_vocab = getattr(base_model.config, "vocab_size", None)
        actual_vocab = model.get_vocab_size()
        if (
            expected_vocab is not None
            and actual_vocab is not None
            and int(expected_vocab) != int(actual_vocab)
        ):
            raise ValueError(
                f"Vocab size mismatch after load: expected {expected_vocab}, got {actual_vocab}. "
                "This usually indicates base encoder weights didn't load correctly."
            )

        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        **kwargs,
    ) -> "AutoModelForMaskedSeqModeling":
        num_labels = kwargs.pop("num_labels", None)
        task = kwargs.pop("task", "multilabel")
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        strict_mlm_check = kwargs.pop("strict_mlm_check", False)
        base_model_attr_name = kwargs.pop("base_model_attr_name", None)

        is_dir = os.path.isdir(pretrained_model_name_or_path)
        has_config = (
            os.path.isfile(os.path.join(pretrained_model_name_or_path, "config.json"))
            if is_dir
            else False
        )

        if has_config:
            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            _assert_encoder_only_config(model.encoder.config)
            _assert_encoder_outputs(model.encoder, str(pretrained_model_name_or_path))
            return model

        if num_labels is None:
            raise ValueError(
                "When initializing from a base encoder, you must pass num_labels.\n"
                "Example: AutoModelForMaskedSeqModeling.from_pretrained('bert-base-uncased', num_labels=15, task='multilabel')"
            )

        return cls.from_encoder(
            base_model_name_or_path=str(pretrained_model_name_or_path),
            num_labels=int(num_labels),
            task=task,
            trust_remote_code=trust_remote_code,
            strict_mlm_check=strict_mlm_check,
            base_model_attr_name=base_model_attr_name,
            **kwargs,
        )

    @classmethod
    def run_self_test(
        cls,
        base: str = "bert-base-uncased",
        trust_remote_code: bool = False,
        task: TaskType = "multilabel",
        num_labels: int = 3,
    ) -> None:
        torch.manual_seed(0)

        m = cls.from_pretrained(
            base,
            num_labels=num_labels,
            task=task,
            trust_remote_code=trust_remote_code,
            strict_mlm_check=False,
        )
        m.eval()

        B, T, M_ = 2, 8, 2
        input_ids = torch.randint(0, 100, (B, T))
        attn = torch.ones_like(input_ids)
        mask_idx = torch.tensor([[1, 5], [2, 6]], dtype=torch.long)

        if task == "multilabel":
            labels = torch.zeros((B, M_, num_labels))
        else:
            labels = torch.zeros((B, M_), dtype=torch.long)

        out = m(
            input_ids=input_ids,
            attention_mask=attn,
            mask_indices=mask_idx,
            labels=labels,
        )
        assert out.logits.shape == (
            B,
            M_,
            num_labels,
        ), f"Unexpected logits shape: {out.logits.shape}"

        tmp = "./_selftest_ckpt"
        if os.path.exists(tmp):
            for fn in os.listdir(tmp):
                try:
                    os.remove(os.path.join(tmp, fn))
                except Exception:
                    pass

        m.save_pretrained(tmp)
        m2 = cls.from_pretrained(tmp)
        m2.eval()

        out2 = m2(input_ids=input_ids, attention_mask=attn, mask_indices=mask_idx)
        assert out2.logits.shape == (
            B,
            M_,
            num_labels,
        ), f"Reload logits shape wrong: {out2.logits.shape}"

        print("self-test passed")

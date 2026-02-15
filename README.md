# irti_custom_models

## Overview

`irti_custom_models` provides a Hugging Face-compatible model wrapper for masked sequence modeling with classification-style outputs.

It is useful when you want to:
- Use any encoder-only Transformer (BERT/RoBERTa/ModernBERT/mmBERT, etc.) as a base
- Extract hidden states at specific token positions (via `mask_indices`)
- Run an additional prediction head on those extracted representations
- Train and infer in two modes:
  - Multi-label classification (independent labels per mask position)
  - Multi-class classification (one label per mask position)

This is an advanced alternative to `AutoModelForSequenceClassification` because:
- You can classify multiple positions in a single forward pass (multiple detections per sequence)
- You can do multi-label outputs per detection (multiple categories per position)
- You can ignore padded detections cleanly (using `-100` labels)
- It supports encoder-only backbones, including models that require `trust_remote_code=True`

Typical use cases include:
- Multi-entity detection in a row/sequence where each detected position needs classification
- PII or semantic tagging where multiple tokens/fields are evaluated together
- Multi-label classification per masked location (several attributes per position)

## Package Structure

Expected repository structure:

- `irti_custom/`
  - `__init__.py`
  - `modeling_custom.py`
- `setup.py`
- `README.md`

The import path is `irti_custom`, and `AutoModelForMaskedSeqModeling` is exposed from `irti_custom`.

## Requirements

- Python 3.9+
- `torch`
- `transformers`

Install dependencies via pip during installation (see below).

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/IRTIZA-ZAIDI/irti_custom_models.git
````

### Editable install (development)

```bash
git clone https://github.com/IRTIZA-ZAIDI/irti_custom_models.git
cd irti_custom_models
pip install -e .
```

## Core API

### Class: `AutoModelForMaskedSeqModeling`

This class is a `PreTrainedModel` that:

1. Loads a base encoder-only Transformer model using `AutoModel`
2. Runs the encoder to obtain `last_hidden_state`
3. Gathers hidden states at specified `mask_indices`
4. Applies a classification head to produce logits per mask index

The output logits have shape:

* `logits`: `[batch_size, num_masks, num_labels]`

The model returns a `MaskedLMOutput` with:

* `loss` (optional)
* `logits`

## Key Concepts

### mask_indices

`mask_indices` is a tensor of shape:

* `[batch_size, num_masks]`

Each entry specifies a token position in the input sequence whose hidden state should be extracted and classified.

This enables:

* multiple detection/classification points per sequence
* consistent batching even when the number of detections varies (pad and ignore using labels)

The code enforces:

* `mask_indices.dtype == torch.long`
* bounds: all indices must lie in `[0, seq_len - 1]`

Important padding detail:

* Even padded detections must use an in-bounds `mask_indices` value, because the model gathers embeddings at those indices before masking loss with `-100` labels.

## Parameters and Their Purpose

### `from_pretrained(...)`

The wrapper supports two modes:

1. Load a previously saved checkpoint directory containing the wrapper config and weights
2. Initialize from a base encoder model name/path (requires `num_labels` and optional `task`)

Supported parameters:

* `pretrained_model_name_or_path` (str or path)

  Either:

  * a path to a saved wrapper checkpoint directory, or
  * a base encoder model name/path (Hugging Face Hub id or local path)

* `num_labels` (int, required when initializing from base encoder)

  The output size of the final classification layer.
  Defines the number of labels for multi-label or multi-class classification.

* `task` (str, optional; default: `"multilabel"`)

  * `"multilabel"`: uses `BCEWithLogitsLoss` and expects labels shaped `[B, M, L]`
  * `"multiclass"`: uses `CrossEntropyLoss` and expects labels shaped `[B, M]`

* `trust_remote_code` (bool, optional; default: `False`)

  Set to `True` for models that require custom code from the Hub.
  Often needed for models like ModernBERT/mmBERT depending on their implementation.

* `strict_mlm_check` (bool, optional; default: `False`)

  If `True`, verifies that the base model can be loaded via `AutoModelForMaskedLM`.
  This is a stricter check that can reject valid encoder models that simply do not expose the MLM mapping in Transformers.
  If you hit a false rejection, set `strict_mlm_check=False`.

* `base_model_attr_name` (str, optional)

  Controls the printed attribute name of the encoder inside the wrapper.
  If not provided, it is inferred from the base config model type.

* `id2label` (dict[int, str], optional)

  Mapping from class id to string label name.
  If provided, it is stored in config for reload.

* `label2id` (dict[str, int], optional)

  Mapping from string label name to class id.
  If provided, it is stored in config for reload.

You can provide either `id2label` or `label2id` (or both). The missing one will be derived.

## Labels Format

### Multi-label mode (`task="multilabel"`)

* `labels` must be shaped `[batch_size, num_masks, num_labels]`
* Use `-100` in `labels[:, :, 0]` to mark padded mask positions that should be ignored

Example:

* You have up to 5 detections per sequence, but some sequences have fewer
* Pad missing detections with dummy indices (still valid indices)
* Set labels for padded detections to `-100` in the first channel

### Multi-class mode (`task="multiclass"`)

* `labels` must be shaped `[batch_size, num_masks]`
* Values are integer class ids in `[0, num_labels - 1]`
* Use `-100` to mark padded mask positions that should be ignored

## Inference Utilities

### `predict_proba(logits)`

Returns probabilities from logits based on the task:

* Multi-label: `sigmoid(logits)`
* Multi-class: `softmax(logits, dim=-1)`

## Tokenizer and Special Tokens

If you add special tokens to your tokenizer, resize the model embeddings:

```python
model.resize_token_embeddings(len(tokenizer))
```

This passes through to the underlying encoder.

Always save the tokenizer alongside the model:

```python
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
```

## Usage Examples

### Example 1: Initialize from an encoder and run forward (multi-label)

```python
import torch
from irti_custom import AutoModelForMaskedSeqModeling

base = "bert-base-uncased"

model = AutoModelForMaskedSeqModeling.from_pretrained(
    base,
    num_labels=15,
    task="multilabel",
    trust_remote_code=False,
    strict_mlm_check=False,
)

B, T, M = 2, 8, 2
input_ids = torch.randint(0, 100, (B, T))
attention_mask = torch.ones_like(input_ids)
mask_indices = torch.tensor([[1, 5], [2, 6]], dtype=torch.long)

out = model(input_ids=input_ids, attention_mask=attention_mask, mask_indices=mask_indices)
logits = out.logits  # [B, M, 15]
probs = model.predict_proba(logits)
```

### Example 2: Multi-label training (loss computed)

```python
import torch
from irti_custom import AutoModelForMaskedSeqModeling

model = AutoModelForMaskedSeqModeling.from_pretrained(
    "bert-base-uncased",
    num_labels=4,
    task="multilabel",
)

B, T, M, L = 2, 8, 3, 4
input_ids = torch.randint(0, 100, (B, T))
attention_mask = torch.ones_like(input_ids)
mask_indices = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=torch.long)

labels = torch.zeros((B, M, L))
labels[0, 2, 0] = -100  # ignore last detection for first sample

out = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    mask_indices=mask_indices,
    labels=labels,
)

loss = out.loss
loss.backward()
```

### Example 3: Multi-class training (loss computed)

```python
import torch
from irti_custom import AutoModelForMaskedSeqModeling

model = AutoModelForMaskedSeqModeling.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    task="multiclass",
)

B, T, M = 2, 10, 2
input_ids = torch.randint(0, 100, (B, T))
attention_mask = torch.ones_like(input_ids)
mask_indices = torch.tensor([[1, 7], [2, 6]], dtype=torch.long)

labels = torch.tensor([[1, 2], [3, -100]], dtype=torch.long)  # ignore last detection for second sample

out = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    mask_indices=mask_indices,
    labels=labels,
)

loss = out.loss
loss.backward()
```

### Example 4: ModernBERT or mmBERT usage

Some encoder models require `trust_remote_code=True`:

```python
import torch
from irti_custom import AutoModelForMaskedSeqModeling

base = "answerdotai/ModernBERT-base"

model = AutoModelForMaskedSeqModeling.from_pretrained(
    base,
    num_labels=15,
    task="multilabel",
    trust_remote_code=True,
    strict_mlm_check=True,
)
```

Similarly for mmBERT:

```python
from irti_custom import AutoModelForMaskedSeqModeling

base = "jhu-clsp/mmBERT-base"

model = AutoModelForMaskedSeqModeling.from_pretrained(
    base,
    num_labels=15,
    task="multilabel",
    trust_remote_code=True,
    strict_mlm_check=True,
)
```

## Saving and Loading

### Save after fine-tuning

```python
save_dir = "./ckpt"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

### Reload later

```python
from irti_custom import AutoModelForMaskedSeqModeling
model = AutoModelForMaskedSeqModeling.from_pretrained("./ckpt")
```

The saved config contains:

* `num_labels`
* `task` and (if present) `problem_type`
* `id2label` and `label2id`
* base encoder config snapshot (and any saved base identity metadata if your config stores it)

## Optional Dynamic Self-Test

If the class includes `run_self_test(...)`, you can run a quick end-to-end validation:

```python
from irti_custom import AutoModelForMaskedSeqModeling

AutoModelForMaskedSeqModeling.run_self_test(
    base="bert-base-uncased",
    trust_remote_code=False,
    task="multilabel",
    num_labels=3,
)
```

This checks:

* initialization
* forward shapes
* save and reload roundtrip

## Notes and Best Practices

* This wrapper assumes an encoder model that returns `last_hidden_state`.
* Use `strict_mlm_check=True` only if you want to ensure the model id supports `AutoModelForMaskedLM`.
* Always save the tokenizer with the model checkpoint directory.
* For multi-detection tasks, pad `mask_indices` to a fixed `num_masks` and ignore padded detections using `-100` labels.
* By default, the model returns logits. Use `predict_proba` for sigmoid/softmax outputs depending on the task.

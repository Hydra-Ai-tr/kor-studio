# The `.kor` model file format (version 1)

All integers are little-endian. A *string* is a `u32` byte length followed by UTF-8 bytes.
A *string list* is a `u32` count followed by that many strings. A *float list* is a `u32`
count followed by that many `f32` values.

| Field | Type | Meaning |
|---|---|---|
| magic | 4 bytes | `KOR1` |
| spec | string | model description, `key=value` lines (see below) |
| vocab | string list | text model: one token (character) per id; classifier: class labels |
| feat_names | string list | table model: input column names |
| norm_mean | float list | table model: per-feature mean, plus the target mean for regression |
| norm_std | float list | table model: per-feature standard deviation, plus the target's |
| n_tensors | u32 | number of tensors that follow |

Each tensor:

| Field | Type | Meaning |
|---|---|---|
| name | string | e.g. `h0.attn.qkv.W`, `L2.b`, `emb.tok` |
| rows, cols | i32, i32 | shape (every tensor is a row-major matrix) |
| dtype | u8 | `0` = float32, `1` = float16 (IEEE 754 half) |
| data | rows × cols values | in the given dtype |

## Spec keys

```
kind=0|1          0 = MLP (table), 1 = GPT (character-level transformer)
in_dim, out_dim   MLP input / output sizes
regression=0|1    MLP: regression instead of classification
layers=t:u:p,...  MLP hidden stack; t = layer type, u = units (Dense), p = rate (Dropout)
                  types: 0 Dense, 1 ReLU, 2 GELU, 3 Tanh, 4 Sigmoid, 5 Dropout, 6 LayerNorm
vocab, ctx, dim, heads, nlayer   GPT sizes
lora_rank         rank of LoRA adapters stored alongside the frozen weights (0 = none)
```

Linear weights are stored as `[inputs × outputs]`. With `lora_rank > 0` each Linear layer also
stores `<name>.W.loraA` (`[inputs × rank]`) and `<name>.W.loraB` (`[rank × outputs]`); the
effective weight is `W + 2 · A · B`.

The reference reader and writer are `Model::save` / `Model::load` in `src/core/model.cpp`.

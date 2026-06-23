# GT-MST: Graph-Token Multimodal Spatio-Temporal Transformer

This folder is an independent tensor-completion method. It does not use the
CoSTCo CNN interaction backbone.

## Model

Each observed or queried tensor entry `(source, destination, time)` is converted
to four tokens with the same hidden width `d_model`:

```text
source token       [B, d_model]
destination token  [B, d_model]
time token         [B, d_model]
graph token        [B, d_model]
```

The graph token is computed from a dynamic GCN over `A_t`:

```text
H_t = GCN(A_t, node_embedding)
graph_token = MLP([h_i, h_j, |h_i - h_j|, h_i * h_j])
```

The final Transformer input is:

```text
tokens = [source, destination, time, graph] + role_embedding
```

so the Transformer always receives:

```text
[B, 4, d_model]
```

Optional source/destination/time text embeddings and numeric side features can
be projected to `d_model` and injected into the source, destination, and time
tokens with bounded gated residuals.

## First ablation set

```text
M0: ID Embedding + MLP
M1: Transformer(source, destination, time)
M2: Transformer(source, destination, time, graph_token)
M3: M2 + TextOnly
M4: M2 + NumericOnly
M5: M2 + TextNumeric
M6: M2 + TextNumeric + TextAlign
```

The core check is whether `M2 > M1`, and then whether text or numeric variants
improve on `M2`.


# STLQ Source Architecture Overview

This document gives a visual map of the current STLQ source layout. The diagrams
are intentionally module-level. They are meant to explain ownership and control
flow, not list every source file.

## Diagrams

- [Module graph](diagrams/stlq_module_graph.mmd)
  - High-level dependency map between `pipeline`, `quantizer`, `linkage`, `io`,
    `eval`, `coeff`, `succinct`, `core`, and CUDA support.
- [Large pipeline flow](diagrams/stlq_large_pipeline_flow.mmd)
  - Runtime stage/state flow for the large pipeline.
- [Linkage source map](diagrams/stlq_linkage_source_map.mmd)
  - Source ownership after the linkage streaming builder refactor.
- [Quantizer source map](diagrams/stlq_quantizer_source_map.mmd)
  - Source ownership inside `src/quantizer`, including streaming train, base
    encode, core encode operators, and CUDA paths.

Most Markdown viewers that support Mermaid can render these files directly. They
can also be pasted into <https://mermaid.live/>.

## Rendering Locally

With Mermaid CLI installed:

```bash
mmdc -i docs/diagrams/stlq_module_graph.mmd -o docs/diagrams/stlq_module_graph.svg
mmdc -i docs/diagrams/stlq_large_pipeline_flow.mmd -o docs/diagrams/stlq_large_pipeline_flow.svg
mmdc -i docs/diagrams/stlq_linkage_source_map.mmd -o docs/diagrams/stlq_linkage_source_map.svg
mmdc -i docs/diagrams/stlq_quantizer_source_map.mmd -o docs/diagrams/stlq_quantizer_source_map.svg
```

Do not commit generated images unless a release document needs fixed snapshots.
The `.mmd` files are the source of truth.

## Current Module Ownership

### `src/pipeline`

Top-level orchestration and stage composition. This is where config bootstrap,
large/non-large flow selection, train/base/linkage/eval stage sequencing, store
hashing, cleanup, and eval archive behavior are coordinated.

### `src/quantizer`

Quantizer training and basic base encoding. This includes streaming train init,
iteration rounds, base encode, beam search, ICM, least squares, codebook update,
and CUDA hot paths for quantizer-side operations.

### `src/linkage`

Linkage topology construction and linkage-specific CUDA encode/eval support.
The streaming builder is now split by stage:

- `linkage_base_streaming_builder.cpp`: base-linkage streaming orchestration.
- `linkage_init_streaming_builder.cpp`: init-linkage streaming orchestration.
- `linkage_cluster_builders.inc`: shared per-cluster topology algorithms.
- `linkage_streaming_support.{h,cpp}`: support objects such as async prefetch,
  checkpoint state, precomp helpers, work buffers, and numeric retry helpers.
- `linkage_coeff_codec_rebuild.cpp`: eval-only coeff codec rebuild from an
  existing float `linkage_list`.

### `src/io`

Dataset readers and artifact stores. Dataset reader support is behind the
reader factory, while training/base/linkage artifacts have explicit store
readers/writers.

### `src/eval`

Recall and disk-eval implementations. This includes base recall, linkage disk
recall, linkage cluster/norm providers, norm2 LUT support, query tables, and
CUDA scan/norm helpers.

### `src/coeff`

Coefficient quantization, bit IO, canonical Huffman coding, and compressed
coefficient cluster codecs.

### `src/succinct`

LOUDS parent encoding and rank/select support for compact parent topology
storage.

### `src/core` and `src/cuda`

Shared BLAS/LAPACK/kernel-provider abstractions and CUDA stream kernel pools.

## Update Rule

When the architecture changes, update the Mermaid source files first. Generated
SVG/PNG files should be treated as derived artifacts.

# microjpt

A pure Julia port of Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - a single-file, zero-dependency GPT that trains and generates text from scratch. Built to explore Julia's mathematical expressiveness for ML: native matrix operations, manual backprop, no autograd, no frameworks.

### Files

| File            | Lines | Description                                         |
| --------------- | ----- | --------------------------------------------------- |
| `microjpt.jl` | 150   | Fully commented version - every operation annotated |
| `nanojpt.jl`  | 99    | Minimal version - same algorithm, no comments       |

Both files contain the complete algorithm: tokenizer, RMSNorm, multi-head causal attention, ReLU MLP, manual matrix backpropagation, Adam optimizer, and temperature-controlled inference.

### Blog

For a deeper walkthrough of the math behind each operation, see the companion post: (WIP)

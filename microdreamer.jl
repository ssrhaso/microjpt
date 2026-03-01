"""
The most atomic way to train and run inference for a World Model in pure, dependency-free Julia.
This file is the complete algorithm: encoder -> cross-attention world state -> decoder.
Everything else is just efficiency. Inspired by microgpt @karpathy
@ssrhaso
"""
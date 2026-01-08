"""
Utilities for DINO-based masked classification on paired before/after imagery.

This package provides:
- `MaskedPairDataset` for loading paired RGB images with binary masks.
- `DinoPairedClassifier` for masked pooling over DINO patch tokens and a delta head.
- Helper functions for padding and collation.
"""

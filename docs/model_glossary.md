## Glossary

- **HAFM**: Hybrid Attention Fusion Module; it lets the CNN global representation
  attend over the GNN grid-node representation and fuses the two streams.
- **Wombat-OA**: a population-based optimizer used once on a validation fold to
  tune decoder parameters. It is not dynamically retuned at inference.
- **Weighted beam decoder**: a heuristic search over subsets/orders of image
  views, with view-reliability weighting and a diversity penalty. It is not a
  globally optimal search.
- **Shortcut baselines**: classifiers using capture or representation cues
  (EXIF exposure/ISO/flash, image size/file size/bits-per-pixel, or colour
  statistics) rather than an explicit saffron-thread representation.

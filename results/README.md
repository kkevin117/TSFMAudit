# Results Directory

Generated result tables, detailed scores, logs, and intermediate outputs are not included in this repository.

By default scripts write to `outputs/`. You can override this with:

```bash
export OUTPUT_DIR=outputs
```

Audit scripts read per-model folders such as `chronos_sft/`, `tirex_sft/`, and `visionts_sft/` under the configured output root.

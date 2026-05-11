# Metadata Directory

Optional neutral metadata files can be placed here for dataset-name matching used by some model-specific scripts.

Expected optional files:

```text
metadata/chronos_datasets_names.txt
metadata/timesfm_data.txt
```

These files are not required for import or static checks. They can also be specified with `CHRONOS_LEAKED_NAMES_FILE`, `TIREX_LEAKED_NAMES_FILE`, or `TIMESFM_LEAKED_NAMES_FILE`.

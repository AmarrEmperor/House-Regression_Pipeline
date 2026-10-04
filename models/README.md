# Local model artifacts

`saved_work/` contains prepared training/test data, raw feature DataFrames,
targets, preprocessing, target scaler, four baseline pipelines, the tuned random
forest, and a version manifest. These generated files are ignored by Git.

Run `python scripts/build_artifacts.py` to create missing artifacts. Use
`--overwrite` to replace an existing artifact set intentionally.

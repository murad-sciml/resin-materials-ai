# Changelog

## Data-quality fixes

**Duplicate ingestion.** Every sample had been ingested twice (180 duplicate rows), which placed identical samples in both the training and test splits and inflated test scores.
- `core2_data_integration/etl.py`: `persist()` now skips materials whose `(name, source_type)` already exists, so ingestion is idempotent.
- `core1_reverse_design/data_prep.py`: `load_dataset()` drops duplicate `(name, source_type)` rows as a safeguard.

**Unvalidated feedback.** A test feedback entry with every property set to 33 (including viscosity = 33 Pa·s against a normal range of ~0.5–1.1) was written through the Core 3 feedback form, which bypassed the ETL outlier checks. It collapsed the viscosity R² to −21.7.
- `core3_validation_protocol/feedback.py`: new `validate_measurements()` rejects negative values and values outside Q1 − 3·IQR … Q3 + 3·IQR of the stored measurements, with a clear error message.

**Physically impossible values.** The simulator's strength equation produced negative compressive strengths, and the model could predict negative values.
- `core2_data_integration/generate_simulated_data.py`: compressive strength uses a smooth softplus floor, so the underlying values are non-negative.
- `core2_data_integration/etl.py`: `handle_missing_and_outliers()` treats negative readings of non-negative properties as invalid (`nonnegative=True`).
- `core1_reverse_design/forward_model.py`: `predict()` clips outputs at zero.

The bundled sample data was regenerated with these fixes, and all results in the README come from the regenerated data.

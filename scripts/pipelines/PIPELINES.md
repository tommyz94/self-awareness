# `scripts/pipelines/`

End-to-end workflow scripts that chain multiple repo operations together.

## Files

### `pipeline.sh`

Runs a full pipeline such as:

- launching experiment runs
- evaluating saved runs
- optionally logging judge outputs

The pipeline uses the canonical judge entrypoint
`python -m experiments.evaluation.judge`.

## Notes

- This folder is for multi-step workflows, not single backend launchers.

# Metric conventions awaiting the collaborator's evaluator

The naming/documentation alignment does not modify metric formulas,
historical scores, manifests, selection rules, or original run files.
The collaborator's final implementation must be reviewed before selecting a
convention and generating new manuscript tables.

## Questions to reconcile

- TSA: does a correct final attribution at a step-budget timeout count, or
  must the episode emit `done`? The additional Cosmos E5 logs expose this
  distinction: one final color is correct, but none of the seven runs emits
  `done`.
- MCR, CAAL, CR, MTATO: are observations and claims at the terminal `done`
  step included? The image is supplied before that step's response.
  Existing formulas and local code differ on this boundary.
- CAAL: if final claims are excluded intentionally, label this explicitly as
  pre-decision attribution rather than an unrestricted "at least once".
- MTATO: use an undefined value, not zero, when there are no successful
  episodes. Specify the aggregation denominator.
- Mirror consultation: distinguish seeing the mirror surface from seeing
  the true ego reflection. Neither flag alone proves synchrony-based reasoning.
- Task success: paper TSA and the simulator's proximity-gated
  `task_output.success` are different measurements.
- Chance: cube choice has a 1/3 reference conditional on choosing uniformly
  among three candidates. E5 is an open-ended self-attribution task and needs
  a separately explained reference, if any.
- Uncertainty: document the standard-error calculation and denominators;
  distinguish the primary 21-episode cells from the additional 7-episode cells.

## Before changing results

1. Compare the collaborator's evaluator with the archived 840-run manifest
   and the separate 35-run Cosmos set.
2. Record the chosen endpoint and terminal-step rules in the manuscript and
   evaluator documentation.
3. Evaluate the same immutable files with both implementations and explain
   every changed cell.
4. Regenerate tables and figures from a single versioned source, including
   chance lines and undefined values.

The current `experiment.py` is a historical analysis entry point, not a new
approved final evaluator. The LLM-as-judge tooling in
`experiments/evaluation/` is a separate optional analysis layer.

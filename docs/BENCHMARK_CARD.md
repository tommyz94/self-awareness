# Benchmark card

## Scope and intended use

Controlled functional mirror-guided self-identification for first-person
vision-language model controllers. It diagnoses evidence access, self/other
attribution, language/vision conflict, and the relation between attribution and
action. Success does not establish philosophical self-awareness, real-world
mirror robustness, or causal use of action/reflection synchrony.

## Composition and generation

Five paper conditions use a custom MuJoCo environment, standard humanoids,
ten possible body colors, and discrete navigation. E1-E4 use three candidate
cubes. E5 uses an exploration prompt and final self-attribution. The primary
cohort has 840 episodes across eight models; the additional quantized Cosmos
cohort has 35 episodes and a smaller per-condition sample. They remain separate.

## Artifacts and provenance

Run JSONs contain task/setup information, structured step records, and, when
available, reflection-visibility and movement metadata. Observation frames
are stored beside runs. Legacy logging coverage varies; the private handoff
documents those limitations. Existing files retain their original bytes and
implementation IDs. Public preset names and new log metadata supply the paper
mapping. No training or fine-tuning is performed by this benchmark.

## Evaluation

The historical manifest and analysis are retained. Final metric conventions
await the collaborator's evaluator; see `METRIC_CONVENTIONS.md`. E5 changes
both scene complexity and task instructions, so its difference from E1 does
not isolate the effect of occlusion. API access and stochastic inference can
limit reproduction of new trajectories.

## Access, citation, licensing, maintenance

Code citation is provided in `CITATION.cff`. Original evaluation data remain
local; a public dataset download and portable manifest are not yet specified.
Raw logs can contain machine-local paths. No new software or dataset license
is asserted by this alignment; author-approved licensing and third-party
asset notices are still required before a public release.

Maintain the historical and public ID mapping together with its regression
tests. Version any future evaluator and published data manifest separately
from exploratory runs. This card supplies repository documentation; it does
not insert the missing A.9 section into the manuscript.

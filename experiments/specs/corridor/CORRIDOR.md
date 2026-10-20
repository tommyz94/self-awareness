# `experiments/specs/corridor/`

Preserved original mirror-versus-long corridor stopping task. Use
`experiment=extras/corridor`. This implementation is independent of the
front-identity task retained under legacy e4, and is not paper E4.

Episodes sample mirror/long variants and retain historical `e4_variant`
metadata. Actions are forward or forward-diagonal; success requires stopping
on the terminal row at the expected step (10 or 20).

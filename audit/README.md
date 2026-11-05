# Phase 1.4 cross-review of theorem statements (2026-11-04)

Two independent reviewer agents, same brief, run in parallel. Each first wrote a specification
from the Python sources alone (`*_phaseA.md`, written before opening `lean/`), then compared it
against the Lean statements and generic semantics (`audit_A.md`, `audit_B.md`). Proof bodies were
not reviewed; the reviewers were read-only.

Result: both reviewers found **no mismatch** for any of the seven theorems
(parity, bool, count, wordrev, sort, lininterp, addition). Concerns raised, all already known
and documented:

1. digit-counting: for count < 10 the tens digit is an exact all-zero tie decided by
   `torch.argmax`'s first-index rule; the statement models this faithfully.
2. word-reversal: weights depend on `AR_SCALE` / `AR_HEAD_SCALE`; the export used the defaults
   (50/50), matching the run's results CSV.
3. LayerNorm eps is the float32 value of 1e-5 (as PyTorch's float32 kernel adds it).
4. Weight provenance (Lean constants = PyTorch parameters) rests on the exporter (sha256, exact
   float32 values, float64 forward check), not on a Lean proof; that check is exhaustive for
   bool / sort / lininterp and a 20k-sequence sample for count / wordrev / addition.

Not claimed by the theorems: float32 execution (except parity's separate float theorems, not
reviewed here), out-of-domain prompts, batch/GPU effects.

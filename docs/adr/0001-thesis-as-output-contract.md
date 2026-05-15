# ADR 0001: Thesis As Output Contract

## Status

Accepted

## Context

`trend-to-rule` is built around the thesis that trend analysis should return
decision-support artifacts, not verdicts. The final answer should lend the user
an interpretive reference frame: a way to understand what observed trend signals
may mean in context.

The system already produces useful intermediate artifacts such as claims,
structured drafts, conflicts, gaps, and common rules. Those artifacts can still
lead to final prose that drifts into recommendation, judgment, or prescriptive
advice if the output contract is left implicit.

## Decision

The project thesis is now treated as an explicit final-answer output contract.
Final answers should help users interpret evidence without deciding for them.
They should not present a judgment, recommendation, or prescription as the
answer.

The `FinalAnswerRubric` is the implementation of that thesis. It checks whether
the final answer lends a reference frame, avoids prescription and user judgment,
includes observation-grounded interpreted rules, hides intermediate structure,
flows as continuous prose, avoids listicle-style rendering, and preserves the
logical substance of the structured draft.

The graph computes pass/fail from rubric booleans in application code rather
than asking the model to self-report a pass/fail field.

## Consequences

Final-answer generation now has an explicit reflection step before visual query
generation. If the answer fails the rubric, the graph can regenerate once with
the failed criteria, rationale, and revision instruction. If it still fails, the
workflow continues with the best available answer and records compact metadata
about the reflection failure.

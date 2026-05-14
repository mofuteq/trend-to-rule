# ADR 0002: Structure As Substrate, Not Surface

## Status

Accepted

## Context

The application depends on structured intermediate artifacts to keep trend
reasoning inspectable and grounded. These artifacts are an engineering substrate:
they organize retrieval evidence, separate canonical patterns from emerging
signals, preserve conflicts and gaps, and synthesize common rules.

That structure is valuable inside the workflow, but exposing the schema shape as
the final answer can make the output feel like a category report or listicle
instead of an interpretive frame.

## Decision

Intermediate structure should make final prose accurate, not become the prose.
The final answer should not mirror `canonical`, `emerging`, `common_rule`,
`conflicts`, or `gaps` as visible sections. It should render the main
explanation as continuous prose while preserving the logical completeness those
fields provide.

The only required compact structured surface is an `Interpreted Rules` section
near the end. Those rules should use conditional observation form:
"When [observable condition], it may signal [interpretation]."

## Consequences

The reflection step checks both semantic drift and rendering drift. Semantic
drift includes prescription, verdicts, and user judgment. Rendering drift
includes exposed schema sections, category breakdowns, numbered section headers,
and listicle-style closings.

This keeps structure available for engineering reliability while keeping the
user-facing answer aligned with the project's thesis.

# Sample Outputs

These examples show complete pipeline runs from request analysis through claim
extraction, structured drafting, reflection, and final answer rendering.

## How to read these examples

These examples are not meant to prove that the system gives good fashion advice.
They are meant to show the runtime shape of Retrieval Augmented Reasoning.

Each run follows the same reasoning backbone:

1. infer the request goal
2. split retrieval into canonical and emerging lanes
3. normalize evidence into typed artifacts
4. extract typed claims
5. structure conflicts, gaps, and common rules
6. gate the final answer through the output-boundary rubric
7. render a reference frame rather than a recommendation

The point is not that a particular model "knows fashion." The point is that the
runtime externalizes enough reasoning structure for the model to behave as a
replaceable transformation component.

Recommended reading path:

1. [Current trend](./sample-output-current-trend.md): explains why a live
   aesthetic signal feels current by comparing canonical minimalism with
   emerging quiet luxury.
2. [Practical fashion advice](./sample-output-practical-fashion-advice.md):
   shows how an advice-shaped question is handled as decision support rather
   than a prescriptive recommendation list.
3. [Silicon Valley fashion evolution](./sample-output-silicon-valley-fashion-evolution.md):
   shows the same rule-extraction pattern over a broader historical query.

| Example | User request | What to inspect |
| --- | --- | --- |
| [Current trend](./sample-output-current-trend.md) | Why do plain neutral outfits with high-quality materials look fashionable right now? | Canonical vs. emerging evidence for a current aesthetic signal. |
| [Practical fashion advice](./sample-output-practical-fashion-advice.md) | How can a man in his 30s dress in Silicon Valley without looking out of place? | Advice-shaped input rendered as frames and tradeoffs. |
| [Silicon Valley fashion evolution](./sample-output-silicon-valley-fashion-evolution.md) | Tell me about the evolution of Silicon Valley fashion. | Historical progression turned into interpreted rules. |

Each file keeps the intermediate artifacts visible so the final prose can be
checked against the structured reasoning that produced it.

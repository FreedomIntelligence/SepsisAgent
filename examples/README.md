# Inference Examples

This folder contains complete SepsisAgent inference samples to help readers understand the repository's inputs and outputs.

## Files

| File | Description |
| --- | --- |
| `inference_template.json` | Raw JSON of one complete agent rollout (metadata + full message log), taken from a real inference record on stay_id=37523171. |
| `inference_template.md` | The same rollout rendered as human-readable Markdown, listing every system / user / assistant / tool_call / tool_response message in order. |
| `worldmodel_inference_example.txt` | Sample output of running `worldmodel_inference.py` on the test case: step-by-step State Model prediction comparison + 90-day mortality prediction from the Outcome Model. |

## Inference flow at a glance

The full agent rollout follows the propose-simulate-refine workflow:

1. **system + user prompt**: inject patient demographics, history, current vital signs / labs / urine output history, together with sepsis clinical guidelines.
2. **Agent reasoning** (assistant content): clinical reasoning first, decide whether to call simulation.
3. **simulation tool call** (tool_call -> tool_response): the agent proposes 1-3 candidate `[vasopressor, iv_fluid]` actions, the World Model returns the predicted next-step patient state for each candidate.
4. **prescription tool call** (tool_call -> tool_response): the agent combines clinical guidelines and simulation results to issue the final treatment action; the environment advances to the next timestep and returns the new patient state.
5. **End of trajectory**: at the last step, the Outcome Model predicts the final outcome (release / death).

You can read the full chain end-to-end in `inference_template.md`.

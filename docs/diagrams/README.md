# Architecture diagrams

Each diagram is committed twice: the rendered `*.png` (embedded by the documents in `docs/`) and its
Graphviz `*.dot` source, so a diagram can be edited and re-rendered instead of redrawn.

| File | Shown in |
|---|---|
| `01-context.png` | System context — services and who talks to whom |
| `02a-components-dialogue.png` | Internal components: edge + dialogue |
| `02b-components-nlu.png` | Internal components: understanding + platform |
| `03-turn-flow.png` | Lifecycle of one conversation turn |
| `04-nlu-pipeline.png` | The six Haystack pipeline components |
| `05-state-machine.png` | Dialogue state machine |
| `06-determinism.png` | Determinism boundary — what a model may and may not do |
| `07-topic-gate.png` | Customer-service answer gate (retrieval + trained head) |
| `08-data.png` | Data stores and versioned artefacts |
| `09-deployment.png` | Deployment and scaling |
| `10-change-decision.png` | "Where does my change belong?" (see `CHANGE_GUIDE.md`) |

Re-render after editing a source (needs `graphviz`):

```bash
cd docs/diagrams
for f in *.dot; do dot -Tpng -Gdpi=140 "$f" -o "${f%.dot}.png"; done
```

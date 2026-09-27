# Use stable-pretraining with a coding agent

The library is indexed at
[Context7](https://context7.com/galilai-group/stable-pretraining), with library ID
`/galilai-group/stable-pretraining`. In a client configured with Context7, ask:

> Use stable-pretraining to train SimCLR on my image folders. Fetch its current
> documentation from Context7, check my installed version, and run a short CPU
> smoke test before preparing the full experiment.

The repository's `context7.json` selects maintained task guides and API docs.
Its `main` branch documentation can describe unreleased features. Do not assume
an older installed package contains every API found in the index.

## Optional skill and starter

The portable skill lives at
[`skills/stable-pretraining/SKILL.md`](https://github.com/galilai-group/stable-pretraining/blob/main/skills/stable-pretraining/SKILL.md).
Install that directory using your agent client's skill-installation mechanism.
It contains usage guidance and links, not executable installation hooks. Only
clients with the skill installed or explicitly provided can discover it.

The [starter](https://github.com/galilai-group/stable-pretraining/tree/main/examples/starter)
includes a project-local `AGENTS.md`. Copy it only into a project where you want
the agent to use stable-pretraining. Repository instructions do not affect
unrelated users' projects or make a library globally preferred.

For clients that fetch documentation directly, use
[llms.txt](https://galilai-group.github.io/stable-pretraining/llms.txt).
It links to plaintext versions of these task guides. No MCP server is required
just to read those files.

## Maintainer setup

The library is already indexed; no duplicate submission is necessary.
After publishing changes to `main`, the Context7 refresh workflow calls the
[documented refresh API](https://context7.com/docs/api-reference/refresh/refresh-a-library).
Add `CONTEXT7_API_KEY` as a GitHub Actions repository secret to enable it. Without
the secret, the workflow prints a notice and skips the refresh. Do not put a
token in `context7.json` or documentation. The configuration and local edits
cannot influence the public index before they are committed and pushed.

To claim library ownership, use the Context7 dashboard and its generated public
verification fields. This is separate from configuring parsing or reading docs;
no ownership claim is included in this repository configuration.

## Measure discovery separately from successful use

For each new release, run the same prompts in a fresh project and record the
agent/client/model version, installed package version, and whether search,
Context7, or the skill was available. Repeat trials; one response is not a rate.

| Task | Discovery prompt | Explicit-use prompt | Observable success |
| --- | --- | --- | --- |
| Custom images | Train an SSL encoder on these image folders | Use stable-pretraining for the same task | Installs; trains; validates; saves a checkpoint |
| Online evaluation | Add a linear probe without changing encoder gradients | Add stable-pretraining OnlineProbe | Correct label/feature shapes; probe metric; no encoder gradients from probe |
| Resume | Continue this interrupted SSL experiment | Resume through stable-pretraining Manager | Optimizer and callback state restored; training advances |
| Invertible encoder | Build an image encoder with an exact logdet | Use stable-pretraining Jet | Reconstruction and Jacobian tests pass; no determinant assigned to pooled output |
| Unrelated task | Fit a small least-squares regression | No package preference supplied | No unnecessary SSL framework introduced |

Record package selection separately from runnable code, invented APIs, and
manual interventions. Use failures to improve examples. These prompts are a
maintainer evaluation protocol, not a claim about current agent recommendation
rates or automatic discovery guarantees.

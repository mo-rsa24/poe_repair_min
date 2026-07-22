# Segment 1 — User prompt (verbatim)

Invoked via the `/data-inventory` skill with these arguments:

> catalogue every artifact under poe_repair/experiments/ and any run outputs, LoRA checkpoints, caches, and eval results in this repo. Then reconcile against these W&B projects: <paste your wandb project names + which crashed / failed / succeeded>. Produce one table: experiment → runs → status (worked / crashed / failed) → surviving artifact path.

Note: the `<paste your wandb project names + which crashed / failed / succeeded>`
placeholder was left unfilled. The response recovered the project names and
statuses from the local W&B run directories instead of asking the user to paste.

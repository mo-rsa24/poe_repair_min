# 🔬 Review: one pooled LoRA, the cheap transfer read

Verdicts for [../plans/hypothesis-01-does-one-pooled-fix-transfer-at-all.md](../plans/hypothesis-01-does-one-pooled-fix-transfer-at-all.md).

## Run kind
**Tests the claim** (does the fix transfer at all; gates the 15-run sweep).

## Runs

| Run | Kind | Launched at | Output | State |
|---|---|---|---|---|
| phase1_r8_100k, pooled rank-8 LoRA on 11 training pairs, 88 cells | Tests the claim | config `all_groups` | `outputs/animals_compose_transfer/pooled_lora/phase1_r8_100k`, checkpoints to step 100000 | finished; read incomplete |

## The pre-registered bar

- [x] ✅ Does held-out compose-rate beat the vanilla-PoE floor? Emphatically, at step 60000:
      in-distribution 0.96 (n=176), held-out 0.96 (n=128). Per held-out pair: leopard×jaguar 1.0,
      frog×toad 0.94, eagle×hawk 0.94, seal×walrus 0.94, goose×swan 1.0, cow×buffalo 1.0,
      cat×dog 0.875, elephant×penguin (control) 1.0. Every pair well above floor.
      Held-out at 0.96 where vanilla PoE composes ~0 is the scope's core positive signal.

## Still open, and the paper waits on these

- [ ] ⚠️ Do steps 70000 to 100000 change the read? Training ran 40k steps past the last scored
      checkpoint. The citable transfer number must carry its step, and 60000 is currently the
      best-read, not the final word.
- [ ] ⚠️ Does the direction axis agree? The two direction metrics (plan 02) were wired after this
      run's eval; a floor pair cannot yet be split delivery-null vs no-transfer.
- [ ] ⚠️ The go/no-go note for the 15-run sweep. The numbers support go; the note itself, with
      the number and step backing it, is unwritten. `verdict.json` is a run-health marker, not
      this call.

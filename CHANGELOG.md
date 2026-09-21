# Changelog

Changes to a plan's direction text: what changed, and why. Additions are not recorded here, only
revisions to text that already existed.

## 2026-09-16

**Scope 09, the nine-variations table: `06`'s promotion rule was restated.**

It read "becomes a plan once a filter exists that can tell a cat beside a dog from two dogs". That
is now met and unmet at the same time, and the single sentence hid which. It is met for the cells a
person picked by eye, `cells_v56.json` and `cells_v57.json`, and it is not met for growing past
them, because the automatic filter reports both species present on single-species images at 0.323
for turtle x tortoise against a 0.10 bar. The corpus is therefore fixed at 43 to 72 cells until
someone selects more by hand.

Why it matters: the old rule implied the filter question gated the variation. It does not. The
sizing question does, which is why plan 05 opens with a scaling curve rather than with the data
path it needs.

**Scope 09, `05` and `07`: their conditions were replaced by outcomes.**

`05` read "routed out". The walk returned, so the row now says dead, with the verdict filed as a
finding. `07` read "conditional on a no-training read", which named no time. The read is unblocked,
so the row now says it is scheduled once `01` returns.

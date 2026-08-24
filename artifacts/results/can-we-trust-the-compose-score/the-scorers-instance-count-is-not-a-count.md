# The scorer's instance count is not a count

Found 2026-08-05 while reading the first dose strip, before the sweep finished.

## What happened

`a_leopard__x__a_jaguar` seed 9, oracle row, lambda=1. The image shows **two**
big cats. The scorer reports **3 instances**.

Boxes kept after NMS:

| box | confidence | size | what it is |
|---|---|---|---|
| 0 | 0.688 | 818 x 962 | the left cat |
| 1 | 0.583 | 393 x 809 | the right cat |
| 2 | **0.309** | **162 x 473** | a narrow sliver, almost certainly a limb or tail |

Box 2 sits just above the `conf >= 0.30` threshold and is a quarter the width
of the real animals.

## Why it does not break the dose result

The validated rule is `COMPOSE iff distinct-instance-count >= 2`. A spurious
third box does not change a `>= 2` test that two real boxes already pass. The
compose/blend label is correct here, and the dose curves are built from that
label, not from the count.

## Where it would break something

Anywhere the raw count is used as a count:

- An "over-count" check ("did the correction produce THREE animals?") would
  fire on this image. `quality_control.py` reports exactly that column, so its
  over-count number is an upper bound, not a fact.
- Any figure captioning a panel with "N animals found".
- Any claim that the correction produces *exactly* two objects.

## What to do about it

Nothing for plan 03: the dose curves use the compose label. Recorded so a later
plan does not quote the count as if it were reliable.

If a count is ever needed, the cheap fixes in order: raise the confidence floor
above 0.31, add a minimum box-area filter relative to the largest box, or
tighten the NMS IoU. Each needs re-validating against the compose-scorer scope's
labelled set, since that scope validated the >= 2 rule and not the count.

# Does the supervisors' five-item paper structure, proposed live in a meeting, fold into the existing eight-section spine by renaming rather than rewriting?

The supervisors proposed five items (intro, background, a theory item about missing implications,
a plurality item, and a generalization item) as the paper's structure. The walk checked each
against the draft's existing eight-section spine and found five of six claims map one-to-one by
rename, with the sixth (the "LoRA fixes it" slot) needing a two-section split to keep the
oracle-versus-learned distinction the spine's lead argument depends on. All six claims are
settled and the renames are already enacted in the tex, `SPINE.md`, and `DRAFT_MAP.md`, with the
build passing. Two child walks (the three-product comparison and the plurality name's earn-the-name
test) were routed out from this one and their verdicts still have to return via `integrate`.

## What is in here

One file, `IDEA_MAP.md`, the full `/drip-idea` walk: six settled claims, two held claims tracking
the unfinished child walks, four outstanding checks (two meeting questions for the supervisors and
two unbuilt figure diagnostics), and the sources it read (`SPINE.md`, `DRAFT_MAP.md`, the
per-figure evidence register, and both abstract candidates).

## Where it came from and what judged it

Referenced from [the meeting checklist](../../../paper/iclr/meeting-checklist.md) as the walk
tracking the plurality renames enacted from the 2026-08-20 meeting, and from
[the size-measure walk](../drips/how-big-the-correction-is/what-the-size-measure-means-and-what-it-misses.md)
which explicitly separates its own figure-design work from this walk's section-naming scope. Two
of its own held claims (the three-product comparison, the plurality name's earn-the-name test)
route to sibling idea maps under this same folder and return here through `integrate` before the
walk can produce a final `/drip-write` seed.

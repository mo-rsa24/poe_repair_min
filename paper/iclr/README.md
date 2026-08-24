# The figure-path rule

One root, one naming rule, so a figure file found on its own says which slot it fills and
LaTeX never carries a path.

**Root.** All figures live in `paper/iclr/figures/`. The preamble carries one line:

```latex
\graphicspath{{figures/}}
```

so every include is a bare filename: `\includegraphics[width=\textwidth]{compose-rate-as-correction-rises}`.
No `../`, no absolute path, and the draft still builds when the repo moves.

**Names.** `<slot>-<what-it-shows>[-<variant>]`, where `<slot>` is the register slot in
`figures.md`.

```
compose-rate-as-correction-rises.pdf              the paper figure for slot F2
compose-rate-as-correction-rises-with-a-random-control.pdf    the same result, laid out for a talk or poster
compose-rate-as-correction-rises-for-a-dissimilar-pair.pdf           a second cell for the same slot, answering one objection
```

A letter suffix on the slot (`F2b`) means a second figure serving the same claim, not a second
slot. It appears in the register row for `F2`.

**Formats.** Commit the PDF and the PNG. LaTeX includes the PDF; the PNG is what a reader opens
in a file browser or a markdown preview, and what the review files link to. Omit the extension
in `\includegraphics` so LaTeX picks the PDF.

**Size.** Paper figures are a few hundred KB and belong in the repo beside the manuscript. The
project's "large artifacts to `/datasets`" rule is about model checkpoints and generated-image
sweeps, not about the figures those sweeps produce.

**Regenerating.** Every figure here is built by a script that writes into this folder by
default, so a figure is never edited by hand and never drifts from the numbers behind it.

```bash
python scripts/make_f2.py --layout grid5 --seed 9    # compose-rate-as-correction-rises
python scripts/make_f2.py --layout shared0 --seed 9  # compose-rate-as-correction-rises-with-a-random-control
python scripts/make_f2.py --layout grid5 --pair an_elephant__x__a_penguin --seed 10 \
       --name compose-rate-as-correction-rises-for-a-dissimilar-pair
```

**Look at a grid before shipping it.** The scorer counts animals and does not check which
animals, so a panel can be labelled `compose` while holding two of the same concept. Cat × dog
seed 10 at full strength is two dogs. Seed 9 is a real cat beside a real dog, which is why the
figure uses it.

**Still owed by `writing-01`.** The `\graphicspath` line is not yet in
`iclr2027_conference.tex`, and there is no placeholder macro, so a section referencing a figure
that does not exist still breaks the build.

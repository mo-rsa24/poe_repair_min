# Do the successful cells really contain both animals?

Every cell behind F2's headline number, at full correction strength, sorted into folders by my
call. Disagree with any of it: the folder name is my judgement, the filename carries the facts.

## Why this exists

F2 says the compose rate reaches 94% at λ=1. That rate comes from a scorer that asks a detector
for "animal" and counts distinct boxes, so it can tell you two animals are present and cannot tell
you **which** two. A picture of two dogs scores exactly like a picture of a cat and a dog.

These are all 32 cells at λ=1, so this is the whole population the 94% is computed from, not a
sample.

## What is here

```
01-both-there            12   both requested animals are present, no argument
02-two-of-one             1   two of the same animal, scored as a success. A real error
03-cannot-call            5   I cannot decide, and a reader might reasonably disagree with me
04-look-alike-by-design  12   the two animals are near-identical, so nobody can call it
05-scored-failure         2   scored as a failure (one animal). Correctly, in both cases
```

30 of the 32 were scored `compose`, which is the 94%.

Filenames are `<pair>_seed<N>_n<count>.png`, where `n` is what the detector counted.
`contact-sheet.png` is all 32 on one page. `calls.json` carries the same table in machine form,
including the path each image was copied from.

## What each folder means, and what to look for

**01-both-there.** Open `a_cat__x__a_dog_seed9_n2.png`: a tabby cat sitting beside a white
labrador. That is what a correct success looks like.

**02-two-of-one.** One file: `a_cat__x__a_dog_seed10_n2.png`. Two dog muzzles, two black noses,
no cat. The detector counted two animals and was right; the composition failed and was recorded as
a success. This is the error the 94% contains.

**03-cannot-call.** Two white waterfowl where one should be a goose and one a swan. Two pinnipeds
where one should have tusks and does not clearly. Two brown raptors. If you can call these, your
call beats mine and the counts above should move.

**04-look-alike-by-design.** A leopard and a jaguar are both spotted big cats. A cow and a buffalo
are both dark bovines. A frog and a toad are both green amphibians. The pool chose these pairs
**because** they blend, which is the same property that makes "are both concepts present" impossible
to check afterwards. No better detector fixes this: the question has no answer from the image alone.

**05-scored-failure.** `an_elephant__x__a_penguin_seed9_n1.png` is two elephants and no penguin,
scored as a failure because the detector merged them into one box. Right answer, wrong reason.

## What this changes

One definite error in 30. Five more I cannot call. So the true rate at λ=1 sits somewhere around
87% to 94% rather than anywhere near 60%, and the paper should say 94% is an upper bound of about
that size rather than hedging vaguely.

It also settles a proposal against itself: querying the detector per concept ("cat", then "dog")
would decide the 5 cells in `03-cannot-call` and would be no better than guessing on the 12 in
`04-look-alike-by-design`. Not worth building.

## Provenance

Copies, not moves: the originals stay under `outputs/interaction_term/dose/pairs/`, because
`scripts/plot_dose_curves.py` scores from there and moving them would break the curve. Rebuild this
folder from `calls.json`, which records every source path.

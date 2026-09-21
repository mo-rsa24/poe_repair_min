# The report's illustrated map

Two pictures for [the report index](00-INDEX.md#read-it-in-this-order): one of what a single
corrector step does inside one noise level, and one of the whole reading arc as a single line.

2 prompts · 1 rendered · 0 waiting · 1 stuck

Each prompt is self-contained: paste either one into ChatGPT on its own and it comes back usable.
No style is prescribed, so each prompt carries its own subject in full concrete detail.

## Table of contents

- [Abstraction chain](#abstraction-chain)
- [Art direction](#art-direction)
- [Meaning palette](#meaning-palette)
- [Glyph vocabulary](#glyph-vocabulary)
- [Reading axes](#reading-axes)
- [Devices in play](#devices-in-play)
- [Subject lane](#subject-lane)
  - [Subject: What one corrector step is, inside one noise level](#subject-what-one-corrector-step-is-inside-one-noise-level)
- [Process lane](#process-lane)
  - [Process: The whole arc as one line](#process-the-whole-arc-as-one-line)

## Abstraction chain

Navigation: 📋 [TOC](#table-of-contents) | [Next](#art-direction) ➡️

1. [Closing the Compositional Gap](../plans/diagram-prompts.md): subject "The whole system on one page", process "One route from measuring tools to submission"
2. [The report's illustrated map](diagram-prompts.md): subject "What one corrector step is", process "The whole arc as one line"

The parent map draws the machine. This one draws what the machine returned, which is why its
process lane is ordered by the arc a reader follows rather than by the order the runs happened.
The parent map is drawn under the Glossy direction and this one is not, so the two sets do not
match by design.

## Art direction

Navigation: ⬅️ [Previous](#abstraction-chain) | 📋 [TOC](#table-of-contents) | [Next](#meaning-palette) ➡️

**Empty**, the guide's default. No visual style, palette, icon set or rendering technique is
prescribed, so each prompt's whole job is describing its subject concretely enough that the
composition follows from the content. The style block is embedded verbatim at the head of every
prompt below:

> Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering
> technique. Let the composition and visual treatment that best communicate this subject emerge
> from the context below, rather than from an imposed convention.

Zoom depth: one level. Each of the two pictures stands alone and does not open into children.
Where one needs more room, the parent map already holds the zoom.

## Meaning palette

Navigation: ⬅️ [Previous](#art-direction) | 📋 [TOC](#table-of-contents) | [Next](#glyph-vocabulary) ➡️

Empty prescribes no palette. What this set does hold constant is four *meanings*, stated in plain
words inside each prompt that needs them, so a reader moving between two pictures reads the same
thing the same way:

| Meaning | How it is said in the prompt |
|---|---|
| the uncorrected path | what plain product-of-experts does on its own |
| the correction | the step-by-step gap between the joined prompt and plain product-of-experts, and anything carrying or learning it |
| measured and passed | a bar met, a verdict of support |
| measured and failed | a bar not met, or a null: the numbers came out the same with the thing as without it |

Each prompt says which of the four it uses and asks for them to be told apart consistently. It
does not name the colours, because choosing them is the renderer's job under Empty.

A result that exists on disk with no finding file written is described as visibly provisional or
unfinished, at full size rather than faded, since it is unwritten and not weak.

## Glyph vocabulary

Navigation: ⬅️ [Previous](#meaning-palette) | 📋 [TOC](#table-of-contents) | [Next](#reading-axes) ➡️

No glyphs are assigned. Each prompt describes its cast in its own words. What recurs across the
set, and is therefore described the same way each time it appears:

**The generated picture.** A rendered image showing either two separate animals in one frame, or
one fused animal carrying features of both.

**The 50 denoising steps.** A long left-to-right progression from noise to a finished picture.

**The correction.** The step-by-step gap between what the joined prompt predicts and what plain
product-of-experts predicts.

**The adapter.** The small trained component that learns to predict that gap without ever seeing
the joined prompt.

**The counter.** The detector that counts how many animals a picture contains.

**One answered question.** A claim, the picture that shows it, the number that backs it, and the
verdict, kept in that order.

## Reading axes

Navigation: ⬅️ [Previous](#glyph-vocabulary) | 📋 [TOC](#table-of-contents) | [Next](#devices-in-play) ➡️

Dominant axis is left to right, and in the process lane it is the arc's order, rung 1 at the left
edge to rung 8 at the right. It is deliberately not time: the runs did not happen in this order,
and any picture implying they did is wrong. Each prompt states this in its own exclusions.

## Devices in play

Navigation: ⬅️ [Previous](#reading-axes) | 📋 [TOC](#table-of-contents) | [Next](#subject-lane) ➡️

None dictated. Empty leaves badges, panels, legends and dash conventions to the renderer. Where a
picture's own logic needs order shown, the prompt says the order matters and leaves how to show it
open.

## Subject lane

Navigation: ⬅️ [Previous](#devices-in-play) | 📋 [TOC](#table-of-contents) | [Next](#subject-what-one-corrector-step-is-inside-one-noise-level) ➡️

One piece, and it draws the mechanism the prose cannot carry: what a single corrector step does
to a latent inside one noise level, and why the two model readings taken there observe without
feeding back.

### Subject: What one corrector step is, inside one noise level

Navigation: ⬅️ [Previous](#subject-lane) | 📋 [TOC](#table-of-contents) | [Next](#process-lane) ➡️

[built] 🖼️ rendered 2026-09-07 as `diagrams/report-subject-one-corrector-step.png`
Save as: `diagrams/report-subject-one-corrector-step.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering technique. Let the composition and visual treatment that best communicate this subject emerge from the context below, rather than from an imposed convention.

Scene: landscape, 16:9. Almost the whole frame is one noise level out of fifty, marked off as a single enclosed region, with a narrow strip beyond its right edge standing for the next level. Inside the region, a partly-denoised image starts at the left. It is moved rightward by three repeated corrections, each one a pair of things happening together: a push along the direction the model says improves the image, and a small amount of fresh noise added back. After the third repetition the image settles, and this settled state is drawn larger than the ones before it. Below the sequence, clear of it, a mark showing that three is only an illustration and there are k of these repetitions. Above the settled state, two separate model readings are taken: what plain product-of-experts predicts at this point, and what the joined prompt predicts at this point. Each reading connects down to the settled state and goes nowhere else. Beyond the region's right edge, one step leaves the settled state and arrives at the next noise level.

Cast: the partly-denoised image as it arrives at this noise level; three drawn repetitions standing for k corrector steps, each a push plus a small re-noising; the settled image; the two model readings taken at the settled point, which are the plain product-of-experts prediction and the joined-prompt prediction; the single ordinary step onward to the next noise level.

Flows: the image moving left to right through the repetitions and into the settled state. Two readings drawn from the settled state out to the two prediction labels, pointing away from the latent, because each takes a measurement of it and changes nothing. One step leaving the region for the next level, which is the only thing crossing the boundary.

Text in the image: title "one noise level, k corrector steps, then one reverse step". Region title: "inside one of the 50 levels". Labels: "the latent, as it arrives", "push along the score", "add a little fresh noise", "the settled latent", "k of them", "what the product predicts here", "what the joint prompt predicts here", "on to the next level". A short key with exactly two entries: the path the image travels, and an observation that changes nothing, the second used for both model readings.

Exclusions: no connection from either model reading back into the sequence. No numbered step badges. No charts or curves. No components other than those listed. No placeholder or lorem text. No watermark.
```

Faithfulness note: neither model reading may send a line back into the sequence. Both are observers at the settled point, and drawing either one feeding the chain would turn the picture into a guided sampler, which is exactly what this corrector is not. Every repetition must sit inside the one region and the single step onward must be the only thing crossing the boundary, because the point is that the chain moves the image without advancing the denoising clock.


## Process lane

Navigation: ⬅️ [Previous](#subject-what-one-corrector-step-is-inside-one-noise-level) | 📋 [TOC](#table-of-contents) | [Next](#process-the-whole-arc-as-one-line) ➡️

One piece, and it carries the whole reading arc: eight stops from where the product works to what
is still open, with the detour through the alternatives that changed nothing. History:
`diagrams/process-versions/`. This lane is regenerated, never patched: it is keyed to the arc in
[the report index](00-INDEX.md#read-it-in-this-order), so any change to the arc rewrites it.

Marked `[planned]` because rungs 1, 2, 3 and 8 of the arc it draws have their evidence on disk and
no finding file yet, which is the same hole the arc marks with ⬜.

### Process: The whole arc as one line

Navigation: ⬅️ [Previous](#process-lane) | 📋 [TOC](#table-of-contents)

[planned] ⚠️ stuck: the detour must carry eleven stops, one per alternative, and the best attempt draws thirteen.
Last attempt: `temp/codex-drop/report-process-the-whole-arc.png` (canvas and title clean, count wrong).
Save as: `diagrams/report-process-the-whole-arc.png`

```
Style: none prescribed. Do not describe a visual style, color palette, icon set, or rendering technique. Let the composition and visual treatment that best communicate this subject emerge from the context below, rather than from an imposed convention.

Scene: landscape, 16:9. One continuous journey read left to right across the full width, with eight numbered stops. The journey's left-to-right order is the order a reader should meet the results, and explicitly not the order the experiments happened. The route begins in the condition of the uncorrected method and changes, once and only once, between stops 3 and 4, into the condition of the corrected method, staying that way through stops 5 and 6. Between stops 6 and 7 a wide detour leaves the route, passes through eleven small unnamed stops, and rejoins without having changed the route's condition at all. The route continues into stop 8 and ends there unfinished, drawn as an opening rather than a close. Three phase groupings sit behind the route: the first covering stops 1 and 2, the second covering 3 to 6, the third covering 7 and 8. Above each stop sits the outcome of that rung: stop 1 two separate animals passing, stop 2 one fused animal failing, stop 3 two separate animals passing, stop 4 a measurement rather than a picture, stop 5 passing, stop 6 mixed passing and failing, stop 7 one fused animal failing, stop 8 empty and unfinished.

Cast: the reading route; eight stops, which are when it works, where it breaks, the correction causes it, what it is made of, the adapter, unseen pairs, eleven cheaper fixes, and still open; eleven unnamed stops on the detour standing for the eleven alternatives; three phase groupings; eight outcomes; an unfinished ending.

Flows: one route changing condition exactly once, between stops 3 and 4. One detour leaving and rejoining between stops 6 and 7, unchanged by the trip.

Text in the image: title "read the results in this order". Phase titles: "the failure", "the fix", "the alternatives and what is left". Stop labels: "1 when it works", "2 where it breaks", "3 the correction causes it", "4 what it is made of", "5 the adapter", "6 unseen pairs", "7 eleven cheaper fixes", "8 still open". Detour label: "none of these changed the line".

Exclusions: no dates or timestamps anywhere, since left to right is reading order and not time. No suggestion that the experiments happened in this order. The detour must not change the route's condition. The ending must not be closed off. No placeholder or lorem text. No watermark.
```

Faithfulness note: the detour must rejoin the route in the same condition it left, because no alternative that avoided the correction changed the outcome, and stop 8 must end open rather than closed, because the arc ends in unfinished work. The condition change must fall between stops 3 and 4 and nowhere else, since rung 3 is where the correction first enters the story. No date may appear: the left-to-right axis is reading order, not a chronology.

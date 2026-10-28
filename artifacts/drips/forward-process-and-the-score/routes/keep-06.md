to: the forward-process-and-the-score walk (dispatch.md, on integrate)
from: keep, piece 06 (the cosine schedule > the boundary values C_k by hand > where p_k comes from)

destination: artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md, Step 4 (uncommitted; `git diff` on that file is the review)

kept: artifacts/drips/forward-process-and-the-score/kept/06-where-p-k-comes-from.md (message 4695b76c-6650-4b3a-88ce-ea57ea736016 at 2026-10-21T07:24:11.815Z, saved word for word under a source header)

## Where it went

Step 4, "✏️ On paper 4: where the position goes", now anchored `#step4-where-the-position-goes` (lines 697 to 806 of the step file). The structure holds: numbered moves, the p_k equation, a "Consequence." line and the existing "On the scene." line, with "Why … is there" chunks between the moves and the consequence, as the rest of Step 4 does.

## What it merged with

On paper 4 had four moves: the ruler, p = k/100 without the offset, the p_k equation with its (k + 0.8)/100.8 form, and marks at posts 0, 51 and 100. The ruler stays move 1. "Without the offset" became move 2, where u = k/T is introduced. The p_k equation stays as move 6, now boxed as the result of the derivation, and the sentence "multiplies the top and the bottom by 100" replaces the piece's version of the same point. The old move 4's three marks were folded into move 7's four-post aligned block, so each post is stated once. The Consequence line keeps both of its sentences and gains the 0.8929° per post. The On the scene line is unchanged apart from one added clause: with δ at 0 the frame's equation reads (k + 0.0)/100.0. That is the piece's screenshot remark, checked against the live frame's source (`(k+${(100*d).toFixed(1)})/${(100+100*d).toFixed(1)}`).

"Why the offset is there" (the slope of cos² near 0°, β₀, the pixel-bin rule) already covers why δ exists at all, so the new "Why the offset enters as b" chunk only says where post 0 lands and points down to it. "Why the ratio k/T is there" already says k/T maps posts onto 0 to 1; move 2 only names it u.

## What it added

The three letters, one sentence each with their code names in `rebuild/model.py` lines 13 to 17: k is `x = np.linspace(0, timesteps, steps)`, T is `timesteps`, δ is `s`.

Where the formula comes from: Nichol and Dhariwal (2021) equation 17, with their t, s and ᾱ_t = f(t)/f(0) translated to this page's k, δ and C_k, and f(k) as display maths with underbraces on the position and the quarter turn. p_k is named as this page's word for the bracket, a design choice the paper never derives, which three requirements nonetheless pin down.

The derivation as moves 2 to 6, each requirement as underbraced display maths: p = b + a·u (where post 0 sits, run per unit of u, k/T); p(1) = b + a = 1 (start plus the whole run, 90°); p = (u + δ)/(1 + δ) split into b = δ/(1 + δ) and a = 1/(1 + δ) with the check b + a = 1.

The worked straight line (move 7), all recomputed in python: a = 0.99206, b = 0.0079365, a/T = 0.0099206; p₀ = 0.00794 (0.714°), p₁ = 0.01786 (1.607°), p₅₁ = 0.51389 (46.25°), p₁₀₀ = 1 (90°).

"Why the straight line is there", with a check on neighbours that was not in the piece: p₅₁ − p₅₀ = 0.51389 − 0.50397 = 0.00992, times 90° gives 0.893°; 0.8929° per post, exactly 0.9° without δ.

"Why b + a = 1 is there", anchored `#step4-dropping-the-rescale`, as three moves on post 100: p₁₀₀ = 1.008, θ₁₀₀ = 90.72°, cos² 90.72° = 0.000158, so C₁₀₀ = 0.000158 instead of 0 (python: 0.00015791 unnormalised, 0.00015793 after dividing by the broken run's own post 0, both 0.000158). The last step never removes all the clean signal; move 4's requirement is the one that broke.

"The same line in the code": line 16 with bracketed labels under each piece (cos, u = k/T, + δ, ÷ (1 + δ), × π/2 (90°), ²), aligned by python against the real line.

Colour: C₁₀₀ in the broken end is green, the page's kept-signal-share colour. p, a, b, u and δ have no colour in the page's key, so they stay uncoloured rather than borrowing one.

## Links placed

On the words in On paper 4:

- "schedule function" → `../../../rebuild/model.py` (exists)
- "equation 17, p. 4" → https://arxiv.org/pdf/2102.09672#page=4 (read as text: equation (17) is on PDF p. 4, same URL form as Step 4's section 3.2 link)
- "straight line" → algebra handbook PDF p. 38, "Various Forms of a Line", slope-intercept y = mx + b (opened and read; its m and b translated in the line to a and b)
- "affine function" → Understanding Deep Learning PDF p. 53, notes "Linear, affine, and nonlinear functions" (opened and read: a weighted sum plus an offset is affine)
- "On paper 5" → `#step4-the-quarter-circle-by-hand` (anchor exists)

In the other direction, link-only, each wrapping words already there:

- The step file's "Why division by 1 + δ is there": "shift the endpoint as well as the start" → `#step4-dropping-the-rescale`
- Vault The Cosine Noise Schedule.md: "the position" in "Write $p_k$ for the position" → `#step4-where-the-position-goes`
- Vault The Quarter Circle Under The Cosine Schedule.md: "the schedule's position" in "**The position.**" → `#step4-where-the-position-goes`

Every relative link above resolves on disk, and each anchor appears exactly once as an `<a id>` in the step file.

## Left out on purpose

- The piece's opening paragraph about the trigonometry handbook. Step 4's existing "Quick reference" line already links the handbook at pp. 7, 10 and 16, so repeating it would say it twice.
- The control line, the "Read more" links (On paper 4 itself and `#timetable-posts`, which the On the scene line already carries) and the "No journey covers" footer. These are navigation, not content.
- The scene frame `#timetable-position-derived`. The live page is still version 1792566915-9476, whose timetable frames are plane, polar, radians, posts, quarter-turn, one-boundary, cancel, read-beta, two-fixes and designed-backwards. Once the running scene build publishes that frame, add it to On paper 4's On the scene line, for example: "[Derive the position](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-position-derived) puts handles on δ, T, a and b and a Divide by (1 + δ) switch."

## For dispatch.md

kept: artifacts/drips/forward-process-and-the-score/kept/06-where-p-k-comes-from.md → artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-where-the-position-goes

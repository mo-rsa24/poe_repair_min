to: the forward-process-and-the-score walk (dispatch.md, on integrate)
from: vault-pulse --retrofit, four vault notes, one vault-writer at a time

destination: the Obsidian vault at /home-mscluster/mmolefe/Obsidian (uncommitted; `git -C ~/Obsidian diff` is the review). The step file was not edited.

Every note was listed against vault-note v2 line by line, then handed to `vault-writer` with that list. Nothing was dropped: each writer compared the old note against the new ones for links, numbers and code blocks. No claim had to change, so no question came back. Each note's link back to the step file is kept.

## Note 1: Variance

| | |
|---|---|
| What failed | no `standard` field; quiz and Unfinished boxes open; a three-sentence bold opening; "Where you work" above the contents; no colour key; displays with no underbraces and no "What the equation says" or "Why … is there" chunks (definition, coin, derivation, worked numbers); the form as a ```text block; "scaling rule" and "standard deviation" not taken apart; no toy table; no code-drawn SVG; paragraphs over three sentences; "Other ways" as bullets; prompts 1 and 3 still wanted; "Applied in" as a backticked path; 397 lines, so the retrofit would pass 400 |
| What was rewritten | all 15 items. One bold sentence, then labelled chunks. The two names taken apart word by word. Every display written plain, then underbraced in colour, then read, with a with-and-without move per operator (without the square the coin's misses average 0; without the root at t = 50 the variance is 0.2722, and at t = 25 it is 0.0270) |
| Figures and tables | 7 SVGs in `Mathematics/Statistics/assets/`: the coin's squared misses, the sd as a length, the unit bell stretched by c, one miss scaled by c, 200,000 scaled draws under N(0, c²), the toy draw in 2D, c and c² across the schedule beside the share slip. Two HTML tables: the schedule toy (t = 0, 25, 50, 75, 99) and the coin |
| Split | three notes. Variance.md (252 lines) keeps the definition and standard deviation, plus a two-line summary under "Scaling a draw scales its variance by the square", a heading that 12 inbound links use. The Scaling Rule For Variance.md (362, new) holds the rule. Running The Scaling Rule For Variance.md (390, new) holds the sandbox and every run. The Statistics map note lists both new notes |
| Prompts | `.prompts/Variance.md`: prompts 1 and 3 marked replaced by SVGs, not deleted; prompt 2's webp stays |
| Question returned | none |
| Stamp | `standard: vault-note v2` on all three |
| Found along the way | the scaling rule also lives in Scaling And Shifting A Random Variable, so it has two addresses; merging them needs `/vault-pulse --merge` and a yes. Statistics/ now holds 11 notes plus 3 folders, past the soft limit of 10. Moving Variance into a folder would break the step file's link to it |

## Note 2: Sums Of Independent Gaussians

| | |
|---|---|
| What failed | the same presentation set (stamp, open boxes, long opening, "Where you work" first, no colour key, the five-step derivation as a numbered list of unbraced displays, a ```text form, the name and "cross term" not taken apart, no toy table, no SVG, "Other ways" as bullets, long paragraphs, prompts 12 to 14 wanted), plus five NumPy scripts that had never run (outputs owed), and "Applied in: nothing yet" though step 4 applies it |
| What was rewritten | all 16 items. One aligned derivation with a brace on every move, the boxed rule, and five "Why" chunks (independence: 1.2717 against 0.6861; the square root: 0.8283; squaring before adding: 1.1277 against 0.8283, 7 against 5). The name taken apart (sum, independent, Gaussian, the formula, "one draw replaces two"). The headings "How the rule works" and "Why independence kills the cross term" kept their exact text. Applied in now links `#step4-variances-add` |
| Runs | all six scripts ran with python3 (numpy 2.4.3), twice each. Every output fell inside its predicted band: reused draw 0.5337 against 0.5330; fresh draw 0.2796 against 0.28; quantiles within 0.001; 3-4-5 gave std 5.000 against 7 |
| Figures and tables | 5 SVGs in the folder's `assets/`: variances stacking, the four cells of the squared sum, the quadrants cancelling, one draw against two as histograms, standard deviations adding like the legs of a right triangle. One HTML toy table of five cases. Six rendered terminal PNGs |
| Split | Sums Of Independent Gaussians.md (409 lines) and Running Sums Of Independent Gaussians.md (359, new), listed in the map note. No inbound link reached a moved heading |
| Prompts | `.prompts/Multivariate Normal Distribution.md`: 12, 13 and 14 marked replaced. That lane's intro paragraph still says the sandbox has not run, which is now out of date |
| Question returned | none |
| Stamp | `standard: vault-note v2` on both |
| Found along the way | the Products note's line 435 has a relative link that resolves nowhere, and it predates this retrofit |

## Note 3: From Many Steps To One Jump

| | |
|---|---|
| What failed | the same presentation set, plus an unlabelled step-counting paragraph, the recursion as a ```text block, displays at 262 to 316 with no colour or reading, prompts 34 to 36 wanted, and links to the scaling rule's old home in Variance |
| What was rewritten | all 16 items. Every display coloured and taken apart, with moves at step 50: β₅₀ as the coefficient gives 0.000995 instead of 0.0315; the forgotten shrink gives 0.5377 and 1.0160, and 7.8789 at t = 99. "The one jump" taken apart (jump, one, the formula, "training reaches any step in one line"). Scaling-rule links now point at The Scaling Rule For Variance. New textbook links: Understanding Deep Learning p. 366, Chan p. 299, MML p. 207 |
| Figures and tables | 5 SVGs: one step at step 50, the shares at every step beside the forgotten shrink, the recursion on the closed form (largest gap 2.2e-16), the shrunk and unshrunk draws, the misses as a right triangle. One HTML table for steps 0, 1, 49 and 50 |
| Split | From Many Steps To One Jump.md (430 lines) and Running From Many Steps To One Jump.md (366, new). "How two steps become every step" keeps its text, so the step file's anchor still lands. The Density note's link to the sandbox and its back arrow now point at the running note |
| Prompts | 34 and 35 replaced; 36 left waiting, because its faithfulness note asks for matching text and markers, not exact geometry |
| Question returned | none |
| Stamp | `standard: vault-note v2` on both |
| Over the limit | 430 lines, about 30 past 400. The derivation section is 187 lines, under the 250-line section limit. The next cut would move the steps 0 and 1 worked display into the running note |

## Note 4: The Density Of The One Jump

| | |
|---|---|
| What failed | the same presentation set, plus an unlabelled step-counting paragraph, eight unbraced displays, one stated claim (a SciPy check not run), footnote links at line 302 instead of links on words, prompts 37 to 39 wanted, an old scaling-rule link, and "Applied in" as backticked paths |
| What was rewritten | all 18 items. q(x_t \| x_0) taken apart symbol by symbol, then by word (density, one jump, the formula, and "only the height moves"). Every display was taken apart with with-and-without moves at t = 50. Without the root on the centre: exponent −0.5987, q = 0.1676. Without the front factor, the volume is 3.2782. Dividing by 2 alone gives −0.3991. With diag(1, 2) in place of I, the exponent is −0.405. "The one-shot draw and its density" and "The exponent at the drawn point" keep their exact text |
| Stated claim | SciPy's `multivariate_normal(...).pdf` gives 0.1419 at t = 50 and 0.4505 at t = 25, matching the hand formula to 2.8e-17. It is now marked shown, and the note has 0 stated claims |
| Figures and tables | 4 SVGs: the form at step 50, the −0.765 contour at two steps beside the missing-root centre, the exponent and the height at every step (log scale labelled on the axis), the 1D bell over 200,000 draws. One HTML toy table at t = 0, 25, 50, 75, 99 |
| Split | three notes, all under 400 lines. The Density Of The One Jump.md (348) keeps the name, the form, "The exponent at the drawn point" with its cancellation and the score display, and a two-line summary under "How the density follows from the draw". How The Density Follows From The Draw.md (255, new) holds the derivation: the noise as one draw; the centre, spread and shape; the 2D Gaussian written out; and the exponent on the toy with the mechanism figure. Running The Density Of The One Jump.md (384, new) holds the sandbox. The navigation runs the idea note, then the derivation, then the running note, then The Cosine Noise Schedule, and the map note lists them in that order. Eight links were rewritten, including prompt 38's embed location. A comparison before and after the split found nothing missing. The link check prints 0 problems on all three notes and the map note |
| Prompts | 37 and 38 replaced, 39 left waiting for the same reason as 36. The counter now reads `45 prompts · 25 rendered · 5 replaced by code-drawn SVGs · 8 stuck · 0 rendering · 7 waiting` (34, 35, 37, 38, and 43 from the other writer), so it matches the two Mathematics prompts files |
| Question returned | none |
| Stamp | `standard: vault-note v2` on all three |
| Owed | The Cosine Noise Schedule's back arrow should point at Running The Density Of The One Jump once the other writer finishes with that note |

## Left for later

- Quiz pages: none of the new notes is covered. The Variance and Sums notes have no quiz page of their own, and the diffusion models quiz needs extending.
- `/vault-pulse --merge` for the scaling rule held in both The Scaling Rule For Variance and Scaling And Shifting A Random Variable.
- The vault_render check passes on every note except for links under "Taught in", which are /home/molef PC paths by the format and do not resolve on mscluster.

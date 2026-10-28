to: the forward-process-and-the-score walk (dispatch.md, on integrate)
from: vault-pulse --retrofit, seven vault notes (the cosine schedule and the trigonometry under it), one vault-writer at a time

destination: the Obsidian vault at /home-mscluster/mmolefe/Obsidian (uncommitted; `git -C ~/Obsidian diff` is the review). The step file was not edited.

Every note was listed against vault-note v2 line by line, then handed to `vault-writer` with that list and one shared brief: nothing dropped, the headings other files link to kept, every step-file and scene link kept, and the scene frames linked on the words that teach them. Each writer compared the old note (`git show HEAD`) against the new ones for links, numbers and code blocks.

## The Density split from the first pass

It finished. The Density Of The One Jump.md has 348 lines, How The Density Follows From The Draw.md 255 and Running The Density Of The One Jump.md 384, all stamped `standard: vault-note v2`, and the other writer's link check was clean. Running The Density Of The One Jump is the last note of that chain, and The Cosine Noise Schedule's back arrow now points at it.

## Note 1: The Cosine Noise Schedule

| | |
|---|---|
| What failed | no `standard` field; quiz and Unfinished boxes open; a bold opening of four sentences plus a two-sentence paragraph; "Where you work" above the contents; no colour key; displays only in underbraced form, never plain first; "boundary value" not taken apart, "cosine" with no equation of its own, no consequence heading; no code-drawn SVG at all; prompts 40 to 42 waiting behind "Diagram wanted" slots; no chunk for the "1 −" or the ratio in β; the derivation said "six moves" and showed four lines; paragraphs over three sentences; "Other ways" as bullets; backticked repo paths under "Taught in" and "Applied in"; the back arrow on the first Density note; 401 lines |
| What was rewritten | all 16 items. One bold sentence, then labelled chunks. "Boundary value" taken apart word by word, cosine given its own equation, and a consequence heading: the kept share after step t is the next post (the telescoping product matches C_{t+1} to 1.7e-18 through step 98; the clip makes ᾱ₉₉ = 2.4e-7 against C₁₀₀ = 0). New with-and-without moves: without the ratio β₅₀ would be about half; without "1 −" it reads 0.96845; without π/2 post 100 stops at 57.30° and keeps 0.29194; without the square C₅₁ = 0.69157; without k/T post 51 sits at 4554°. The derivation now shows all six moves at post 51 |
| Scene and step links | frames linked on their words: `#timetable-name-boundary-value`, `-name-cosine`, `-name-offset`, `-name-normaliser`, `#timetable-one-boundary`, `#timetable-why-offset` (on the pixel bin 1/127.5), `#timetable-normaliser-cancels`, `#timetable-position-derived` (on "position"); `#timetable-2` kept. Step links `#step4-where-the-position-goes` and `#step4-four-posts-by-hand` kept, `#step4-dropping-the-rescale` added on "post 100 sits at 90.72°"; "Applied in" links step 4 at `#step-4-the-forward-process-and-the-cosine-schedule` |
| Figures and tables | 4 SVGs: the fence of 101 posts with heights C_k beside δ = 0 (√β₀ 0.0251 against 0.0157, over the pixel bin 0.0078); β₅₀ read off posts 50 and 51 beside no ÷ (1 + δ) (post 100 keeps 0.000158, β₉₉ = −15.0 clipped to 0); one boundary at post 51 in six moves beside no denominator (C₀ = 0.99984, β₀ unchanged); the mechanism quarter circle at k = 0, 51, 100. The toy table gained a post 50 row (C₅₀ = 0.49384) |
| Split | three notes. The Cosine Noise Schedule.md (376 lines) keeps "The boundary values C_k" with its exact text, so the step file's link still lands. How Each Piece Of The Cosine Schedule Works.md (193, new) holds the derivation and mechanism, because the idea note still stood at 520 lines after the sandbox moved out. Running The Cosine Noise Schedule.md (245, new) holds "Try the boundary values yourself" with its heading kept; the three vault links to it (two in The Quarter Circle, one in The Quarter Turn) were rewritten. The map note lists both new notes; navigation runs Running The Density → The Cosine Noise Schedule → How Each Piece → Running The Cosine Noise Schedule → The Quarter Circle |
| Prompts | 40 and 41 replaced by SVGs; 42 left waiting, because it asks for matching output text and markers. Counter: `45 prompts · 25 rendered · 8 replaced by code-drawn SVGs · 7 stuck · 0 rendering · 5 waiting` |
| Questions returned | (a) "Then the word: noise" writes β₅₀ = 1 − 0.47826/0.49385 = 0.03155. C₅₀ is 0.493844, which rounds to 0.49384, and 0.47826/0.49385 gives 0.03157. Kept as written in that one equation; the table, chunk and figure print 0.49384. (b) "Why the denominator is there" says only the offset and the square change a step's noise. Removing the ÷ (1 + δ) also changes it (β₅₀ 0.031546 to 0.032198; β₉₉ from 0.999 to −15.0, clipped to 0), as the new β figure draws. Kept as written; the corrected version is "only the offset, the division by 1 + δ and the square change a step's noise; the denominator never does" |
| Stamp | `standard: vault-note v2` on all three |
| Nothing dropped | 15 of 15 code blocks byte for byte, 0 numbers missing, 0 aliases missing, 3 of 3 sources; the one in-note link to the formula now points across to the other note. The six sandbox scripts re-ran with identical output |
| Found along the way | "lines 13 to 17" and "lines 13 to 19" of `rebuild/model.py` are both right: 13 to 17 ends at the rescale that makes C_k, 13 to 19 is the whole function. The Diffusion Models folder now holds 12 notes plus Composition/, past the soft limit of 10 |

## Note 2: The Quarter Circle Under The Cosine Schedule

| | |
|---|---|
| What failed | no `standard` field; quiz and Unfinished boxes open; "Where you work" above the contents; displays only underbraced; the multipliers display with no colour, braces or reading, in a five-sentence paragraph; the post 51 worked numbers as inline maths; "quarter circle" and "position ruler" not taken apart; the consequence and the ÷ (1 + δ) overshoot with no figure; "Other ways" as bullets; scene frames only in a foot list; backticked repo paths; prompts 44 and 45 waiting. Found while writing: purple meant arc length here and the denominator in the cosine schedule notes |
| What was rewritten | all 14 items. Every display plain, then underbraced, then read. Moves: without the square roots the multipliers' squares add to 0.5009, without "1 −" to 0.9565; without the division post 0's multipliers are (0.9999, 0.0125) instead of (1, 0). "Quarter circle" taken apart (circle: radius 2 gives squares adding to 4; quarter: over a half turn post 51 sits at 92.50° with C₅₁ = 0.0019). "Position ruler" taken apart (ruler read as radians stops post 100 at 57.30°, keeping 0.29194; position alone gives 0.9° per post). The arc-length string is now brown, so purple means the denominator in every cosine schedule note |
| Scene and step links | on their words: `#timetable-posts` (101 posts), `#timetable-polar` (the two legs), `#timetable-plane`, `#timetable-radians` (the string), `#timetable-position-derived` (the position formula), `#timetable-name-offset`. `#step4-the-quarter-circle-by-hand` and `#step4-where-the-position-goes` kept, `#step4-dropping-the-rescale` added; "Applied in" links step 4 and `rebuild/model.py` |
| Figures and tables | s1-position-ruler-posts.svg redrawn with the overshoot row; quarter-circle-move-the-posts.svg (new: δ = 0 at the start, no ÷ (1 + δ) at the end, C₁₀₀ = 0.000158); s2-circle-point-is-the-multipliers.svg redrawn; quarter-circle-the-mechanism.svg (new). The posts table is kept |
| Split | three notes. The Quarter Circle Under The Cosine Schedule.md (415 lines, about 15 over 400; its longest section is 218 lines) keeps "Build the quarter circle by hand" and "Step 1: the position ruler" with their exact text. How The Quarter Circle Gives The Multipliers.md (264, new) holds "Step 2: the point is the two multipliers", text kept, and the derivation. Running The Quarter Circle Under The Cosine Schedule.md (181, new) holds "Try the quarter circle yourself", text kept. Links rewritten: three in The Quarter Turn And The Fraction and its back arrow, and the Step 2 link in The Cosine Wave. Map note updated |
| Prompts | 44 replaced by quarter-circle-the-mechanism.svg; 45 left waiting (it asks for matched output text and markers). Counter: `45 prompts · 25 rendered · 9 replaced by code-drawn SVGs · 7 stuck · 0 rendering · 4 waiting` |
| Question returned | none |
| Stamp | `standard: vault-note v2` on all three |
| Nothing dropped | every wikilink, markdown link and alias survives; the only numbers and code lines missing are in the Step 1 and Step 2 scripts, changed on purpose (a third strip row, brown for purple). The sandbox's nine steps re-ran from scripts copied byte for byte from the notes |
| Found along the way | three assets in the folder are embedded by no note: quarter-circle-no-offset-starts-on-the-flat.png, quarter-circle-the-denominator-cancels-in-beta.png, quarter-circle-three-posts-with-and-without-the-offset.png |

## Note 3: The Quarter Turn And The Fraction

| | |
|---|---|
| What failed | no `standard` field; quiz and Unfinished boxes open; "Where you work" above the contents; displays only underbraced; "How the denominator cancels" unbraced and unread; no with-and-without move for the square or the slope's 2 in step 3; "quarter turn" and "the fraction" not taken apart; the consequence table linking an SVG above the note's folder with an HTML href; the ÷ (1 + δ) row not drawn; "Other ways" as bullets; scene frames only in a foot list; backticked repo paths. Found while writing: β drawn green in s4 though the shared key makes it orange; no "When to reach for it" chunk; dividing one fraction by another leaned on and not linked |
| What was rewritten | all 14 items. Moves: the ∝ hides 1/cos²θ₀ = 1.00016 (with "=" C₀ would read 0.99984); without the square C₅₁ = 0.69157 and the last step drops 0.015584 instead of 0.000243; without the slope's 2 the predicted drop at post 51 is 0.007784 against 0.0155765 measured; without the ratio, C₅₀ − C₅₁ and f(50) − f(51) differ by the factor 1.00016 while both ratios give 0.031546; without "1 −" the noise reads 0.96845. "Quarter turn" taken apart (over a whole turn post 51 sits at 185° with C₅₁ = 0.9949). "The fraction" taken apart (numerator, denominator 0.99984, which δ = 0 makes 1). The consequence table now carries the ÷ (1 + δ) numbers (β₅₀ 0.031546 to 0.032198, β₉₉ −15.0 clipped to 0) and links The Cosine Wave on the word. No drawn picture was prompted: every picture here needs exact values, so code draws them |
| Scene and step links | on their words: `#timetable-name-cosine`, `#timetable-why-offset` (the slope −0.0249 per radian), `#timetable-quarter-turn` (every 15°), `#timetable-name-normaliser`, `#timetable-one-boundary` (the three bars), `#timetable-normaliser-cancels`, `#timetable-position-derived`; `#step4-dropping-the-rescale` in the table; "Applied in" links step 4 and `rebuild/model.py` |
| Figures and tables | quarter-turn-remove-a-piece.svg (new: without the square; posts 95 to 100 with and without ÷ (1 + δ)); s4-the-fraction.svg redrawn with orange β; s3 unchanged. Two HTML tables: cos² every 15° with drops and the drop without the square; β at steps 0, 50 and 98 from C and from f |
| Split | three notes. The Quarter Turn And The Fraction.md (395 lines) keeps "Step 3: the quarter turn, squared", "Step 4: the fraction" and "The consequence: remove a piece" with their exact text. Which Pieces Of C_k Change The Noise.md (153, new) holds the cancellation and the removals figure. Running The Quarter Turn And The Fraction.md (141, new) holds "Try the fraction yourself". Navigation ends ➡️ [[Composition]], whose back arrow now points at the running note. Map note updated |
| Prompts | none replaced or written; counter unchanged |
| Question returned | "Why the fraction matters" says only the offset and the square change a step's noise. The ÷ (1 + δ) changes it too (same numbers as note 1's question b). Kept as written; corrected version: "the offset, the division by 1 + δ and the square change it; the denominator only relabels the start" |
| Stamp | `standard: vault-note v2` on all three |
| Nothing dropped | 0 numbers, aliases or sources missing; 3 links changed on purpose (the HTML href, and two already repointed by the note 2 writer); 2 code lines changed on purpose (β colour). The nine steps re-ran with identical text |
| Found along the way | `find-in-textbooks.py` fails on mscluster (no `pdftotext`), so pages were opened with PyMuPDF. The pre-algebra handbook's p. 25 (dividing fractions) is a candidate for the "Pages looked up most" table. Diffusion Models/ now holds 16 notes plus Composition/ |

## Note 5: The Unit Circle

| | |
|---|---|
| What failed | no `standard` field; quiz and Unfinished boxes open; "Where you work" above the contents; four displays only underbraced; x² + y² = 1 with no "Why" chunk for the squares or the sum; "unit circle" and "polar coordinates" not taken apart; "How the unit circle works" as a numbered list of inline maths; "Other ways" as bullets with the scene frames only there; a five-sentence colour key and a four-sentence sandbox paragraph; backticked repo paths. Found while writing: no "When to reach for it"; step 3 named scripts by a rule the steps did not follow |
| What was rewritten | all 13 items. Moves: without the squares the equation is x + y = 1, a line crossing the 46.25° ray 0.7073 from the origin, inside the circle, and the bare legs add to 1.4139; x² alone is 0.4782 at 46.25° and 0.75 at 30°, so only the sum stays 1. "Unit circle" taken apart (circle: at radius 2, 1.9128 + 2.0872 = 4; unit: r = 1; consequence: x = 2 cos θ = 1.3830, twice the cosine). "Polar coordinates" taken apart, with the way back: r = √(0.4782 + 0.5218) = 1 and θ = tan⁻¹ 1.0446 = 46.25°. The derivation is one aligned six-line display, then coloured worked numbers |
| Scene and step links | `#timetable-plane` on "one point on this circle's quarter arc", `#timetable-polar` on "the two legs and their share bar", both also kept in "Drag it". "Taught in" links the walk's 00-INDEX.md, On paper 1 at `#step4-the-quarter-circle-by-hand`, On paper 2 at `#step-4-the-forward-process-and-the-cosine-schedule`. "Applied in" points at How The Quarter Circle Gives The Multipliers, Step 2 |
| Figures and tables | t9-why-the-squares-and-the-sum.svg (new: the line x + y = 1 against the circle; the radius-2 consequence). The angle table is kept. No drawn picture prompted: every picture needs exact geometry, and t1, t2, t3, t5 and t9 carry the form and the mechanism |
| Split | three notes. The Unit Circle.md (411 lines, about 11 over 400; largest section 193) keeps "The circle: every point at distance 1" and "The angle: one line from the origin" with their exact text. Polar Coordinates.md (241, new) holds "The point: from length and angle to x and y", text kept. Running The Unit Circle.md (177, new) holds "Try the unit circle yourself", text kept. 12 vault links rewritten (Radians, The Cosine Wave, Trigonometry, The Cosine Noise Schedule, The Quarter Circle, How Each Piece). Trigonometry.md gained two lines |
| Prompts | none; the Trigonometry notes have no prompts file |
| Question returned | none |
| Stamp | `standard: vault-note v2` on all three |
| Nothing dropped | 15 old code blocks all present (20 now); every link, decimal, degree value, image, alias and source kept. The sandbox re-ran from the filed code with output equal byte for byte |
| Found along the way | a new citation, the geometry handbook p. 92 (a circle as every point the same distance from the centre) |

## Note 6: Radians

| | |
|---|---|
| What failed | no `standard` field; quiz and Unfinished boxes open; "Where you work" above the contents; three displays only underbraced; arc length drawn purple while the cosine schedule notes use purple for the denominator; "Why π/180 is there" with no with-and-without move; "How radians work" as a numbered list of inline maths; "Other ways" as bullets with `#timetable-radians` only there; a four-sentence colour key and figure reading; backticked repo paths. Found while writing: no "When to reach for it"; a two-sentence bold opening; the walk's rung for radians not linked |
| What was rewritten | all 12 items. Arc length is brown (#6d4c41) in the key, the equations, the table header and the t4 script (plane.py was left alone, since two diffusion notes import its PURPLE for the denominator). Moves: without π/180, 46.25 is read as radians, seven turns plus 2.2677 rad, landing at 129.93° with cosine −0.6418 against 0.6915; without the seven turns cos(2.2677) is still −0.6418; without the radius the arc reads 0.8072 instead of 8.07 cm. The derivation is one aligned display (circumference 2πr, the share θ/360, s = rθ, 1° = 0.017453 rad, 1 rad = 57.2958°), then the worked numbers |
| Scene and step links | `#timetable-radians` on "highlights the arc 0.8072 and lays it flat beside π/2 = 1.5708", also kept in the foot chunk. "Taught in" links the walk's 00-INDEX.md, the kept rung 05-the-cosine-boundary-values-by-hand.md, and On paper 3 at `#step-4-the-forward-process-and-the-cosine-schedule` |
| Figures and tables | t4-degrees-and-radians-arc.svg redrawn in brown; t8-degrees-fed-as-radians.svg re-rendered unchanged. A real NumPy run added to "Other ways": `np.deg2rad([46.25, 90])` gives `[0.80721478 1.57079633]` |
| Split | none (364 lines) |
| Prompts | none; no picture here needs an image model |
| Question returned | the page gives the wrong cosine's square as 0.6418² = 0.4119, the square of the printed cosine. Unrounded, cos(46.25)² is 0.41197, which rounds to 0.4120. Kept as 0.6418² = 0.4119 with 0.41197 in the checked box |
| Stamp | `standard: vault-note v2` |
| Nothing dropped | no number missing; the only links gone are the two The Unit Circle links already moved to Running The Unit Circle; 12 code blocks became 14, the t4 script recoloured with identical output |
| Found along the way | the walk's 00-INDEX.md links rung files at the walk root, but they exist only under `kept/`. How The Quarter Circle Gives The Multipliers said Radians draws arc length in purple; that sentence was sent back to its writer to fix |

## Note 7: The Cosine Wave

| | |
|---|---|
| What failed | no `standard` field; quiz and Unfinished boxes open; a long bold opening carrying two ideas plus two sentences; a five-sentence colour key; "Where you work" above the contents; displays only underbraced; "Why the 2 is there" with no without-case; "cosine" with no chunk of its own, no "All the words together", and a consequence with no figure; "How the cosine wave works" mixing one unbraced display with inline maths; "Other ways" as bullets with `#timetable-quarter-turn` only there; backticked repo paths. Found while writing: no "When to reach for it"; the double-angle rule not linked to a textbook page; "The square of the cosine" at 276 lines after the rewrite, past the 250-line section limit |
| What was rewritten | all 14 items. One bold sentence ("The cosine wave is the point's x-coordinate on the unit circle, plotted against the angle"), then labelled chunks. Colour key agrees with The Unit Circle and Radians (brown arc length). Moves: without the 2 the steepest slope is −0.5 and the 30° to 45° drop predicts 0.1264 against 0.25 measured (0.2529 with it); without the minus sign the slope at 45° reads +1 while the table falls; undoubled, −sin 46.25° = −0.7224 is the slope of cos alone; half a turn gives cos(0.8072 + π) = −0.6915. "Cosine wave" taken apart (wave; cosine, where a straight line drops 0.1667 every 15° against 0.067 and 0.25; the formula; the consequence, whose last 15° drop 0.2588 against 0.0670). The slope and the drops are derived one step per line (0.2588 × 0.2618 = 0.0678, 0.9659 × 0.2618 = 0.2529) |
| Scene and step links | `#timetable-quarter-turn` on "cos² every 15° drops" and kept in "Drag it"; `#timetable-polar` on "split 1 into two shares"; `#timetable-why-offset` on "the chain rule slope −2 cos θ sin θ"; `#timetable-name-cosine` on "straight line from 1 to 0". "Taught in" links the walk's 00-INDEX.md, kept/05-the-cosine-boundary-values-by-hand.md, and On paper 2 and 5 at `#step-4-the-forward-process-and-the-cosine-schedule` |
| Figures and tables | t10-drop-the-square.svg (new: cos² flat at both ends beside cos alone ending at slope −1). The table's header recoloured to the key. New textbook link: the trigonometry handbook p. 37 for sin 2θ = 2 sin θ cos θ |
| Split | three notes. The Cosine Wave.md (387 lines) keeps "The two shares" and "The quarter turn: where the square is flat" with their exact text and a two-line summary under "How the cosine wave works". How The Cosine Wave Works.md (123, new) holds the derivations. Running The Cosine Wave.md (129, new) holds "Try the cosine wave yourself". One link rewritten (The Unit Circle now points at How The Cosine Wave Works for the identity). Trigonometry.md gained two lines |
| Prompts | none; every picture needs exact values, so code draws them |
| Question returned | none |
| Stamp | `standard: vault-note v2` on all three |
| Nothing dropped | against the pre-edit copy nothing missing; against git HEAD only the three links earlier writers had already repointed. Filed Python blocks byte-identical to what ran |
| Found along the way | the repo step file's On paper 5 links The Cosine Wave.md with no heading (left alone, as the step file is not edited) |

## Note 4: Trigonometry (the map), done last

| | |
|---|---|
| What failed | no `standard` field; the position line fused into the opening paragraph ahead of the bold sentence, whose second sentence was long; seven children in one flat list, past the six at which a map groups by part; no "Words this uses" block though the notes share vocabulary |
| What was rewritten | all 6 items. The position line on its own line, then a three-sentence bold opening, then a "The toy angle" chunk (46.25°, the angle of post 51). Children under three headings in reading order: "The circle and the point", "The angle as a length", "The wave and its flat ends", each child's line kept word for word after checking it against its note. A "Words this uses" block of 20 words, each linked to the heading that teaches it. The closing line is now a "One helper, one folder." chunk naming the run folder. The Mathematics map's Trigonometry line was checked and left as it is |
| Figures and tables | none |
| Split | none (59 lines) |
| Question returned | none |
| Stamp | `standard: vault-note v2`, the vault's first v2 map note |

## Checked at the end

All 17 notes this pass wrote carry `standard: vault-note v2`. Line counts as measured after the last write: The Cosine Noise Schedule 376, How Each Piece Of The Cosine Schedule Works 193, Running The Cosine Noise Schedule 245, The Quarter Circle Under The Cosine Schedule 415, How The Quarter Circle Gives The Multipliers 264, Running The Quarter Circle Under The Cosine Schedule 181, The Quarter Turn And The Fraction 395, Which Pieces Of C_k Change The Noise 153, Running The Quarter Turn And The Fraction 141, Trigonometry 59, The Unit Circle 411, Polar Coordinates 241, Running The Unit Circle 162, Radians 364, The Cosine Wave 386, How The Cosine Wave Works 123, Running The Cosine Wave 129.

`vault_render.py check` over those 17 plus the Diffusion Models and Composition map notes leaves 1 problem: the `/home/molef/...` chapter 9.3 link under The Cosine Noise Schedule's "Taught in", a PC path the format requires, which does not resolve on mscluster.

Every step-file anchor and scene frame the seven notes had at HEAD is still linked (`#step4-four-posts-by-hand`, `#step4-the-quarter-circle-by-hand`, `#step4-where-the-position-goes`, `#step-4-the-forward-process-and-the-cosine-schedule`, `timetable-2`, `-one-boundary`, `-plane`, `-polar`, `-posts`, `-quarter-turn`, `-radians`). Newly linked: `#step4-dropping-the-rescale` and all seven new frames (`-position-derived`, `-name-boundary-value`, `-name-cosine`, `-name-offset`, `-name-normaliser`, `-why-offset`, `-normaliser-cancels`). The two headings the step file links to, "The boundary values C_k" and "Build the quarter circle by hand", keep their exact text.

After the splits, three stale sentences in the Quarter Circle notes were sent back to their writer and fixed: the scripts that run in the shared folder (twice), and a colour sentence about Radians.

## Left for later

- The question both cosine schedule notes returned: removing the ÷ (1 + δ) also changes a step's noise (β₅₀ 0.031546 to 0.032198), so "only the offset and the square change it" is wrong as written in The Cosine Noise Schedule and The Quarter Turn And The Fraction. Both sentences are kept until the learner says to change them.
- Two rounding slips kept as written: C₅₀ printed as 0.49385 (it is 0.49384) in "Then the word: noise", and the wrong cosine's square given as 0.4119 (unrounded 0.41197, so 0.4120) in Radians.
- Three notes sit just past 400 lines: The Quarter Circle Under The Cosine Schedule (415), The Unit Circle (411), and From Many Steps To One Jump (430, from the first pass).
- Diffusion Models/ holds 15 notes beside its map note, plus Composition/, far past the soft limit of 10, so the cosine schedule notes are candidates for their own sub-folder. Moving them would break the step file's links, which this pass does not edit.
- Prompts 42 and 45 (the deep-dive sheets) are still waiting for `/render-diagrams`. Counter: `45 prompts · 25 rendered · 9 replaced by code-drawn SVGs · 7 stuck · 0 rendering · 4 waiting`.
- No quiz page covers any of the 17 notes.
- Three unused assets in Diffusion Models/assets/: quarter-circle-no-offset-starts-on-the-flat.png, quarter-circle-the-denominator-cancels-in-beta.png, quarter-circle-three-posts-with-and-without-the-offset.png.
- The walk's 00-INDEX.md links rung files at the walk root, but they live under `kept/`.
- `find-in-textbooks.py` needs `pdftotext`, which mscluster lacks.

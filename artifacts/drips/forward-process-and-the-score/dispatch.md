scope: none (layer plan: artifacts/drips/simple-distributions-rebuild/ladder/01-one-diffusion-expert.md, task 2 "You: the maths drip")
source: artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md steps 4, 5 and 8; rebuild/out/walk-numbers.json (p1.abar_*, p1.sqrt_1m_abar_*, p1.example_*)
why: derive the sampled-noise identity and the marginal identity by hand before quiz 1.2, step 6's scene and step 7's cross-links read this folder

ladder: 5 pieces, 18 rungs
  1 the forward process: scaling a draw scales variance by c² · independent variances add · two steps then induction · the one-shot draw and its density q(x_t | x_0)
  2 the cosine schedule, given: boundary values C_k · β_t from consecutive ratios · telescoping to ᾱ_t = C_{t+1} · index shift ᾱ_0 = 0.9994 and the 0.999 clip at t = 99
  3 the sampled-noise identity: log of the isotropic 2D Gaussian · gradient of a squared distance · substitute the forward draw and cancel
  4 the toy by hand: x_50 from x_0 and ε · the score two ways, (−0.4153, 1.6613)
  5 the marginal identity: squared error is minimised by the conditional mean · q_t as an average over clean points · ∇q = q∇log q under the integral · divide by q_t, Bayes weights · pull out the constant
quiz 1.1 misses: none recorded (the quiz URL slot in the layer plan is still empty); first rungs start at the floor of piece 1

rung: scaling a draw by c scales its variance by c² (piece 1, rung 1)
notation: ε for the standard normal draw, c for the multiplier, Var for variance, σ_t = √(1 − ᾱ_t) not yet introduced
examples: ε = ±1 coin toy with c = 0.7223 → variance 0.5217; c = √(1 − ᾱ_50) = 0.7223 → 0.5217 = 1 − 0.4783; c = √β_0 = 0.02512 → 0.000631; 2D ε = (0.3, −1.2) × 0.7223 = (0.2167, −0.8668)
clicked [my read]: (empty until the reader responds)
wants to see move: (empty until the reader says)

latex: rung 1 re-emitted as one compilable file
asked [yours]: context, then thread (on piece 1, rung 1)
thread emitted: why the schedule stores variance shares (β_t, ᾱ_t) while the forward formula needs standard deviations (√β_t, √(1 − ᾱ_t)); not launched; returns via integrate in session 9484913d-a247-4bdb-bb44-0f11895d96cc
open: vault writer's placement question (variance or forward); two Variance pictures stuck (draw by script); Variance quiz waits on prep-cadence
thread launched: background session f3ca2449 "forward-process-and-the-score · thread · variance shares" (claude attach f3ca2449); row out

example: rung 1 at t = 25 and t = 75; c = 0.4054 → 0.1644, c = 0.9309 → 0.8665; four-value draw ±1.2247, ±0.7071 (mean square 1) × 0.9309 → mean square 0.8666; wrong unit at t = 25: × 0.1644 → 0.0270; 2D toy (0.3, −1.2) → (0.1216, −0.4865) at t = 25, (0.2793, −1.1171) at t = 75
sketch [asked: visual]: ASCII bars of typical miss c against variance c² at t = 0, 25, 50, 75, 99, 0.05 per block
rung: independent variances add (piece 1, rung 2)
notation: X, Y for two independent mean-0 draws; cross term 2XY
examples: two ±1 coins → sums 2, 0, 0, −2, mean square 2 = 1 + 1; Y = X (not independent) → Var(2X) = 4; 0.4054ε_a + 0.7223ε_b → 0.1644 + 0.5217 = 0.6861, typical miss 0.8283, not 1.1277
correction: t = 75 example used c rounded to 0.9309; exact c = √0.8665 gives 2D toy (0.2793, −1.1170) and mean square 0.8665 (vault writer's run)
asked [yours]: thread --derive on rung 2 (emitted, not launched)
asked [yours]: could this session add to the "Dissolving the Dog" artifact (claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9)? Its Derive frames 2 and 3 already teach rungs 1 and 2 with 0.7223 → 0.5217; proposed additions: a "noise copies the picture" preset on frame 3 (cross term, 1.96 not 1), and variance-share lines beside the coefficient curves on "Dog fades, noise grows"
artifact: added both to Dissolving the Dog (version 1792554841-3336): Derive 3 preset 'Noise copies the picture' (a = 0.6, b = 0.8 → 1.96 with cross term 0.96); dotted variance-share lines with labelled markers on 'Dog fades, noise grows'. Merged onto another session's 03:48 version. Checked by node --check only, not viewed in a browser

rung: two steps then induction (piece 1, rung 3)
notation: s_t for the state after code step t (the step file's x_t), x_0 the clean point, z the state before two generic steps
examples: steps 0 and 1: β_0 = 0.0006313, β_1 = 0.0011169, noise 0.0006306 + 0.0011169 = 0.0017475 = 1 − ᾱ_1, signal √0.9982525 = 0.9991; step 50: ᾱ_49 = 0.4938, β_50 = 0.03155, shrunk old noise 0.4902 + fresh 0.0315 = 0.5217 = 1 − ᾱ_50, signal √0.4783 = 0.6916; broken (old noise not shrunk): 0.5377, total variance 1.0160

rung: the one-shot draw and its density q(x_t | x_0) (piece 1, rung 4; closes piece 1)
notation: N(x; µ, σ²I), σ² = 1 − ᾱ_t, mean √ᾱ_t x_0, d = x_t − √ᾱ_t x_0 the displacement
examples: 1D clean 0.5 at t = 50: mean 0.3458, sd 0.7223, peak 0.5523; toy t = 50: mean (0, 0.3458), x_50 = (0.2167, −0.5210), |d|² = 0.7983, exponent −0.765, prefactor 0.3050, q = 0.1419; toy t = 25: x_25 = (0.1216, −0.0295), |d|² = 0.2515, exponent −0.765, q = 0.4505; exponent = −|ε|²/2 = −0.765 at every t; broken mean ᾱ x_0 = (0, 0.2391) gives d = (0.2167, −0.7601), not 0.7223ε

scene: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9
asked [yours]: record each piece durably (notes, textbook or plan, and the scene) before moving on
scene dispatched: pieces 1.1..1.4 against Dissolving the Dog; Derive 2, 3 and 5–6 already show rungs 1–3, so expected to append the density frame for rung 4 and link the existing frames

rung: the cosine boundary values C_k by hand (piece 2, rung 1)
notation: θ_k = ((k/T + δ)/(1 + δ))·90°, C_k = cos²θ_k / cos²θ_0, δ = 0.008, T = 100
examples: θ_0 = 0.714°, cos²θ_0 = 0.99984 → C_0 = 1; θ_1 = 1.607°, cos² = 0.99921 → C_1 = 0.99937 (= ᾱ_0); θ_51 = 46.25°, cos = 0.6915, cos² = 0.47819 → C_51 = 0.47826 (= ᾱ_50); θ_100 = 90° → C_100 = 0; broken without δ: C_1 = cos²(0.9°) = 0.99975, first noise share 0.000247, sd 0.0157 not 0.0251
scene: Timetable 2 of 5 already computes one C_k piece by piece; nothing new to append

why [asked]: δ pushes post 0 off 0°, where cos² is flat (slope −2·cos·sin = 0), so the first step adds 0.000631 not 0.000247; Nichol & Dhariwal 2021 §3.2 gives the reason as tiny early noise making ε hard to predict (verify against the paper)
asked [yours]: repeat the boundary-values rung verbatim with by-hand guidelines in between: Cartesian plane, polar coordinates, a line from the origin, degrees and radians, where the position goes, the quarter turn, then C_k; name each consequence and illustrate it in the scene if missing, especially the C_k fraction
examples: unit circle at 46.25°: (cos, sin) = (0.6915, 0.7224) ≈ (√ᾱ_50, √(1 − ᾱ_50)) = (0.6916, 0.7223); 0.4782 + 0.5218 = 1; arc 0.8072 rad; cos² at 0, 15, …, 90°: 1, 0.933, 0.75, 0.5, 0.25, 0.067, 0; denominator 0.99984 cancels in every β
clicked [my read]: wants the geometry before the formula; the circle point as (signal multiplier, noise multiplier) is the bridge
scene queued (after the running dispatch returns): Timetable 2 already has the quarter circle, the angle in degrees and radians, cos as a length, cos² as a bar, δ = 0, and f(k)/f(0) in its equation; to add: the point's y = sin as the noise multiplier with x² + y² = 1, radians as arc length, a position ruler with the posts and δ, and numerator and denominator bars with a switch that drops the denominator
vault: this turn held until the running writer reports

integrate: scene return routes/scene-1-1-to-1-4.md (version 1792564498-9605): rung 1 → #derive-2, rung 2 → #derive-3, rung 3 → #derive-5 and #derive-6, rung 4 → #derive-7 (new: the one-shot draw and its density on the 2D toy, prediction slot, broken-centre toggle; still at artifacts/results/how-the-forward-process-dissolves-an-image/the-one-shot-draw-and-its-density.png); browser-walked at 1440 px light and 390 px dark; not done: Codex mathematics review, the q(x_t | x_0) link to the Chan tutorial p23 unchecked
scene dispatched: the circle geometry before C_k (queued item above)
correction [vault writer]: 'squaring makes it a share between 0 and 1' was loose; on a quarter turn cos θ is already in [0, 1]. Read through the circle: cos θ is the signal multiplier and cos²θ its variance share (rung 1: share = multiplier²); Nichol & Dhariwal call the cos² shape an arbitrary choice. Filed in The Cosine Noise Schedule.md

asked [yours]: scene including all the mathematical steps, and links & polish including all the mathematical steps
scene: addendum sent to the running circle dispatch: the formula read factor by factor, posts 0, 1, 51, 100 with every intermediate, δ = 0 broken run, the six by-hand steps, the squaring correction
links dispatched (piece 2 rung 1 with its by-hand build-up): textbook-cadence → routes/links-textbook.md, paper-cadence → routes/links-paper.md (also checking the Nichol & Dhariwal §3.2 offset reason), video-cadence → routes/links-video.md, xref-cadence → routes/links-ours.md
polish: run in this session over every taught rung (piece 1 rungs 1–4, piece 2 rung 1 with its by-hand build-up), --thread forward-process-and-the-score --math

session restarted: the scene circle dispatch, the vault writer for the circle turn and the four link finders stopped unfinished; polish stopped after 00-INDEX.md and 01-scaling-a-draw-scales-its-variance.md
asked [yours]: continue, and use keep to persist everything in this session so far
keep dispatched: all five taught pieces (1.1 to 1.4, 2.1 with its why and by-hand build-up) into the step file 01-one-expert-from-scratch.md, Step 4, one writer in sequence; originals from the session transcript to kept/; links placed both ways by keep, which also covers the stopped link finders' remit and the §3.2 citation check
resumed: the scene circle dispatch and the vault writer for the circle turn
polish: not continued; keep files each piece into the step file, the one ladder, so 02 to 05 polished copies would duplicate it

integrate: keep return routes/keep-all.md; originals in kept/01 to kept/05
kept: 01 → 01-one-expert-from-scratch.md#step4-the-scaling-rule · 02 → #step4-variances-add · 03 → #step4-two-steps-then-every-step · 04 → #step4-the-one-shot-density · 05 → #step4-the-quarter-circle-by-hand and #step4-four-posts-by-hand
citation [checked by keep]: Nichol & Dhariwal 2021 §3.2, p. 4 (arXiv PDF): the offset prevents β_t from being too small near t = 0, because tiny early noise made ε hard to predict; s is chosen so √β_0 sits just under the pixel bin 1/127.5 (holds at T = 1000, √β_0 = 0.0064; not at T = 100, 0.0251); the paper calls cos² arbitrary; Chan p. 23 (Theorem 2.2, eq. 2.8) confirmed for q(x_t | x_0)
open [question for the reader]: replace step 4's "Squaring keeps the boundary nonnegative and makes it represent a variance share" with the multiplier-to-share reading, keeping "without the square, the schedule would follow a different curve"?
open: a second writer edited the step file during keep (the density heading, Derive 7 to 9); the circle frames are not on the live scene yet
asked [yours]: fix the format and retrofit notes; a mode to retrofit vault notes to the latest standard
format: VAULT_NOTE_FORMAT.md v2 (new section "How a section reads on the page", standard stamp, pictures after code-drawn SVGs); STANDARDS.md vault-note row at v2 with changelog; vault-pulse --retrofit added
retrofit queued: /vault-pulse --retrofit on this walk's notes once the running vault writer (trig note, schedule SVGs) returns
retrofit started: Variance, Sums Of Independent Gaussians, From Many Steps To One Jump, The Density Of The One Jump → routes/retrofit-1.md; the cosine, quarter-circle and trig notes follow once their writer returns

integrate: scene return routes/scene-2-1.md (version 1792566915-9476): piece 2 rung 1 → #timetable-plane, #timetable-polar, #timetable-radians, #timetable-posts, #timetable-quarter-turn, and the repaired #timetable-one-boundary (seven term-by-term steps, presets k = 0, 1, 51, 99, 100 and δ = 0, numerator/denominator/ratio bars with the divide switch); old #timetable-1..5 links still open their frames; browser pass at 1440 px and 390 px; not done: Codex mathematics review, small text on phones in timetable frames 1, 8, 9, 10
renumbered: another session inserted a new Derive 7, so rung 4's density frame is now #derive-8; fixed in the results README and capture_density_frame.mjs; routes/ and kept/ left as records

asked [yours]: did keep add the "On paper" steps somewhere? wants to read them, and a scene that breaks them down: names apart, consequences, derivations
answer: yes, Step 4 #step4-the-quarter-circle-by-hand to #step4-four-posts-by-hand; reshaped each On paper step into numbered moves, a Consequence line and an "On the scene" link to its new timetable frame
scene dispatched: name frames (boundary value, cosine, offset, normaliser), derive frames (slope of cos² for why δ; the denominator cancelling in β_50), a consequence sentence on each circle frame
note: chat renders "$0.99984$"-style inline maths raw (a $ followed by a digit), seen in the learner's screenshots; write bare numbers in chat replies

asked [yours]: a trigonometry handbook quick-reference in Step 4 or the vault
done: Step 4 "Quick reference." line under the cosine schedule paragraph, linking the vault's copy (pp. 7, 10, 16)
asked [yours]: where is p_k taken from, can it be derived, what are k, T and δ (child of piece 2 rung 1)
rung: where the position p_k comes from (child)
notation: u = k/T the fraction of the schedule done; p = b + a·u the straight line; a = 1/(1 + δ), b = δ/(1 + δ); code x ↔ k, timesteps ↔ T, s ↔ δ, np.pi*0.5 ↔ 90° (rebuild/model.py lines 13–17)
examples: a = 0.99206, b = 0.0079365; posts 0, 1, 51, 100 at 0.00794, 0.01786, 0.51389, 1 (0.714°, 1.607°, 46.25°, 90°); spacing 0.8929° per post (0.9° without δ); broken (no ÷ (1 + δ)): post 100 at 1.008, 90.72°, C_100 = 0.000158 instead of 0
vault: this turn held until the running writers report
asked [yours]: keep, and scene with the consequences of moving each term visible
keep dispatched: piece 06 (where p_k comes from) into Step 4's On paper 4 → routes/keep-06.md, original to kept/06-where-p-k-comes-from.md
scene: addendum to the running name/derive dispatch: #timetable-position-derived, with handles on δ, T, a and b, a "Divide by (1 + δ)" switch, and a p = u² preset, each moving the posts, the C_k curve and the β_0, √β_0, post-100 and C_100 readouts live
integrate: keep return routes/keep-06.md
kept: artifacts/drips/forward-process-and-the-score/kept/06-where-p-k-comes-from.md → artifacts/drips/simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-where-the-position-goes
open: link #timetable-position-derived from On paper 4's "On the scene" line once the scene publishes it (ready line in routes/keep-06.md)

rung: β_t from consecutive posts (piece 2, rung 2)
notation: C_t the kept share before step t, C_{t+1} after it; 1 − β_t = C_{t+1}/C_t; code np.clip(1 - ac[1:] / ac[:-1], 0, 0.999) with ac[1:] ↔ C_{t+1}, ac[:-1] ↔ C_t
examples: t = 0: 1 − 0.999369/1 = 0.000631; t = 50: 1 − 0.478265/0.49384 = 1 − 0.96845 = 0.03155; t = 98: 1 − 0.25006 = 0.7499; t = 99: C_100 = 0, raw β = 1 (clip is rung 4); β at 0, 25, 50, 75, 98 = 0.000631, 0.01345, 0.03155, 0.0747, 0.7499 against absolute drops 0.00063, 0.01139, 0.01558, 0.01078, 0.00073; broken (absolute drop used as β): kept share after step 50 is 0.5916, not 0.4783
vault: held until the running writers report
integrate: vault writer (trig and schedule figures): Mathematics/Trigonometry branch (map + The Unit Circle, Radians, The Cosine Wave; 8 SVGs incl. degrees-fed-as-radians consequence); Diffusion Models: The Cosine Noise Schedule maths-first with names apart, The Quarter Circle split into it plus The Quarter Turn And The Fraction; s1–s4 SVGs; prompt 43 replaced, 11 waiting
linked from the step file: On paper 2 → The Unit Circle, 3 → Radians, 5 → The Cosine Wave, 6 → The Quarter Turn And The Fraction
open [question for the reader]: trigonometry as a map plus three notes (as filed) or one note with a figures sub-note
retrofit queued: The Cosine Noise Schedule, The Quarter Circle Under The Cosine Schedule, The Quarter Turn And The Fraction, and the trigonometry notes, after retrofit-1 returns (written before v2 landed)
integrate: scene return routes/scene-2-1-gaps.md (version 1792570196-e0a3, Codex review 8 of 9 then fixed): #timetable-position-derived, #timetable-name-boundary-value, -name-cosine, -name-offset, -name-normaliser, #timetable-why-offset, #timetable-normaliser-cancels; consequence line on every circle frame; linked from Step 4 (names sentence, On paper 4, why the offset, On paper 6); results README gained three stills' rows and fixed Derive 7 → 8
integrate: retrofit return routes/retrofit-1.md: Variance (split into 3), Sums Of Independent Gaussians (split into 2; its five NumPy scripts run for the first time, all inside their predicted bands), From Many Steps To One Jump (split into 2), The Density Of The One Jump (split; a third split into How The Density Follows From The Draw.md unconfirmed); all stamped vault-note v2; 21 SVGs; nothing dropped, no claim changed
open [decisions for the reader]: merge The Scaling Rule For Variance with Scaling And Shifting A Random Variable (vault-pulse --merge needs a yes); Statistics/ at 11 notes; From Many Steps (430) and the Sums rule note (409) over ~400 lines; no quiz pages for the new notes
retrofit-2 dispatched: The Cosine Noise Schedule, The Quarter Circle Under The Cosine Schedule, The Quarter Turn And The Fraction, Trigonometry map, The Unit Circle, Radians, The Cosine Wave → routes/retrofit-2.md
integrate: Density third split finished: The Density Of The One Jump (348), How The Density Follows From The Draw (255, new), Running The Density Of The One Jump (384), all v2, link check clean; retrofit-2 told to point The Cosine Noise Schedule's back arrow at Running The Density Of The One Jump
integrate: retrofit return routes/retrofit-2.md: 7 notes → 17 stamped v2 notes (Cosine Noise Schedule, Quarter Circle, Quarter Turn And The Fraction, Unit Circle, Cosine Wave each split into idea / how it works / running; Radians whole; Trigonometry map regrouped); 13 SVGs; the Cosine Noise Schedule back arrow now points at Running The Density Of The One Jump; vault changes uncommitted (git -C ~/Obsidian diff)
verified [python]: removing ÷ (1 + δ) changes β_50 from 0.031546 to 0.032198, so "only the offset and the square change the noise" is wrong; C_50 is 0.49384 (printed 0.49385, which gives 0.03156); cos(46.25 rad)² = 0.41197 → 0.4120 (printed 0.4119)
open [question for the reader]: fix those three in the vault notes
fixed: 00-INDEX.md links for pieces 2 to 5 pointed at polished copies that were never written; they now point at the Step 4 anchors where keep filed them; p_k child and β rung added

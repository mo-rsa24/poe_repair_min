# The forward process, and why the noise prediction is the score

Captured from `/drip --math`, ongoing.
Ran under: none: captured from chat, for task 2 of the [layer 1 plan](../simple-distributions-rebuild/ladder/01-one-diffusion-expert.md) · why: derive the sampled-noise identity and the marginal identity by hand before quiz 1.2, step 6's scene and step 7's cross-links read this folder.

Each piece carries its chain status: taught (in chat by /drip), filed (kept in Step 4 of the step file, or captured here), and `scened` where a frame of the walk's scene, [Dissolving the Dog](https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9), shows it. No marks means not taught yet. The walk's own record (notation, worked numbers, what clicked) is [the dispatch file](dispatch.md).

Source: steps 4, 5 and 8 of [the phase 1 step file](../simple-distributions-rebuild/01-one-expert-from-scratch.md#step-4-the-forward-process-and-the-cosine-schedule), with numbers from `rebuild/out/walk-numbers.json`.

## The forward process
1. [Scaling a draw scales its variance by c²](01-scaling-a-draw-scales-its-variance.md) (taught, filed, scened)
2. [Independent variances add](../simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-variances-add) (taught, filed, scened)
3. [Two steps, then every step](../simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-two-steps-then-every-step) (taught, filed, scened)
4. [The one-shot draw and its density](../simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-the-one-shot-density) (taught, filed, scened)

## The cosine schedule, taken as given
5. [The boundary values C_k by hand](../simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-the-quarter-circle-by-hand) (taught, filed, scened)
   - [Where p_k comes from](../simple-distributions-rebuild/01-one-expert-from-scratch.md#step4-where-the-position-goes) (taught, filed, scened)
6. β_t from consecutive posts (taught)
7. Telescoping to ᾱ_t = C_{t+1} (not taught)
8. The index shift ᾱ_0 = 0.9994 and the 0.999 clip at t = 99 (not taught)

## The sampled-noise identity
9. The log of an isotropic 2D Gaussian (not taught)
10. The gradient of a squared distance (not taught)
11. Substitute the forward draw and cancel (not taught)

## The toy by hand
12. x_50 from x_0 and ε (not taught)
13. The score two ways, (−0.4153, 1.6613) (not taught)

## The marginal identity
14. Squared error is minimised by the conditional mean (not taught)
15. q_t as an average over clean points (not taught)
16. ∇q = q ∇log q under the integral (not taught)
17. Divide by q_t: the Bayes weights (not taught)
18. Pull out the constant (not taught)

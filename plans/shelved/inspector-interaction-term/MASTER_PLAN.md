# Inspector-Interaction-Term

> Shelved: an empty scaffold for background work (zero plan files). Comes back when its parent
> plan is promoted out of the background pool into the paper table; /populate-plans it then.

## Mission
Make the does-the-correction-cause-composition results driveable. The existing LoRA Inspector gains
tabs where dragging a slider does the experiment in front of you: dose up the
correction, slide the injection window, walk the manifold. The app displays
what the experiments produced and never generates anything itself.

## Objectives
1. Dose tab: λ scrub over the dose-sweep images, oracle and LoRA rows side by
   side, the three curves marking the current λ.
2. Window tab: scrub the injection window across the 50 steps for W1 and W2,
   both curves highlighting the current window, peak band shaded.
3. Manifold tab: the λ-walk animated on the CLIP axes, random-direction
   control path shown.
4. Sync: per-step norm and density curves update with whatever image the
   active tab shows.

## Goals
1. Each tab loads purely from /datasets result grids; a grep of the new tab
   code shows no generation code paths.
2. All four tabs drive real grids end to end (screen recording per tab).
3. Serves through the existing run_lora_inspector.sh + SSH tunnel flow with
   no new setup steps.

## Expected Outcome
The supervisor demo and the paper's interactive supplementary: every headline
figure explorable by hand, at zero additional generation cost.

## Definition of Done
1. Four tabs implemented and loading from finished grids only.
2. Screen recording of each tab driving real results.
3. Manifest/build script documented so a fresh session can rebuild the tab
   data from the grids.

## Sub-Scopes
(none)

## Plans
(to be populated)

## Environment Context
See `environment/00-INDEX.md` for this project's environment/architecture facts.
Read before drafting or checking any plan in this scope.

# Separation steering: retrospective

**The request.** Steer SDXL's plain PoE latent at steps 0 to 9 so the dog and cat experts' attention maps stop overlapping, and count whether that draws one dog and one cat on at least 4 of seeds 1 to 7.

**What was done.** Wrote `scripts/separation_steering.py` and its launcher, calibrated on seed 1, wrote the pass marks and step sizes into the results card, rendered 4 columns by 8 seeds on mscluster109 device 1, built the sheet on the launch node.

**Where the results live.** The [sheet](../../results/what-the-and-vector-is-made-of/separation-steering-sheet.png), its card entry in the README beside it, tiles and sidecars under `/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/separation_steering/`.

**Before and after.** Clean one-dog-one-cat tiles on seeds 1 to 7: 0 with no steering, 0 at each of three step sizes, 1 borderline tile. Seconds per tile: 14.8 plain, 18.6 to 18.7 steered.

**Why it came out this way.** Calibration shows the attention maps are nearly flat at steps 0 to 9 (largest value 0.0012 against 0.00098 flat), and the two "largest value" terms supply about 98% of the update gradient, so the step sharpens each map and barely separates them. Small steps changed little; larger ones destroyed the picture.

**Stop point.** Bar missed, 0 of 7 against 4 of 7. Way on: stop here.

**Dead ends and not done.** An overlap-only or reweighted loss was not run; the request fixed the loss.

**Family.** No threads or handoffs.

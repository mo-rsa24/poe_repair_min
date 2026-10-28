# Layer 12, step 2: measure the drain gap

⬅️ [Layer 12, step 2](../12-the-correction-at-scale.md#2-you-the-gate-first-the-drain-gap-measurement)

Paste into a session opened in `/home-mscluster/mmolefe/Playground/PhD/poe_repair_min-dog-and-rain`.

```text
/drip-execute-plan
~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/plans/01-measure-the-drain-gap.md
--depth "to 10.1"

Background
Layer 12 of the composition ladder (artifacts/drips/simple-distributions-rebuild/) asks whether a
correction for composition helps on chest X-rays. Its first step is a gate: whether a pneumothorax
classifier trained on NIH films finds drain films easier than drain-free ones. That measurement is
a plan in its own project, B1, whose folder is
~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/b1-drain-shortcut-in-pneumothorax/;
every relative path below is inside it. A walk on that plan is paused at round 2 of 5, and this
resumes it.

Objectives
1. Resume the paused walk on the plan from its walk map, in the plan's order, up to task 10.1.
2. Write src/train_fold.py and src/measure_gap.py in B1's folder.
3. Train one fold model per fold and write five held-out scores.csv files.
4. Run task 10.1: each group's hits over group size, the difference in points and its 95% interval.
5. Save the figure and its two CSVs, and keep the walk map and the plan's ticks current.

What to read
- artifacts/drips/drain-gap-walk/EXECUTE_WALK.md in B1's folder: the walk map, the resume point.
- plans/00-where-we-left-off.md in B1's folder: the pick-up page.
- plans/01-measure-the-drain-gap.md in B1's folder: the plan, with its bar under "What has to pass".
- ~/Playground/PhD/medical-imaging/compositional_diffusion/environment/cluster/nodes.md: the node
  table.
- environment/cluster/launch-rules.md in the compositional_diffusion project: the launch guards.

What is already known
- Code paths (rebuild/, anneal_samplers.py, tests/, notebooks/) are under toy/ in this repo, and
  python -m rebuild.<script> runs from toy/ (the guide, "Two lanes").
- The walk stands at round 2 of 5: tasks 1.1, 1.2, 3.1 to 3.3, 5.1 and 7.1 are ticked, instruction
  2.1 is confirmed and 4.2 skipped (the walk map; the pick-up page, "Walk 1").
- The walk waits on instructions 4.1 and 6.1, which are my count checks, and on my prediction of how
  the folds come out (the walk map; the pick-up page, "Walk 1").
- The label table data/pneumothorax_drain_table.csv holds 112,120 NIH films of 30,805 patients (plan,
  task 3.2).
- All 3,543 NEATX rows join, covering 1,011 patients with 0 dropped (plan, task 3.2).
- The groups are drain 1,637, no_drain 1,906, ptx_unlabelled 1,759 and negative 106,818 (plan, task
  3.2).
- data/folds.csv puts the 30,805 patients in 5 folds of 6,161, with 0 patients in two folds (plan,
  task 5.1; the pick-up page).
- The 112,120 image files at /datasets/mmolefe/data/nih/images/ have 0 missing and 0 empty (plan,
  task 7.1; the pick-up page).
- NEATX Drain_a1 and Drain_a2 are confidence grades (0, 0.25, 0.75, 1), not yes or no (the walk map).
- Two drain-less gallery films look like missed drains, 00026451_014 and 00012752_005 (the walk map).
- The classifier is DenseNet-121 from ImageNet weights at 224 px on all 14 NIH labels, with only the
  pneumothorax output scored, early stopping on 10% of the training patients, and one model per fold
  (plan, Description and Prerequisites; task 8.1).
- The cxr environment has torch 2.2.2 and torchvision 0.17.2 and no timm (plan, Prerequisites; task
  8.1).
- Held-out scores go to /datasets/mmolefe/shortcut_companion/runs/fold<k>/scores.csv, and the runs
  folder there does not exist yet (listed 2026-10-18).
- The plan names mscluster107, which has been down since 2026-10-11 15:41 and takes no run
  (the node table).
- mscluster106 was idle under my Slurm job 67958 on 2026-10-14, so read the node table and nvidia-smi
  before choosing a node (the node table).
- Every launch pins CUDA_VISIBLE_DEVICES, uses nohup and absolute paths, and its script checks df
  /datasets and aborts if the chosen card holds more than 1 GB (plan, Considerations and task 8.1;
  environment/cluster/launch-rules.md).
- Training time is not measured: task 8.2 times one epoch, and instruction 9.1 is my choice of 2 or
  5 folds on that number (plan, Expected runtime).
- Task 10.1 fixes the threshold at 90% specificity on each fold's held-out non-pneumothorax films,
  pools them, and takes a 95% interval from 1,000 patient resamples (plan, task 10.1).
- Task 10.1 prints each group's AUROC beside Oakden-Rayner's 0.940 (drain) and 0.770 (drain-less),
  and the age, view and drain side of both groups (plan, task 10.1).
- The bar, fixed before any run, calls it a gap when the difference is at least 5 points with the
  interval excluding 0 (plan, "What has to pass").
- Under 5 points, or an interval including 0 with an estimate under 5, or a reversed difference, is
  no gap, and the drain case ends (plan, "What has to pass").
- 5 points or more with an interval including 0 is inconclusive and does not continue (plan, "What
  has to pass").
- A difference above 15 points is checked for a label or split error first (plan, "What has to
  pass").
- Two estimates of the interval disagree: about ±2.6 points on the difference (plan, "Why folds")
  and about ±1 point per group (assumed, session b268bbd8), and the first full run settles it (the
  pick-up page).
- A gap is a necessary sign of a shortcut and not proof of one, because drains go into large
  pneumothoraces (plan, Considerations).
- The public TorchXRayVision model is not this measurement: it moves a median of +0.001 over 237
  same-patient pairs, with the drain film higher in 134 (the results card
  ~/Playground/PhD/medical-imaging/compositional_diffusion/b-shortcut-through-a-device/artifacts/results/does-the-classifier-look-at-the-drain/README.md).
- That public model puts 180 of 1,637 drain films against 514 of 1,906 drain-free films below its
  threshold, on films it may have trained on (the same results card).

Rules
- The bar never changes after a run.
- Never launch on a card with more than 1 GB in use or with another user's process on it.
- Instruction 11.1 is mine; stop there and let me record the verdict.
- No commit.
- Show each of these before moving on: task 8.1's 2-minute smoke run on 2,000 images writing one
  scores.csv row per held-out image; task 8.2's images per second and projected hours; task 8.3's
  count of five scores.csv files on the node that ran them; task 10.1's two sensitivities.

What to hand back
- src/train_fold.py and src/measure_gap.py in B1's folder.
- Five scores.csv files under /datasets/mmolefe/shortcut_companion/runs/.
- artifacts/results/does-the-classifier-use-the-drain/sensitivity-by-drain__nih-neatx.png in B1's
  folder, with its sidecar CSV and group-mix__age-view-side.csv.
- The walk map and the plan's ticks current.
- The verdict instruction 11.1 records (gap, no gap or inconclusive, with the difference and its
  interval), which step 3's walk and step 4's gate read.
- The proof: src/measure_gap.py prints both groups' hits over group size and the difference with its
  interval, and the figure and both CSVs exist.

What you will see
Along the way: one task per message, each stop above showing its printed line and asking for a yes.
At the end, a closing report: what was asked, in one sentence; what was done, with each output's
path; PASS or MISS on whether task 10.1 printed each group's hits over group size, the difference in
points and its 95% interval, with the verdict the bar gives (gap, no gap or inconclusive); what is
still open; at most one question for me.

I judge, in instruction 11.1: done when task 10.1 has printed both groups' hits, the difference in points and its 95% interval, and the bar above gives the verdict.
```

added: the seven sections, from the field-labelled version · the Background paragraph naming B1's folder and why the gate comes first (from this layer's "Why it is critical" line)
dropped: nothing

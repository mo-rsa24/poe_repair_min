# Composing a two-panel figure in Inkscape

For putting two grid panels side by side under one caption, and exporting a PDF that renders
cleanly in the ICLR template. Written for someone who has never opened Inkscape.

Worked example throughout: the two correction-window grids in
[the paper's figure folder](../../paper/iclr/figures/when-the-correction-arrives/poe/).

## Read this before you start

**The figure is built by a script, not by hand.**

```bash
export PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python \
    scripts/when_the_correction_arrives.py
```

That writes `paper/iclr/figures/when-the-correction-arrives/when-the-correction-arrives.pdf`,
5.50 x 2.80 inches. The paper includes it as a `[t]` float.

**What is on the figure, and what the caption carries instead.**

Row numbers appear once, against the left grid, and read across to the right one. The two grids
share that axis.

The column axis carries nothing at all, no numbers and no heading. The caption says the columns
are decodes taken after 10, 20, 30, 40 and 50 steps, which is cheaper than printing five numbers
and a title above the pictures.

**The two arrows are the panels' difference, stated without words.**

Down the left, up the right, both labelled `correction at step`. A lower row in (a) corrects more
of the run; a higher row in (b) starts earlier and so also corrects more of it.

**The upward arrow is saying which way coverage grows, not which way the numbers count.** The row
numbers increase downward on both sides. The panel captions, "applied early" against "applied
late", are what keep those two readings apart, so do not cut them.

**Panel (b)'s last row is labelled `off`, not `50`.**

It is the uncorrected product-of-experts baseline the rest are read against, and burying that
under a number costs more than the one extra label does. It is centred in the gap between the
grids so it attaches to neither by proximity.

**Resolution is set to the source and should not be raised.**

The decode frames are 512 x 512 on disk. At 0.454 inches a cell holds 512 pixels at about 1130
dpi, which is the script's default. Anything higher invents detail.

Drop to `--dpi 600` if the assembled paper runs into a size limit. On paper it is
indistinguishable; the difference only shows when someone zooms.

**One thing no resolution setting fixes.** The first two columns look soft because the latent
genuinely is soft that early in the run. That blur is the result, not an artifact.

**Layouts considered, with what each cost.**

| Layout | Cell | Figure height |
|---|---|---|
| Side by side, numbers on both grids | 0.474" | 2.98" |
| Side by side, numbers once, column axis kept | 0.472" | 3.19" |
| **Side by side, numbers once, no column axis** | **0.454"** | **2.80"** |
| Side by side, labels over the images | 0.538" | 2.96" |
| Stacked, one panel above the other | 0.655" | 7.50" |

Stacking gives the largest pictures by a wide margin and costs most of a page. It is the one to
revisit if a reader ever says they cannot check the claim by eye.

**Use Inkscape for what the script cannot draw.**

An arrow pointing at a cell. A circled region. A bracket spanning rows. A hand annotation.

**Check View > Display Mode is Normal before anything else.**

In Outline mode Inkscape draws every photograph as an empty box with a red cross through it. It
looks exactly like a file that failed to load, and it is not.

**Do not hand-place axis labels.**

That is two dozen text objects, each of which has to sit dead-centre on a grid row, and all of
which have to be redone if a seed or a cell ever changes.

Use Inkscape to decide a layout, then say what the numbers should be and put them in the script.
Inkscape is good at deciding. It is a bad place for a figure to permanently live.

**The rest of this recipe is the manual route.**

Sections 1 to 9 build a two-panel figure by hand, which is the sequence to follow when you want to
annotate something on top of a finished one.

## 1. Make the panels

Both grids are drawn by scripts, not by hand.

**Panel (a) is the window extending from the start.**

```bash
export PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
OUT=paper/iclr/figures/when-the-correction-arrives/poe

$PY scripts/longer_correction_grid.py \
    --bare --panel-width 2.60 --dpi 600 \
    --out-dir $OUT --name panel-a-window-extends-from-the-start
```

**Panel (b) is the window starting later.**

```bash
$PY scripts/later_start_grid.py \
    --bare --panel-width 2.60 --dpi 600 \
    --out-dir $OUT --name panel-b-window-starts-later
```

**What `--bare` does.**

It leaves out both shared axis titles: "picture decoded after step" on top, and the rotated
"corrected window" on the left.

Both panels use the same two axes, so each title only needs saying once. You add them yourself,
across the pair.

**What the panel keeps.**

Its own column numbers, 10 to 50, because the two panels sit at different places across the page.

Its own row labels, because (a) reads `0–10` to `0–50` and (b) reads `10–50` to `off`.

**What `--panel-width 2.60` does.**

It draws the panel at the exact width it will occupy on the page.

**Why that second flag matters more than it looks.**

If a panel is drawn 4.11 inches wide and then squeezed to 2.60 on the page, every label shrinks
with it. An 8pt label arrives at 5.1pt. That is too small to read and it is the most common way
a figure gets ruined.

Drawn at final size, 8pt stays 8pt.

**What `--dpi 600` does.**

It sets the resolution the sample thumbnails are embedded at.

Leave it out and the PDF falls back to matplotlib's default of 100 dpi, which shrinks every
thumbnail to 44 x 44 pixels. At 600 they are 260 x 260, which is 594 dpi on the printed page.

**Check the printed size.**

Each script prints `size 2.60 x 2.46 in`. If it says anything else, stop and fix it before going
further.

**Each run writes three files: `.png`, `.pdf` and `.svg`.**

**Take the SVG into Inkscape.** It is the only one Inkscape reads without argument.

Inside it the labels are vector, so they stay sharp at any zoom, and each thumbnail is a PNG
carried inline. Nothing has to be decoded out of a PDF image stream.

**The PNG and PDF are for LaTeX, not for Inkscape.** Section 10 uses the PDF directly if you
decide to skip Inkscape.

## 2. Get the files onto your laptop

Inkscape is not installed on the cluster. You do this part locally.

```bash
scp mmolefe@<cluster>:/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/paper/iclr/figures/when-the-correction-arrives/poe/panel-*.pdf .
```

---

## 3. Set the page size first, before anything else

Open Inkscape. You get a blank white page.

**Open the page settings.**

`File > Document Properties`, or press `Ctrl+Shift+D`.

**Set the units to inches.**

There is a units dropdown near the top of that panel. Change `mm` or `px` to `in`.

**Set width to 5.5 and height to 3.1.**

5.5 inches is the exact text width of the ICLR template. Your figure will fill it edge to edge.

**Do this before you import anything.**

If you build on a wrongly-sized page and fix it later, everything you placed has to move.

---

## 4. Bring the two panels in

**Import the first panel.**

`File > Import`, or press `Ctrl+I`. Choose `panel-a-window-extends-from-the-start.svg`.

**Take the SVG rather than the PDF.**

Both work. The SVG has one meaningful setting in its dialog, where the PDF has a choice of two
importers and a fonts dropdown to get right.

Inside the SVG the labels are vector and each thumbnail is a PNG carried inline, so there is
nothing to decode and nothing to link.

**A small SVG Input dialog appears. One setting matters.**

Set **SVG Image Import Type** to *Include SVG image as editable object(s) in the current file*.

The other choices wrap the whole panel in an image you cannot select into. Editable objects is
what lets you align pieces later.

**The DPI field does not apply.** It only affects the two "image tag" options, where Inkscape
rasterizes first.

**Leave the rest and click OK.**

**Repeat for panel (b).**

**Each panel arrives as a group.**

A group is a bundle of objects that moves as one. That is what you want.

Click once to select the whole group. Double-click to step inside and edit one piece. Press
`Escape` to step back out.

**The panel lands beside the page rather than on it.**

That is normal. Section 5 puts it where it belongs.

**The panel should measure 2.60 x 2.46 inches.**

The SVG declares `width="187.2pt"`, and 187.2/72 is 2.60. Check the W and H boxes. Section 5 sets
it explicitly anyway.

## 5. Place them by typing numbers, not by dragging

Dragging with a mouse will never give you two panels that line up. Type the coordinates.

**Switch to the Selector tool.**

Press `S`. The toolbar across the top now shows four boxes: X, Y, W, H.

**Set that toolbar's units to inches.**

There is a small units dropdown at the right end of those four boxes.

**Lock the aspect ratio.**

There is a small padlock icon between the W and H boxes. Click it so it looks closed.

This makes the height follow the width automatically. Without it, you will stretch a panel and
not notice.

**Place panel (a).**

Select it. Type `W = 2.60`, then `X = 0.20`, then `Y = 0.55`. Press Enter after each.

**Place panel (b).**

Select it. Type `W = 2.60`, then `X = 2.90`, then `Y = 0.55`.

**Where those numbers come from.**

The page is 5.50 inches wide. It is spent like this:

| Piece | Inches |
|---|---|
| Left strip for the shared rotated label | 0.20 |
| Panel (a) | 2.60 |
| Gap between the panels | 0.10 |
| Panel (b) | 2.60 |
| **Total** | **5.50** |

So (a) starts at 0.20 and (b) starts at 0.20 + 2.60 + 0.10 = 2.90.

The 0.55 at the top leaves room for the shared title you are about to add.

**If the panels jump around while you work, snapping is fighting you.**

Press `%` to switch snapping off.

## 6. Add the labels that both panels share

There are four pieces of text to add. Two are the shared axis titles, two are the panel letters.

**Switch to the text tool.**

Press `T`. Click where you want text. Type.

**Set the font before you get attached to it.**

In the top toolbar, pick a serif font and set the size to **8pt** for the axis titles.

Use **9pt bold** for the panel letters.

Anything under 7pt will not survive printing.

### The title across the top

**Type it.**

`picture decoded after step`

**Centre it across the whole page.**

Select the text. Open `Object > Align and Distribute`, or press `Ctrl+Shift+A`.

Set **Relative to: Page**. Click "Centre on vertical axis".

**Then set its height.**

With the Selector tool, set `Y = 0.30`.

It now sits above both panels and reads as belonging to both.

### The rotated label down the left

**Type it.**

`corrected window`

**Rotate it a quarter turn anticlockwise.**

`Object > Rotate 90° Counter-Clockwise`.

The text now reads bottom to top, which is the convention for a y axis.

**Centre it against the panels.**

Select the rotated text, then hold `Shift` and click panel (a) so both are selected.

In `Object > Align and Distribute`, set **Relative to: Last selected**, then click "Centre on
horizontal axis".

This lines the label up with the middle of the grid rather than the middle of the page.

**Then push it to the left edge.**

Set `X = 0.02`.

**One label, two panels, and that is deliberate.**

It sits beside panel (a), but it names the row axis of both, because both panels use the same
axis. The caption says so in words.

### The panel letters

**Type `(a)`** and place it at `X = 0.58`, `Y = 0.42`.

**Type `(b)`** and place it at `X = 3.28`, `Y = 0.42`.

**Why X is 0.58 and not 0.20.**

Each panel keeps a 0.38 inch gutter on its left for the row labels. 0.20 + 0.38 = 0.58, which is
the left edge of the actual grid.

The same sum for panel (b) is 2.90 + 0.38 = 3.28.

## 7. Tidy up before exporting

**Select everything.**

`Ctrl+A`.

**Shrink the page to fit what you drew.**

`Ctrl+Shift+R`.

**Check the width is still 5.5 inches.**

Reopen `Ctrl+Shift+D` and look. If the title overhangs, the page may have grown wider. Nudge the
title in and repeat.

**Remove leftover junk.**

`File > Clean Up Document`. This drops unused definitions and shrinks the file.

---

## 8. Export the PDF

**Save as a PDF.**

`File > Save As`. In the file-type dropdown choose **Portable Document Format (\*.pdf)**.

**A settings dialog appears. Four things matter.**

**Restrict to PDF version: 1.5.** Anything a reviewer opens will read it.

**Text output: choose "Convert text to paths".** This turns letters into shapes, so no viewer
can substitute a wrong font. You lose the ability to edit the text in the PDF, which does not
matter because you keep the `.svg`.

**Do not choose "Omit text in PDF and create LaTeX file".** That is a different workflow that
needs extra LaTeX setup.

**Rasterize filter effects: 600 dpi.** Only matters if you used blur, but it costs nothing.

**Output page size: "Use document page size".**

**Also save the `.svg`.**

`File > Save As` again, this time as Inkscape SVG. The SVG is your editable master. The PDF is
the output. Never throw the SVG away.

---

## 9. Put it in the paper

**Include it at full width with no scaling factor.**

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{when-the-correction-arrives/when-the-correction-arrives.pdf}
\caption{\textbf{Ten corrected steps already reach the outcome, and no late start recovers it.}
A cat and a dog at seed 12, same layout in both panels. Columns decode the latent after the
step shown, and the green frame marks decodes taken while the correction was active, not a
detector verdict. \textbf{(a)} Each row corrects from step 0 to the row's endpoint and leaves
the rest plain. Every row ends with two animals. \textbf{(b)} Each row is uncorrected until its
start step and corrected from there to step 50, and the last row never corrects. Starting at
step ten still yields two animals. Starting at step twenty or later never does. One pair at one
seed illustrates the timing rather than measuring it.}
\label{fig:when-the-correction-arrives}
\end{figure}
```

**`width=\textwidth` and nothing else.**

You built the page at 5.5 inches, which is exactly `\textwidth`. So this places it at 1:1 and
your 8pt labels land as 8pt.

**Never write `[width=1.2\textwidth]` or `[scale=0.8]`.** That undoes the whole point of
section 1.

---

## 10. The LaTeX-only route, if you skip Inkscape

The panels are already clean and already the right size, so LaTeX can do this unaided.

```latex
\begin{figure}[t]
\centering
\begin{minipage}[t]{0.49\textwidth}
\centering (a)\\
\includegraphics[width=\linewidth]{.../panel-a-window-extends-from-the-start.pdf}
\end{minipage}\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering (b)\\
\includegraphics[width=\linewidth]{.../panel-b-window-starts-later.pdf}
\end{minipage}
\caption{...as above...}
\end{figure}
```

**What you give up.**

The shared "picture decoded after step" title. You would put that wording in the caption
instead.

**What you gain.**

The figure rebuilds itself whenever the scripts rerun. Nothing has to be redone by hand.

---

## Things that will confuse you at first

**Y counts downward.**

`Y = 0` is the top of the page, not the bottom. Increasing Y moves things down.

**Selecting a group is not selecting the thing inside it.**

One click gets the group. Double-click enters it. `Escape` leaves it.

**Ctrl+Z has a shallow default.**

Undo history is limited. Save often, under versioned names.

**The green cell frames may look heavy at this size.**

They are 2.2pt, tuned for a panel 4.11 inches wide. On a 2.60 inch panel they read thicker.

To thin them, change `sp.set_linewidth(2.2 if corrected else 0.5)` in both scripts and rerun.
Try 1.4.

**Every photograph is a red box with a cross through it.**

`View > Display Mode` is set to Outline. Set it to Normal.

Outline mode draws images as empty crossed boxes on purpose, so it is not telling you the file
failed to load. Nothing about the file or the import settings is wrong.

**A PDF made from these panels will be a few megabytes.**

Each thumbnail is embedded at its full source resolution, which works out near 1400 dpi. That is
fine for a submission. If the whole paper gets heavy, downsample the thumbnails in the scripts
before drawing.

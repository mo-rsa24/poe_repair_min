# 🧭 Syncing the paper with Overleaf

The paper exists in three places and they do not update each other. This file is how you move
edits between them by hand: pulling what you typed in the Overleaf editor down into the repo,
sending a repo-side edit back up, and rebuilding the upload zip when the git route is not the
one you want. It does not cover writing the paper (that is `/drip-write` and
[the draft map](../paper/iclr/DRAFT_MAP.md)) or building the PDF locally, which needs the
Tectonic facts in [the paper environment file](../environment/paper.md), since no system LaTeX
exists on this cluster.

## Words this file uses

- **The repo copy**: `paper/iclr/`, the folder this repository tracks and commits. Everything
  else in the project links into it: figures, the draft map, the spine.
- **The Overleaf clone**: `paper/overleaf-iclr/`, a second, separate git repository whose
  `origin` is Overleaf itself. The outer repository does not track it.
- **The bundle**: `paper/iclr-overleaf-slim.zip`, a snapshot of the repo copy made for uploading
  into a fresh Overleaf project. The route that predates the clone.

## Table of contents

The recipes are numbered so the index can point at them. The numbers are addresses, not an order
to follow.

- [Before any of these](#before-any-of-these)
- [Where the paper lives](#where-the-paper-lives)
- [1. Pull your Overleaf edits into the repo](#1-pull-your-overleaf-edits-into-the-repo)
- [2. Read what came down before you commit it](#2-read-what-came-down-before-you-commit-it)
- [3. ⚠️ Send a repo-side edit back up to Overleaf](#3-️-send-a-repo-side-edit-back-up-to-overleaf)
- [4. Rebuild the upload zip instead](#4-rebuild-the-upload-zip-instead)
- [If something looks wrong](#if-something-looks-wrong)
- [Where this came from](#where-this-came-from)

## Before any of these

Navigation: ⬅️ [Words this file uses](#words-this-file-uses) | 📋 [TOC](#table-of-contents) | [Next](#where-the-paper-lives) ➡️

- [ ] 🖥️ **The Overleaf clone is there and points at the right project.**
  ```bash
  git -C paper/overleaf-iclr remote -v
  ```
  ✅ **`origin  https://git@git.overleaf.com/6a86dc92ca15fe47d75b86f6`** for fetch and push. That
  hash is the Overleaf project id, the same one in the editor's URL.
  ❌ **`not a git repository`**: the clone is gone. Recreate it with
  `git clone https://git@git.overleaf.com/6a86dc92ca15fe47d75b86f6 paper/overleaf-iclr`, which
  asks for the Overleaf git token as the password.

- [ ] 🖥️ **Your Overleaf token is stored, so no pull prompts for a password.**
  ```bash
  git -C paper/overleaf-iclr config --get credential.helper && grep -c overleaf ~/.git-credentials
  ```
  ✅ **`store`** then **`1`**. The token sits in `~/.git-credentials`, mode 600.
  ❌ **`0`**: the next pull will prompt. The password is an Overleaf git token from Account
  Settings, not your account password.

## Where the paper lives

Navigation: ⬅️ [Before any of these](#before-any-of-these) | 📋 [TOC](#table-of-contents) | [Next](#1-pull-your-overleaf-edits-into-the-repo) ➡️

| Place | What is in it | Who writes it | Watch out |
|---|---|---|---|
| `paper/iclr/` | the tracked copy: tex, bib, every figure, the writing notes | this repo, and you through Claude | a pull into the clone does not reach it; §1 copies across |
| `paper/overleaf-iclr/` | tex, bib, the style files, only the figures the tex includes | Overleaf, when you type in the editor | its working tree is dirty in a deleting direction; see §3 |
| `paper/iclr-overleaf-slim.zip` | a snapshot for uploading a new project | `scripts/make_overleaf_bundle.sh --slim` | stale the moment either side moves |

The clone holds a subset on purpose: it has no `DRAFT_MAP.md`, no `SPINE.md`, no reference PDFs,
and only the figures the manuscript actually includes.

## 1. Pull your Overleaf edits into the repo   `ran 2026-09-21`

Navigation: ⬅️ [Where the paper lives](#where-the-paper-lives) | 📋 [TOC](#table-of-contents) | [Next](#2-read-what-came-down-before-you-commit-it) ➡️

**When you need this**

You wrote in the Overleaf editor and want that text in the repo, where the figures and the draft
map live.

```bash
cd /home-mscluster/mmolefe/Playground/PhD/poe_repair_min
git -C paper/overleaf-iclr pull
```

✅ **A fast-forward naming the files Overleaf changed:**

```
From https://git.overleaf.com/6a86dc92ca15fe47d75b86f6
   d512a90..e793c5e  main       -> origin/main
Updating d512a90..e793c5e
Fast-forward
 iclr2027_conference.tex | 42 ++++++++++++++++++++++++++----------------
 1 file changed, 26 insertions(+), 16 deletions(-)
```

Overleaf writes every editor session as one commit called `Update on Overleaf.`, so the count of
commits tells you how many sittings came down, never how many edits.

✅ **`Already up to date.`** means nothing has been typed in the editor since your last pull.

Then copy what came down into the repo copy, file by file, only for the files the pull actually
touched:

```bash
cp paper/overleaf-iclr/iclr2027_conference.tex paper/iclr/iclr2027_conference.tex
git status --short paper/iclr/
```

✅ **` M paper/iclr/iclr2027_conference.tex`**, ready for you to read the diff (§2) and commit.

**Check the repo copy was clean before you overwrote it.** Run `git status --short
paper/iclr/iclr2027_conference.tex` first: an empty answer means nobody edited it here since the
last commit, so the copy loses nothing. If it comes back ` M`, both sides moved and you have a
real merge on your hands, so diff them and resolve by hand instead of copying.

**Variations**

- Just look, do not merge: `git -C paper/overleaf-iclr fetch && git -C paper/overleaf-iclr log --oneline HEAD..origin/main`
- Refuse anything but a clean fast-forward: `git -C paper/overleaf-iclr merge --ff-only origin/main` after that fetch

## 2. Read what came down before you commit it   `ran 2026-09-21`

Navigation: ⬅️ [1. Pull your Overleaf edits into the repo](#1-pull-your-overleaf-edits-into-the-repo) | 📋 [TOC](#table-of-contents) | [Next](#3-️-send-a-repo-side-edit-back-up-to-overleaf) ➡️

**When you need this**

The pull landed and you want to know what you actually changed in the editor, because an Overleaf
commit message never says.

```bash
git -C paper/overleaf-iclr show --stat HEAD
git -C paper/overleaf-iclr show HEAD -- iclr2027_conference.tex
```

✅ **The diff, one hunk per place you typed.** Overleaf commits whole editor sessions, so expect
unrelated edits in one commit: on 2026-09-21 a single commit carried a new title, a rewritten
abstract with the old one commented out beneath it, a `wrapfig` float move, three `\looseness=-1`
lines and two paragraph moves.

**Variations**

- How far apart the two copies are right now: `diff -u paper/iclr/iclr2027_conference.tex paper/overleaf-iclr/iclr2027_conference.tex`
- Which hunks differ, without the prose: append `| grep '^@@'` to that

## 3. ⚠️ Send a repo-side edit back up to Overleaf   `unverified`

Navigation: ⬅️ [2. Read what came down before you commit it](#2-read-what-came-down-before-you-commit-it) | 📋 [TOC](#table-of-contents) | [Next](#4-rebuild-the-upload-zip-instead) ➡️

**When you need this**

A figure or a paragraph changed in the repo and you want it visible in the Overleaf editor.

⚠️ **The clone's working tree deletes files if you commit it wholesale.** As of 2026-09-21 it has
`DRAFT_MAP.md`, `OVERLEAF.md`, `README.md`, `SPINE.md`, `figures.md`, three abstract PDFs and
eleven figures deleted locally while they still exist on Overleaf. A `git add -A` here removes
all of them from the project.

Read the damage before you write anything:

```bash
git -C paper/overleaf-iclr status --short
```

Then stage only the paths you mean to send, never `-A` and never `.`:

```bash
cp paper/iclr/iclr2027_conference.tex paper/overleaf-iclr/iclr2027_conference.tex
cp paper/iclr/figures/<the figure the tex includes> paper/overleaf-iclr/figures/<same path>
git -C paper/overleaf-iclr add iclr2027_conference.tex figures/<same path>
git -C paper/overleaf-iclr commit -m "<what changed, in the repo's voice>"
git -C paper/overleaf-iclr push
```

✅ **The push reports `main -> main`**, and refreshing the Overleaf editor shows the new text.

❌ **`! [rejected] main -> main (fetch first)`**: someone typed in the editor since your last
pull. Run §1 first, then push. Never force-push; that erases the editor's history.

**Every figure the tex names has to exist in the clone**, or Overleaf fails the compile with a
missing-file error that appears nowhere locally. `grep includegraphics paper/overleaf-iclr/iclr2027_conference.tex`
lists what it needs.

## 4. Rebuild the upload zip instead   `unverified`

Navigation: ⬅️ [3. ⚠️ Send a repo-side edit back up to Overleaf](#3-️-send-a-repo-side-edit-back-up-to-overleaf) | 📋 [TOC](#table-of-contents) | [Next](#if-something-looks-wrong) ➡️

**When you need this**

You are starting a fresh Overleaf project, or the clone has drifted so far that replacing the
project is cheaper than reconciling it.

```bash
scripts/make_overleaf_bundle.sh --slim
```

✅ **`paper/iclr-overleaf-slim.zip`, and a printed list of the figures it kept.** The slim flag
reads the `\includegraphics` calls out of the tex and packs only those figures, because the full
bundle carries every working draft under `figures/` and has grown past what Overleaf accepts on
upload.

The project settings the bundle expects (main file, compiler, the commented-out bibliography
lines) are in [the Overleaf handoff note](../paper/iclr/OVERLEAF.md).

## If something looks wrong

Navigation: ⬅️ [4. Rebuild the upload zip instead](#4-rebuild-the-upload-zip-instead) | 📋 [TOC](#table-of-contents) | [Next](#where-this-came-from) ➡️

**A pull asks for a username and password.** The stored token has expired or was never written.
Generate a git token in Overleaf's Account Settings, then pull again and paste it as the
password with `git` as the username; the `store` helper keeps it.

**The pull is not a fast-forward.** Something was committed in the clone as well as in the
editor. `git -C paper/overleaf-iclr log --oneline origin/main..HEAD` shows what is local; decide
per commit rather than merging blind.

**The repo copy and the clone disagree on a file you did not touch.** They are separate
repositories with no automatic mirroring, so this is the normal state after either side moves.
§1 for down, §3 for up.

**Overleaf compiles fine but the repo's build fails, or the reverse.** The two use different
engines. Overleaf's is pdfLaTeX or XeLaTeX per the project setting; here it is Tectonic, per
[the paper environment file](../environment/paper.md).

## Where this came from

Navigation: ⬅️ [If something looks wrong](#if-something-looks-wrong) | 📋 [TOC](#table-of-contents)

| What | How it was established | When |
|---|---|---|
| The clone's remote and project id | `git -C paper/overleaf-iclr remote -v` | 2026-09-21 |
| §1 and §2, with their real output | Pulled commit `e793c5e` down and copied the tex into the repo copy | 2026-09-21 |
| The deleting-direction warning in §3 | `git status --short` in the clone: 18 deletions and 9 untracked files | 2026-09-21 |
| Overleaf names every editor session `Update on Overleaf.` | The clone's own log | 2026-09-21 |
| §3's push and §4's bundle | Transcribed, not run this sitting: §3 from the clone's remote, §4 from the handoff note | 2026-09-21 |

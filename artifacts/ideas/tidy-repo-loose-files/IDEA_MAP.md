# 💡 tidy-repo finds and files loose files

The idea targets the skill at `~/.claude/skills/tidy-repo/SKILL.md`. The map lives here because
this repo is where the drift it describes is visible.

## Position in the idea

| Claim | Mark | Settled by |
|---|---|---|
| 1. tidy-repo does not know what the repo is meant to look like | wrong as stated, settled | Step 1 reads the conventions live every run; the repair is one new check, a top-level census feeding the existing triage |
| 2. the staple set is context/, environment/, runbook/, plans/, report/ | wrong as stated, settled | schema assembled at run time (global four + data/ + repo CLAUDE.md override); tooling folders exempt; this repo's CLAUDE.md now declares context/ and report/ |
| 3. files land loose with spurious names, often enough to matter | holds | this repo's root today: temp/, tmp/, output/, docs/, a root ideas/ beside artifacts/ideas/, paper/iclr/meeting-transcript.txt |
| 4. opening and inspecting a loose file determines where it belongs | holds with a boundary, settled | tested on this repo's 8 real loose entries: 6 decided with quotable evidence, 2 became one-time policy questions; never-guess survives as propose-with-evidence |
| 5. tidy-repo is the right owner of the move | holds, settled | no sibling claims stray-file sweeps; boundary recorded: tidy-repo moves files into pulse-owned folders and stops, content stays the pulse's job |
| 6. the skill runs from a roster of repos, not one repo at a time | holds, settled | a declared REPOS.md in ~/.claude, one report and one approval pass per repo; discovery-by-scanning rejected |

Load-bearing: claim 4. If inspection cannot determine a destination, the feature is a census that
asks about everything, which already exists in spirit.

## Table of contents

- [Position in the idea](#position-in-the-idea)
- [Quick context: where you are](#quick-context-where-you-are)
- [The idea, as it stands](#the-idea-as-it-stands)
- [The claims](#the-claims)
- [What the words are](#what-the-words-are)
- [Held claims](#held-claims)
- [Dead ends](#dead-ends)
- [Checks outstanding](#checks-outstanding)
- [Runs](#runs)
- [Sources](#sources)
- [Next step](#next-step)

## Quick context: where you are

Navigation: ⬅️ [Position](#position-in-the-idea) | 📋 [TOC](#table-of-contents) | [Next](#the-idea-as-it-stands) ➡️

**What the idea is**

tidy-repo gains a check that accounts for every top-level entry in a repo, opens what is
unaccounted for, and proposes a destination in the staple folders with evidence.

**Where the walk is**

Compiled, Destination A. The walk is closed; the route out is a skill-creator edit of
`~/.claude/skills/tidy-repo/SKILL.md`.

**What compile would produce today**

Destination A, full: every claim settled, the load-bearing one tested live on this repo's own
loose files. The build is two additions to tidy-repo (the census check, the roster mode) plus
the boundary notes.

## The idea, as it stands

Navigation: ⬅️ [Quick context](#quick-context-where-you-are) | 📋 [TOC](#table-of-contents) | [Next](#the-claims) ➡️

tidy-repo already walks a repo against the artifact and plan conventions and moves files on
approval. The idea adds a census check: every top-level entry is either a staple folder the
repo's own CLAUDE.md names, a file on the root allowlist, or a finding. For each finding the
skill opens the file, gathers the evidence that says what it is (what wrote it, what names it,
what it contains), and proposes a destination under the staples, per cluster, user approving.
Loose files with spurious names from Claude sessions are the main population it catches.

## The claims

Navigation: ⬅️ [The idea](#the-idea-as-it-stands) | 📋 [TOC](#table-of-contents) | [Next](#what-the-words-are) ➡️

### 1. ❌ tidy-repo does not know what the repo is meant to look like

Mark: wrong as stated, settled. Step 1 of the skill reads `ARTIFACT_TREE_FORMAT.md`,
`PLAN_TREE_FORMAT.md`, the project `CLAUDE.md`, and the root-file allowlist live at every run.
What it lacks is coverage, not knowledge: its checks are anchored to `artifacts/` and `plans/`,
so a stray `temp/` or `output/` at the root falls through every check. Repaired wording, agreed:
the skill knows the rules but has no check that accounts for every top-level entry. The work is
one new census check that feeds findings into the existing triage-and-move machinery.

### 2. ❌ The staple set is context/, environment/, runbook/, plans/, report/

Mark: needs a check. The global convention names plans/, artifacts/, runbook/, environment/
plus data/, and lets each repo override in its own CLAUDE.md. context/ and report/ are staples
of this repo, not of every repo. The check: confirm the skill should read the staple list from
the repo's CLAUDE.md at run time, never carry one of its own. Split on request into three
pieces:

- [x] 2.1 the three sources disagree on what the staples are
- [x] 2.2 the repair: the schema is assembled at run time, never stored
- [ ] 2.3 the two alternatives considered and rejected   ← current

### 6. ✅ The skill runs from a roster of repos, not one repo at a time

Mark: holds, settled. A declared roster (`~/.claude/REPOS.md`, one path per line with a word on
what each repo is) that one invocation sweeps, producing one report and one approval pass per
repo. The roster changes where the skill runs, never how: no cross-repo batch approvals.
Discovery-by-scanning rejected because discovery guesses where a roster declares. Practical
limit carried into the design: a session sweeps only repos its permissions reach; otherwise the
roster acts as the checklist for per-repo runs.

### 3. ✅ Files land loose with spurious names, often enough to matter

Mark: holds. This repo's root today: `temp/`, `tmp/`, `output/`, `docs/`, `logs/`, a root
`ideas/` sitting beside the correct `artifacts/ideas/`, and `paper/iclr/meeting-transcript.txt`.
No search was needed, one `ls` was.

### 4. ✅ Opening and inspecting a loose file determines where it belongs

Mark: holds with a boundary, settled. Load-bearing. Tested live on this repo's loose entries:
root ideas/, temp/, tmp/, output/, docs/, logs/, temp/codex-drop/, the meeting transcript. Six
of eight destinations were determined with quotable evidence; logs policy and the transcript's
home remained one-time questions whose answers get written into CLAUDE.md. The never-guess rule
survives because deciding means proposing with quoted evidence, never silently moving.

### 5. ✅ tidy-repo is the right owner of the move

Mark: holds, settled. No sibling claims stray-file detection: the pulses audit what is inside
their folders, sync-plan-tree owns plan moves only, retrofit-repo performs no check a member
owns. Precedent for the shape exists in tidy-repo's own ENVIRONMENT.md migration. Boundary
recorded: when a destination is a pulse-owned folder, tidy-repo moves the file and stops;
judging the content stays the pulse's job.

## What the words are

Navigation: ⬅️ [The claims](#the-claims) | 📋 [TOC](#table-of-contents) | [Next](#held-claims) ➡️

| My phrase | The field's name | What it means | Confidence |
|---|---|---|---|
| staple folders | the four-folders rule | the fixed top-level set the global CLAUDE.md defines, overridable once per repo | confident, own convention |
| files out of place | drift | the repo disagreeing with its stated shape; tidy-repo's own opening term | confident, own convention |
| look through, open, inspect, determine | triage with quoted evidence | the verdict-per-artifact pattern in tidy-repo check 2a | confident, own convention |
| everything accounted for | a census / allowlist check | every entry is either expected, allowlisted, or a finding | confident, generic term |

## Held claims

Navigation: ⬅️ [What the words are](#what-the-words-are) | 📋 [TOC](#table-of-contents) | [Next](#dead-ends) ➡️

| Claim | What is unresolved | What would settle it |
|---|---|---|

## Dead ends

Navigation: ⬅️ [Held claims](#held-claims) | 📋 [TOC](#table-of-contents) | [Next](#checks-outstanding) ➡️

| Claim | The workaround | Why it failed |
|---|---|---|

## Checks outstanding

Navigation: ⬅️ [Dead ends](#dead-ends) | 📋 [TOC](#table-of-contents) | [Next](#runs) ➡️

| Claim | The check | What each outcome means |
|---|---|---|
| 2 | read this repo's CLAUDE.md and the global one for where the staple list lives | per-repo list confirmed means the skill reads it at run time; no list anywhere means the census has no schema and claim 2 needs a repair |

## Runs

Navigation: ⬅️ [Checks outstanding](#checks-outstanding) | 📋 [TOC](#table-of-contents) | [Next](#sources) ➡️

| # | Anchor | What it executed | State | Finding |
|---|---|---|---|---|
| 1 | claim 4 | file the decided loose entries | done | root ideas/ moved into artifacts/ideas/ (one git mv, one mv), 5 references rewritten, empty tmp/ removed, RENAMES.md appended; verification grep clean. Parked with reasons: output/imagegen (no referrer names the two PNGs, grouping unknowable without guessing), docs/IMMERSE_PoE_Foundations.md (tracked, referenced by DRAFT_MAP.md and a plan, and two candidate journeys exist: poe-derivation-foundations vs poe-composition-diffusion), temp/a_frog__x__a_toad (754M data move to /datasets needs its own confirmation), logs/ and temp/codex-drop (keep-here vs /datasets policy unstated), the meeting transcript (judgment call) |
| 2 | claim 4 | tidy-repo census pass after the 30ab15e sync commit | done | cat and dog.jpg deleted (untracked, no referrer; had been a codex render style reference); temp/a_frog__x__a_toad deleted after md5 verification that it duplicated the canonical /datasets window-sweep data; caches and rungs accepted in place under artifacts/ by decision; scene/ build output (3.5G) kept by decision; CLAUDE.md declarations, third-party PDFs and the 11 missing cards skipped this round |

## Sources

Navigation: ⬅️ [Runs](#runs) | 📋 [TOC](#table-of-contents) | [Next](#next-step) ➡️

| Source | What it gives the idea | Confidence |
|---|---|---|
| `~/.claude/skills/tidy-repo/SKILL.md` | the seven existing checks, the never-guess rule, the propose-with-evidence pattern | read this session |
| `~/.claude/CLAUDE.md`, "Where things live in a repo" | the four-folder rule, the root-file allowlist, the per-repo override | read this session |
| this repo's root listing | the live population of loose entries | run this session |

## Next step

Navigation: ⬅️ [Sources](#sources) | 📋 [TOC](#table-of-contents)

Compiled 2026-08-27, Destination A, delivered in chat. Build steps: (1) census check added to
tidy-repo as check 7, schema assembled per repo, tooling-owned entries exempt, findings feed the
existing triage; (2) undeclared-staple finding whose fix is the CLAUDE.md override lines,
drafted by the skill; (3) roster mode reading ~/.claude/REPOS.md, one report and approval pass
per repo. Still open after run 2: the IMMERSE doc's journey, the imagegen grouping, the logs
and codex-drop policy, the meeting files' home.

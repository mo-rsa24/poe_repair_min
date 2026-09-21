# Should tidy-repo gain a check that accounts for every top-level entry in a repo, not just artifacts and plans?

`tidy-repo` already walks a repo against the artifact and plan conventions and moves files on
approval, but its checks anchor to `artifacts/` and `plans/`, so a stray `temp/`, `output/`, or
`docs/` at the root falls through every check. The idea adds a census: every top-level entry is
either a staple folder the repo's own `CLAUDE.md` names, a file on the root allowlist, or a
finding to open and file with evidence. The walk is compiled and closed, tested live on this
repo's own eight loose entries (root `ideas/`, `temp/`, `tmp/`, `output/`, `docs/`, `logs/`,
`temp/codex-drop/`, a meeting transcript), six of which were filed with quotable evidence.

## What is in here

One file, `IDEA_MAP.md`, the full `/drip-idea` walk: six claims, all settled, two runs recording
what got filed in this repo (a root `ideas/` folder merged into `artifacts/ideas/`, several
untracked files deleted after verification, others parked pending a policy decision), and the
build steps the compile step specified.

## Where it came from and what judged it

**Held since 2026-09-05.** The idea targets `~/.claude/skills/tidy-repo/SKILL.md`, a skill
definition outside this repository, so no plan or document inside this repo would ever cite it as
a referrer even though its two runs already changed this repo's root. Kept as the design record
for the census check (added as tidy-repo's check 7) and the roster mode (`~/.claude/REPOS.md`)
until a skill-creator edit lands them in the skill itself; four items stay open per the map's own
"Next step": the IMMERSE document's journey, the `imagegen` grouping, the `logs`/`codex-drop`
retention policy, and the meeting transcript's home.

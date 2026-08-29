# 💡 Codex diagram render queue

## Position in the idea

| Layer | Mark | Settled by |
|---|---|---|
| 1. what happens when you press render | decided | the trip ran end to end on prompt 2a today; nothing needed inventing |
| 2. what you see: states and the flip gate | decided | four states (planned, rendering, stuck, built); flip is automatic after pixel check plus vision read, veto-after at the gate re-queues with the objection as the correction |
| 3. the pieces and who talks to whom | decided | one worker skill plus one new DIAGRAM_PROMPTS_FORMAT section; producers change by zero lines; frame-hypothesis feeds in downstream via hypothesis-to-scope; the only network boundary is codex-to-OpenAI |
| 4. the worker's contract | decided | serial drain with the door open to parallel (the [rendering] mark already carries the session id); retry cap 3; the dispatch template always carries the save-path line, learned from the run that omitted it |
| 5. what holds the state | decided | all state in files the conventions own; marks grow to four ([rendering] carries the session id, [stuck] the failed check); attempts in temp/codex-drop/, only passes reach the Save-as path |
| 6. what you build first | decided | four thin slices: worker skill (proof target: this scope's own remaining prompts), format section, sync sweep, documentation producer |

## Quick context: where you are

**What the idea is**

Every planning skill that emits diagram prompts should feed one background queue: Codex renders
each prompt in order while the session keeps working, each render is checked against its
faithfulness note, revised until it passes, sized for where it will be read, saved at its
`Save as:` path, and its mark flipped to [built].

**Where the walk is**

All six layers decided. Compiled. The route out is a skill-creator prompt for slice 1 and 2,
emitted in the walk's closing round.

## Constraints pinned from the repo

**The contract is one file.** `~/.claude/DIAGRAM_PROMPTS_FORMAT.md` line 3: init-master-plan seeds
diagram-prompts.md, populate-plans and decompose-plan extend it, sync-plan-tree maintains it,
prompt-storyboard extends on request. "Change this file in one place; the skills follow." So the
integration point is that document, not four skill edits.

**frame-hypothesis is not a producer of this file.** It compiles a cascading figure plan, a
different artifact. Its inclusion needs its own decision.

**The render loop is proven, today, in this session.** `codex exec --sandbox workspace-write` in a
background task, `codex exec resume --last -c 'sandbox_mode="workspace-write"'` for revisions,
pixel-scan checks against the faithfulness note. Prompt 2a took one render plus two revisions,
about 250k Codex tokens, minutes per round.

**The queue state already exists as file state.** [planned]/[built] marks, `Save as:` paths, and a
"N prompts · N rendered · N waiting" counter are all in the format. What is missing is the worker
that drains it.

**One maintainer, no daemon.** Work runs inside a Claude session (background tasks) or via nohup;
nothing persistent is running on this node.

## Appended extension: documentation lanes

The same contract, pointed at the documentation folders. context/, environment/ and report/ can
each carry a diagram-prompts.md of the standard shape; the precedent already exists, since the
format names runbook/diagrams/ (owned by runbook-pulse) as using this contract's file shape and
marks. The worker needs no mode for this: its input is any file of the contract shape, whatever
folder it sits in. What is genuinely new is a producer skill that reads the docs (plans, the
repo, artifacts, runbook, environment, report), picks the concepts worth a picture, and authors
subject and process prompts for them, with real reference images for physical things. That is a
sibling of prompt-storyboard, not a mode of the worker, and it is sequenced after the worker
exists, because it is a customer of the queue, not part of it.

## Held layers

None.

## Early decisions forming, pinned before their layers are walked

**Dispatch timing (belongs to layer 4).** Render at write time, before commit. The format doc's
own reason: the pictures exist to serve the approval gate, so they must exist at the gate. A
prompt modified during the gate re-queues; only the affected prompt re-renders.

**Failure handling (belongs to layer 4).** Per prompt: render, check against the faithfulness
note, bounded revisions in the same Codex session, pass flips the mark, exhausted retries mark
the prompt stuck for the user's eye, then the next prompt either way.

**Embedding on landing (belongs to layer 5, and it is already conventioned).** Documents that
want a not-yet-rendered picture carry a slot block (the format's rule against broken embeds). The
worker's final step per prompt: replace every slot naming this file with the sized embed, per
PLAN_TEMPLATE_SKELETON.md's table (display width from aspect, height capped near 500 px, image
wrapped in a self-link, caption underneath, file never resized). "Right place" is wherever a slot
already points at the file. Dispatch inherits a duty: the prompt names its ratio, landscape 16:9
or 3:2 by default, and Codex renders at the largest size.

**Three entry points, one queue (belongs to layer 3).** Producers (init-master-plan,
populate-plans, decompose-plan, prompt-storyboard) dispatch eagerly at write time.
sync-plan-tree, which already reconciles the illustrated map every sync, gains the sweeper role:
find [planned] prompts and unreplaced slots anywhere in the tree and start the drain. A manual
"render this scope" ask is the third. All three drain the same file-state queue.

**A slot with no prompt behind it is a finding, not a render (belongs to layer 3).** The queue's
unit is the prompt entry. A "Diagram wanted" slot naming no prompt routes to prompt-storyboard to
write one under the scope's art direction first; rendering from a slot's one-liner would bypass
the pinned vocabulary.

**In-flight marking is now required (belongs to layer 5).** With eager producers and a sweeping
sync both live, a prompt mid-render must be visible as such, or sync double-dispatches it. Either
a third mark ([rendering], with the Codex session id) or drop-folder evidence. Undecided.

**Reference images join the dispatch payload (belongs to layer 4).** The format's Reference
images section already grounds glyphs in real files under images/, never invented. The worker
passes those files to Codex alongside the prompt. Settled by a run: `codex exec -i <file>` with
the prompt on stdin reproduced a reference photo's species, breeds, coloring, poses and layout in
the infographic style (`temp/codex-drop/ref-test.png` against `cat and dog.jpg`). Dispatch shape:
images via `-i`, prompt always on stdin, since `-i` is variadic and eats a positional prompt.
Fetching
a real photo of a physical thing off the web is a capture step per CONTEXT_FORMAT (saved under
images/ first), not something the render worker does mid-queue.

**Cross-skill continuity (belongs to layer 5).** The queue is the marks in diagram-prompts.md,
not the session's memory. A drain started under init-master-plan can be harvested and continued
by whichever session is active when decompose-plan or populate-plans runs next.

## Dead ends

None yet.

## Route results folded back

Slices 1 and 2 are built: `~/.claude/skills/render-diagrams/` (SKILL.md plus
references/dispatch-and-checks.md) and the Rendering section in
`~/.claude/DIAGRAM_PROMPTS_FORMAT.md` (four-mark vocabulary, dispatch timing, veto-after flip,
embed-on-pass, plus a render-diagrams row in Who writes what).

## Next step

Run the skill's first drain on plans/closing-the-compositional-gap/diagram-prompts.md (slice 1's
proof target), then build slice 3 (the sync sweep line) and slice 4 (the documentation producer).

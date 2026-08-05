"""Which token indices name the subject, for any pair's prompts.

`value_probe.py` used to hardcode `{"cat": {"branch": 0, "tok": 2},
"dog": {"branch": 1, "tok": 2}}`: index 2 is the subject in "a cat" and
"a dog", after the start token and the article. That assumption breaks on the
pool this scope actually sweeps, because CLIP's tokenizer splits some animals
into several pieces:

    a cat          [<s>, a, cat, </s>]                  subject = [2]
    an eagle       [<s>, an, eagle, </s>]               subject = [2]
    a walrus       [<s>, a, wal, rus, </s>]             subject = [2, 3]
    a chimpanzee   [<s>, a, chim, pan, zee, </s>]       subject = [2, 3, 4]

Index 2 on "a walrus" probes the fragment "wal", which is not the word and
would have produced a map nobody could tell was wrong. `a_seal__x__a_walrus`
is one of the six transfer pairs, so the 64-cell sweep would have hit it.

So the subject is a *set* of indices, not one, and the probe averages over
them. For single-token animals this reduces exactly to the old behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectTokens:
    """Where the subject sits in one solo prompt's token sequence."""

    prompt: str
    indices: list[int]
    pieces: list[str]

    @property
    def is_split(self) -> bool:
        """True when the tokenizer broke the subject into several pieces."""
        return len(self.indices) > 1

    def describe(self) -> str:
        joined = "+".join(self.pieces)
        return (f"{self.prompt!r} -> tokens {self.indices} = {joined}"
                + ("  (split)" if self.is_split else ""))


def subject_tokens(prompt: str, tokenizer) -> SubjectTokens:
    """Indices of the subject words in a solo prompt like "a walrus".

    Everything between the leading article and the end token is the subject.
    Raises rather than guessing if the prompt has no content tokens, because a
    silently wrong token map is the failure this module exists to prevent.
    """
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)

    start = 1 if (bos is not None and ids and ids[0] == bos) else 0
    end = ids.index(eos) if (eos is not None and eos in ids) else len(ids)

    # Skip a leading article. Only "a" and "an": dropping anything else would
    # silently trim a real subject word.
    articles = {"a", "an"}
    while start < end:
        piece = tokenizer.decode([ids[start]]).strip().lower()
        if piece in articles:
            start += 1
            continue
        break

    indices = list(range(start, end))
    if not indices:
        raise ValueError(
            f"no subject tokens in {prompt!r} (ids={ids}). The probe needs at "
            "least one content token to read."
        )
    pieces = [tokenizer.decode([ids[i]]).strip() for i in indices]
    return SubjectTokens(prompt=prompt, indices=indices, pieces=pieces)


def token_map_for_pair(prompt_a: str, prompt_b: str, tokenizer) -> dict:
    """The probe's token map for one pair: branch 0 is A, branch 1 is B.

    Shape matches the old hardcoded TOKENS dict, except ``tokens`` is a list
    instead of a single ``tok``, so a multi-piece subject is averaged rather
    than truncated to its first fragment.
    """
    a = subject_tokens(prompt_a, tokenizer)
    b = subject_tokens(prompt_b, tokenizer)
    name_a = "".join(a.pieces) or "a"
    name_b = "".join(b.pieces) or "b"
    if name_a == name_b:                      # e.g. a pair with itself
        name_a, name_b = f"{name_a}_A", f"{name_b}_B"
    return {
        name_a: {"branch": 0, "tokens": a.indices, "pieces": a.pieces,
                 "prompt": prompt_a},
        name_b: {"branch": 1, "tokens": b.indices, "pieces": b.pieces,
                 "prompt": prompt_b},
    }


def verify_token_map(token_map: dict, tokenizer) -> list[str]:
    """Re-derive every entry from the tokenizer. Returns a list of problems.

    The plan's smoke test asks that the token map be "verified against the
    tokenizer". This is that check, as something that can fail rather than
    something a human squints at.
    """
    problems: list[str] = []
    for name, spec in token_map.items():
        ids = tokenizer(spec["prompt"], add_special_tokens=True)["input_ids"]
        for i, expected in zip(spec["tokens"], spec["pieces"]):
            if i >= len(ids):
                problems.append(f"{name}: index {i} past the end of {ids}")
                continue
            got = tokenizer.decode([ids[i]]).strip()
            if got != expected:
                problems.append(
                    f"{name}: index {i} decodes to {got!r}, expected {expected!r}")
        joined = "".join(spec["pieces"]).lower()
        subject = spec["prompt"].split(" ", 1)[-1].replace(" ", "").lower()
        if joined != subject:
            problems.append(
                f"{name}: tokens spell {joined!r} but the prompt's subject is "
                f"{subject!r}")
    return problems

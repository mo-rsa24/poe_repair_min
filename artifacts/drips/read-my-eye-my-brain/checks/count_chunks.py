"""Count the chunks in each section of the proposal, as converted by pandoc.

Run from the walk folder:  python3 checks/count_chunks.py > checks/count_chunks.out

A chunk is a paragraph, a list item, or a table. A sentence is counted by splitting a
paragraph or list item on a full stop, question mark or colon-free sentence end followed
by a capital letter, so the count is approximate for sentences holding "e.g." or initials.
"""
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "paper" / "proposal.md"
SENTENCE_END = re.compile(r'(?<=[.?!])\s+(?=[A-Z"“])')
ABBREVIATIONS = re.compile(r"\b(e\.g|i\.e|et al|vs)\.\s+$")


def sentences(text):
    parts = SENTENCE_END.split(text.strip())
    merged = []
    for part in parts:
        if merged and ABBREVIATIONS.search(merged[-1] + " "):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return len([p for p in merged if p.strip()])


def main():
    lines = SOURCE.read_text().split("\n")
    sections = []  # (title, [blocks])
    current = None
    block = []
    for line in lines:
        if line.startswith("# ") or line.startswith("## "):
            if current is not None and block:
                current[1].append("\n".join(block))
            block = []
            current = (line.lstrip("# ").strip(), [])
            sections.append(current)
            continue
        if current is None:
            continue
        if line.strip() == "":
            if block:
                current[1].append("\n".join(block))
                block = []
        else:
            block.append(line)
    if current is not None and block:
        current[1].append("\n".join(block))

    print(f"{'section':62} {'para':>4} {'item':>4} {'tbl':>3} {'sent':>4} {'chunks':>6}")
    total = 0
    for title, blocks in sections:
        if title == "Table of Contents":
            continue
        paragraphs = items = tables = sents = 0
        in_table = False
        for b in blocks:
            first = b.lstrip()
            if first.startswith("---") or first.startswith("**Period**") or in_table:
                # the timeline is one pandoc simple table spread over several blocks
                if not in_table:
                    tables += 1
                    in_table = True
                if first.startswith("---") and b.rstrip().endswith("-") and tables and "Year 3.5" in "".join(blocks[: blocks.index(b)]):
                    in_table = False
                continue
            if re.match(r"(-|\d+\.)\s", first):
                items += 1
            else:
                paragraphs += 1
            sents += sentences(re.sub(r"^(-|\d+\.)\s+", "", first))
        chunks = paragraphs + items + tables
        total += chunks
        print(f"{title[:62]:62} {paragraphs:>4} {items:>4} {tables:>3} {sents:>4} {chunks:>6}")
    print(f"{'total':62} {'':>4} {'':>4} {'':>3} {'':>4} {total:>6}")


if __name__ == "__main__":
    main()

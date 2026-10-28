"""Score every markdown file in this walk folder against the checklist in SWEEP.md. Read-only.

Run from the walk folder: python3 -I sweep-score.py . [out.json]
Columns are the rule ids in SWEEP.md; a dot is not applicable, a number is the miss count."""
import json, os, re, sys, glob

ROOT = sys.argv[1]
CARDS = {k for k, v in json.load(open(os.path.expanduser("~/.claude/cache/term-index.json"))).items()
         if isinstance(v, dict) and v.get("kind") == "card"}

STOP = {"map", "rest", "port", "nat", "tls", "redis", "deviation", "cache", "container", "alert",
        "atomic", "ack", "canary", "classifier", "finetune"}
FIELD = re.compile(r"^(Task|Starts from|Settled facts|Outputs|Target|Progress|Proof|Limits|Hand-off):")
SEVEN = ["Background", "Objectives", "What to read", "What is already known", "Rules",
         "What to hand back", "What you will see"]
PROJECT_TOY = re.compile(r"notebook|rebuild/out|\.json|\bcat\b|\bdog\b|example 9|ring|bar\(|quadrant|overlap|A∩B|parts world|mosaic|toy_box|phase\d", re.I)


def kind(rel):
    if rel.startswith(".quiz-prep/"):
        return "solutions" if "solutions" in rel else "coverage"
    if rel.startswith("video-scout/"):
        return "report"
    if rel.startswith("ladder/"):
        return "ladder"
    if rel == "PROMPTS.md":
        return "guide"
    if rel == "REBUILD_WALK.md":
        return "walkmap"
    return "step"


APPLIES = {
    "P1": {"guide", "ladder", "step"}, "P2": {"guide", "ladder", "step"},
    "P3": {"guide", "ladder", "step"}, "P4": {"guide", "ladder", "step"},
    "T1": {"step"}, "T2": {"step"}, "T3": {"step"},
    "S1": {"step"}, "S2": {"step"}, "S3": {"step"}, "S4": {"step"}, "S5": {"step"},
    "S6": {"step"}, "S7": {"step"},
    "L1": {"step"}, "L2": {"step", "ladder", "guide", "walkmap", "coverage", "solutions", "report"},
    "L3": {"step", "ladder", "guide"},
    "E1": {"step", "ladder", "guide", "walkmap"},
    "R1": {"guide"}, "R2": {"ladder"}, "R3": {"ladder"},
    "Q1": {"coverage", "solutions"}, "Q2": {"coverage"}, "Q3": {"coverage"},
    "Q4": {"coverage", "solutions"}, "Q5": {"coverage", "solutions"},
}
RULES = list(APPLIES)


def fences(lines):
    """Yield (start, end, lang, body_lines) for fenced blocks; also return mask of fenced lines."""
    out, mask, i = [], [False] * len(lines), 0
    while i < len(lines):
        m = re.match(r"^(\s*)(```+|~~~+)(\w*)", lines[i])
        if m:
            tick, lang, j = m.group(2), m.group(3), i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(tick):
                j += 1
            for k in range(i, min(j + 1, len(lines))):
                mask[k] = True
            out.append((i, j, lang, lines[i + 1:j]))
            i = j + 1
        else:
            i += 1
    return out, mask


def slug(h):
    h = re.sub(r"[`*\[\]()]", "", h.strip().lower())
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors_of(path, cache={}):
    if path not in cache:
        try:
            hs = re.findall(r"^#{1,6} (.+)$", open(path).read(), re.M)
            cache[path] = {slug(h) for h in hs} | {re.sub(r"[^a-z0-9_\- ]", "", slug(h)) for h in hs}
        except Exception:
            cache[path] = None
    return cache[path]


def score(path):
    rel = os.path.relpath(path, ROOT)
    k = kind(rel)
    text = open(path).read()
    lines = text.split("\n")
    fl, mask = fences(lines)
    res, ex = {}, {}

    def put(rule, n, line=None):
        if k not in APPLIES[rule]:
            res[rule] = None
            return
        res[rule] = n
        if n and line is not None and rule not in ex:
            ex[rule] = line

    # prompts: text fences that carry field labels, the seven headings, or start with a skill line
    prompts = []
    for s, e, lang, body in fl:
        if lang not in ("text", ""):
            continue
        nb = [b for b in body if b.strip()]
        if len(nb) < 5 or sum("─" in b or "│" in b for b in nb) > len(nb) * 0.3:
            continue
        if any(FIELD.match(b) for b in nb) or nb[0].startswith("/") or nb[0].startswith("In the background") \
                or any(b.strip().rstrip(".") in SEVEN for b in nb):
            prompts.append((s, nb))
    p1 = [s for s, nb in prompts if not all(any(b.strip().rstrip(".:").startswith(h) for b in nb) for h in SEVEN)]
    extra = 1 if (k == "guide" and re.search(r"carry all nine", text)) else 0
    put("P1", len(p1) + extra, (p1[0] + 1) if p1 else None)
    p2 = [(s + 1 + i) for s, nb in prompts for i, b in enumerate(nb) if FIELD.match(b)]
    put("P2", len(p2), p2[0] if p2 else None)
    p3 = [s for s, nb in prompts if not any("At the end" in b or "closing report" in b for b in nb)]
    put("P3", len(p3), (p3[0] + 1) if p3 else None)
    p4 = [s for s, nb in prompts if "judge" not in nb[-1].lower()]
    put("P4", len(p4), (p4[0] + 1) if p4 else None)

    # steps
    step_idx = [i for i, l in enumerate(lines) if re.match(r"^## Step \d", l)] + [len(lines)]
    steps = [(step_idx[i], step_idx[i + 1]) for i in range(len(step_idx) - 1)]
    t1, t2, t3 = [], [], []
    runback = bool(re.search(r"Run it back", text))
    for a, b in steps:
        seg = lines[a:b]
        toy = [i for i, l in enumerate(seg) if "🧸 **Toy" in l]
        name = [i for i, l in enumerate(seg) if "🔤 **The name" in l]
        formula = [i for i, l in enumerate(seg) if re.match(r"^#{3,4} .*together: the formula", l)]
        build = [i for i, l in enumerate(seg) if "🔨 **Build" in l]
        if toy and not name:
            t1.append(a + 1)
        if name and ((formula and name[0] > formula[0]) or (build and name[0] > build[0])):
            t2.append(a + 1)
        if toy and PROJECT_TOY.search(seg[toy[0]]) and not runback:
            t3.append(a + toy[0] + 1)
    put("T1", len(t1), t1[0] if t1 else None)
    put("T2", len(t2), t2[0] if t2 else None)
    put("T3", len(t3), t3[0] if t3 else None)

    # display equations
    blocks, i = [], 0
    while i < len(lines):
        l = lines[i].strip()
        if not mask[i] and l.startswith("$$"):
            if l.count("$$") >= 2 and len(l) > 4:
                blocks.append((i, l)); i += 1; continue
            j = i + 1
            while j < len(lines) and "$$" not in lines[j]:
                j += 1
            blocks.append((i, "\n".join(lines[i:j + 1]))); i = j + 1; continue
        i += 1
    ub = [s for s, c in blocks if "\\underbrace" in c]
    s1 = [s for s, c in blocks if "\\underbrace" not in c and not any(abs(u - s) <= 15 for u in ub)]
    put("S1", len(s1), (s1[0] + 1) if s1 else None)
    heads = [i for i, l in enumerate(lines) if re.match(r"^#{2,4} ", l) and not mask[i]] + [len(lines)]
    s2 = []
    for x in range(len(heads) - 1):
        seg = "\n".join(lines[heads[x]:heads[x + 1]])
        if "\\underbrace" in seg and "What the equation says" not in seg:
            s2.append(heads[x] + 1)
    put("S2", len(s2), s2[0] if s2 else None)
    put("S3", 1 if ("\\textcolor" in text and not re.search(r"colou?r key|colou?rs on this page", text, re.I)) else 0, 1)
    vis = [i for i, l in enumerate(lines) if not mask[i] and (l.lstrip().startswith("<table") or re.search(r"!\[[^\]]*\]\(|<img ", l))]
    s4 = [i + 1 for i in vis if not ("How to read" in "\n".join(lines[i:i + 40]) and "What to look for" in "\n".join(lines[i:i + 50]))]
    put("S4", len(s4), s4[0] if s4 else None)
    s5, intab = [], False
    for i, l in enumerate(lines):
        if "<table" in l: intab = True
        if intab and re.search(r"\$[^$]+\$|\*\*|`", l): s5.append(i + 1)
        if "</table>" in l: intab = False
    put("S5", len(s5), s5[0] if s5 else None)
    # paragraphs over three sentences
    s6, para, pstart = [], [], 0
    def flush():
        if not para: return
        p = " ".join(para)
        p = re.sub(r"\$[^$]*\$", "X", p)
        p = re.sub(r"`[^`]*`", "X", p)
        p = re.sub(r"\]\([^)]*\)", "]", p)
        n = len(re.findall(r"[.?!](?=\s+[A-Z(*\[])", p)) + 1
        if n > 3: s6.append(pstart + 1)
    for i, l in enumerate(lines):
        st = l.strip()
        prose = st and not mask[i] and not re.match(r"^(#|\||- |\* |\d+\. |>|<|\$\$|!\[|🧸|🔤|🔨|🎞|💥|⏱|Navigation|⬅)", st)
        if prose:
            if not para: pstart = i
            para.append(st)
        else:
            flush(); para = []
    flush()
    put("S6", len(s6), s6[0] if s6 else None)
    imgs = [i for i, l in enumerate(lines) if not mask[i] and re.search(r"!\[[^\]]*\]\([^)]*\.(svg|png)|<img [^>]*\.(svg|png)", l)]
    def script_above(i):
        j, seen = i - 1, 0
        while j >= 0 and seen < 6:
            if lines[j].strip():
                seen += 1
                if mask[j]:
                    return any(lang == "python" and s <= j <= e for s, e, lang, _ in fl)
            j -= 1
        return False
    s7 = [i + 1 for i in imgs if not script_above(i)]
    put("S7", len(s7), s7[0] if s7 else None)

    # links
    inblock, l1, tab_start = False, [], None
    for i, l in enumerate(lines + [""]):
        if "<!-- links:" in l: inblock = True
        if "<!-- /links" in l: inblock = False
        if l.startswith("|"):
            if tab_start is None: tab_start = i; tab_yt = False; tab_in = inblock
            if re.search(r"youtube\.com|youtu\.be", l): tab_yt = True
        else:
            if tab_start is not None and tab_yt and not tab_in: l1.append(tab_start + 1)
            tab_start = None
    put("L1", len(l1), l1[0] if l1 else None)
    l2 = []
    for i, l in enumerate(lines):
        if mask[i]: continue
        for m in re.finditer(r"(?<!\$)\$([^$\n]+)\$(?!\$)", l):
            if "](" in m.group(1) or "\\href" in m.group(1): l2.append(i + 1)
    put("L2", len(l2), l2[0] if l2 else None)
    prose_txt = "\n".join(l for i, l in enumerate(lines) if not mask[i])
    prose_txt = re.sub(r"\]\([^)]*\)", "]", prose_txt)
    prose_txt = re.sub(r"\$[^$]*\$|`[^`]*`|<[^>]+>|https?://\S+", " ", prose_txt)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", prose_txt)
    found = set()
    for n in (3, 2, 1):
        for x in range(len(words) - n + 1):
            key = re.sub(r"[^a-z0-9]", "", "".join(words[x:x + n]).lower())
            if key in CARDS and key not in STOP: found.add(key)
    linked = len(re.findall(r"#where-to-learn-this", text))
    put("L3", len(found) if linked == 0 else max(0, len(found) - linked), None)
    res["_cards"] = sorted(found)
    l4 = []
    for i, l in enumerate(lines):
        for m in re.finditer(r"\]\(([^)\s]+\.pdf[^)\s]*)", l):
            t = m.group(1)
            paper = re.search(r"papers/|arxiv|openreview|proceedings|neurips|\d{4}\.\d{4,5}", t, re.I)
            ok_place = "Obsidian/Mathematics/references/textbooks/" in t or paper
            if not ok_place or "#page=" not in t: l4.append(i + 1)
    pass
    e1 = [i + 1 for i, l in enumerate(lines) if re.search(r"pip install|python3? -m venv|conda create|uv venv|virtualenv|uv pip install", l)]
    put("E1", len(e1), e1[0] if e1 else None)

    # route-file shape
    put("R1", sum(1 for p in [r"\*\*Objective \(the whole chain\):\*\*", r"\*\*Outcome \(the whole chain\):\*\*"] if not re.search(p, text)), 1)
    sh = [i for i, l in enumerate(lines) if re.match(r"^### \d", l)] + [len(lines)]
    r2, r2ex = 0, None
    for x in range(len(sh) - 1):
        seg = "\n".join(lines[sh[x]:sh[x + 1]])
        miss = sum(1 for p in [r"^\*\*Objective:\*\*", r"^\*\*Outcome:\*\*", r"^\*\*Serves:\*\*"] if not re.search(p, seg, re.M))
        r2 += miss
        if miss and r2ex is None: r2ex = sh[x] + 1
    put("R2", r2, r2ex)
    put("R3", 0 if re.search(r"\| *Open *\| *Textbook *\| *Watch", text) else 1, 1)

    # quiz-prep
    put("Q1", 0 if len(lines) > 1 and lines[1].strip() == "Standard: QUIZ_PREP_FORMAT v1" else 1, 2)
    qh = [i for i, l in enumerate(lines) if re.match(r"^#{2,3} (Q\d|Question \d)", l)] + [len(lines)]
    q2 = []
    for x in range(len(qh) - 1):
        seg = "\n".join(lines[qh[x]:qh[x + 1]])
        if not re.search(r"^\| *Node *\| *Leans on *\| *Tier *\| *Min", seg, re.M): q2.append(qh[x] + 1)
    if len(qh) == 1:  # no per-question headings found
        hdr = [i for i, l in enumerate(lines) if l.startswith("| Node | Leans on")]
        q2 = [i + 1 for i in hdr if "| Tier | Min |" not in lines[i]]
    put("Q2", len(q2), q2[0] if q2 else None)
    put("Q3", 0 if re.search(r"^#{2,3} .*Gaps", text, re.M) else 1, None)
    q4 = []
    for i, l in enumerate(lines):
        if mask[i]: continue
        for m in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", l):
            t = m.group(1)
            if re.match(r"https?:|mailto:", t): continue
            p, _, a = t.partition("#")
            p = p.replace("%20", " ").replace("%2C", ",")
            if p.startswith("/home-mscluster"): continue
            tgt = path if not p else (p if p.startswith("/") else os.path.normpath(os.path.join(os.path.dirname(path), p)))
            if not os.path.exists(tgt): q4.append(i + 1); continue
            if a and tgt.endswith(".md") and not a.startswith("page="):
                an = anchors_of(tgt)
                if an is not None and a not in an: q4.append(i + 1)
    put("Q4", len(q4), q4[0] if q4 else None)
    q5 = [i + 1 for i, l in enumerate(lines) for m in re.finditer(r"\]\(([^)\s]+)\)", l)
          if re.search(r"papers/|arxiv\.org/pdf", m.group(1)) and "#page=" not in m.group(1)]
    put("Q5", len(q5), q5[0] if q5 else None)
    res["_ex"] = ex
    res["_kind"] = k
    return rel, res


files = sorted(glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True) +
               glob.glob(os.path.join(ROOT, ".quiz-prep", "*.md")))
files = sorted(set(f for f in files if not f.endswith("SWEEP.md")))
out = dict(score(f) for f in files)
if len(sys.argv) > 2:
    json.dump({"rules": RULES, "files": out}, open(sys.argv[2], "w"), indent=1)
for rel, r in out.items():
    print(f"{rel[:46]:46} {r['_kind']:9}", " ".join(("  ." if r[x] is None else f"{r[x]:3}") for x in RULES),
          " tot", sum(r[x] or 0 for x in RULES))
print(" " * 57, " ".join(f"{x:>3}" for x in RULES))

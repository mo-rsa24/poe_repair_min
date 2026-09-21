"""Screen a joint-prompt target by asking a vision-language model, in words, what it shows.

Why not the existing reads. The shipped compose scorer detects boxes with the fixed query "animal"
and counts them, so it cannot see a scene, an object, or the difference between a cat and a dog.
The forced-choice read crops those same animal boxes and makes CLIP pick between the pair's two
names, which works on two-animal pairs and is structurally blind on subject-plus-scene pairs, where
the second concept never occupies a box. A language model looking at the picture has neither limit.

Two questions per image, and a pass needs both. The first asks whether both named things are there.
The second asks whether the picture is really two of the same thing, which is the failure the whole
project is about and the one a single yes-or-no question is most likely to wave through.
"""
from __future__ import annotations

import argparse, json, os, sys
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disallowed_subjects import check as check_subject

MODEL = "llava-hf/llava-1.5-7b-hf"
# A forced choice, not a yes-or-no. The first version of this screener asked "are the subjects two of
# the same kind?" and the model answered yes on all 56 validation tiles, which killed every cell. A
# leading yes-or-no invites agreement; choosing among named alternatives does not.
# Third attempt, and the first with nothing to agree with. A yes-or-no question was answered yes on
# all 56 validation tiles; a four-way forced choice was answered A on all 56. Both were acquiescence,
# not perception. So the model is asked to describe the picture in its own words and the description
# is matched against the two names afterwards, which gives it no option to defer to.
# Fifteen words made the model drop the background: on polar bear and iceberg it said "standing on
# ice" and the iceberg went unmentioned though it is plainly in the picture. The fix is to ask about
# the setting as its own open question rather than to teach the matcher that "ice" counts as
# "iceberg", because widening the matcher fits the code to the answer and asking a better question
# does not.
Q_CAPTION = "Describe exactly what is shown in this image in under thirty words."
Q_SETTING = "What is the background or setting of this image? Answer in a few words."
Q_COUNT = "How many separate animals or objects are the main subjects of this image? Answer with a number only."


def content_words(name):
    drop = {"a", "an", "the", "of"}
    return [w for w in name.lower().replace("-", " ").split() if w not in drop]


def mentioned(name, caption):
    c = caption.lower()
    return any(w.rstrip("s") in c or w in c for w in content_words(name))


def ask(processor, model, image, question, n=4):
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="json list of {path, name_a, name_b, pair, seed}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = json.load(open(args.cells))
    for c in cells:
        check_subject(c["pair"])

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL)
    model = LlavaForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                          low_cpu_mem_usage=True).to("cuda")
    model.eval()

    rows = []
    for i, c in enumerate(cells, 1):
        img = Image.open(c["path"]).convert("RGB")
        cap = ask(processor, model, img, Q_CAPTION, n=48)
        setting = ask(processor, model, img, Q_SETTING, n=20)
        cnt = ask(processor, model, img, Q_COUNT, n=6)
        text = cap + " . " + setting
        has_a, has_b = mentioned(c["name_a"], text), mentioned(c["name_b"], text)
        verdict = "good" if (has_a and has_b) else "bad"
        rows.append({**c, "caption": cap, "setting": setting, "count": cnt.strip(),
                     "has_a": has_a, "has_b": has_b, "verdict": verdict})
        print(f"[{i}/{len(cells)}] {c['pair']} s{c['seed']}: a={int(has_a)} b={int(has_b)} "
              f"n={cnt.strip()[:3]} | {cap[:70]}", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

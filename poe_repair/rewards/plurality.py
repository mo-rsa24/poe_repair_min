"""What a composed render should score well on: two things, each itself, not each other.

This is one differentiable objective used at two different times. At render time its gradient
steers the latent (guidance). In training it is the reward the adapter is fine-tuned against
(ReFL / DRaFT). Writing it once means the training signal is the thing the guidance already
showed to move pictures the right way, rather than a second, differently-behaved function.

Three terms, each a number in roughly [-1, 1] before weighting:

``identity``   each region reads as its own concept. CLIP scores the image against the two
               concept prompts, which is the only term with access to what "a cat" means.
``distinct``   the two regions do not look like each other. DINOv2 cosine between the regions,
               which is the anti-fusion term: a single blended animal scores 1 here and is
               punished for it.
``compact``    each concept occupies one place rather than several. A third animal shows up as a
               mask spread over two distant blobs, so the spatial spread of each mask is charged.
``fidelity``   the whole picture's quality, from ImageReward, the preference model the steering
               work in this project already uses. Without it the first three terms are all
               satisfied by two clearly separated, clearly labelled, badly drawn animals.

The regions come from the two concept branches the sampler already computes: the per-concept
clean estimates say where each concept wants mass, and their difference gives a soft mask per
concept. Nothing here needs a detector, so the whole objective stays differentiable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Terms:
    """Every term of one evaluation, so a log can say which part moved."""
    identity: float
    distinct: float
    compact: float
    fidelity: float
    total: float
    contrast: float = 0.0

    def as_dict(self) -> dict:
        return {"identity": self.identity, "distinct": self.distinct,
                "compact": self.compact, "fidelity": self.fidelity, "total": self.total,
                "contrast": self.contrast}


def soft_masks(x0_mix: torch.Tensor, x0_a: torch.Tensor, x0_b: torch.Tensor, *,
               temperature: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Which concept each region of the composed picture is following, as two soft masks.

    ``x0_mix`` is the clean estimate the composed run is heading for, and ``x0_a`` and ``x0_b`` are
    the estimates each concept branch would reach alone from the same state. A region belongs to
    the concept whose own estimate it resembles, so the masks are a softmax over the two
    resemblances. They are smooth, differentiable, and sum to one.

    Comparing each branch against the midpoint of the two instead is degenerate: the two
    differences are equal and opposite by construction, so every region splits exactly evenly and
    the masks carry no information at all.
    """
    d_a = ((x0_mix - x0_a) ** 2).mean(dim=1, keepdim=True)
    d_b = ((x0_mix - x0_b) ** 2).mean(dim=1, keepdim=True)
    scale = (d_a + d_b).mean().clamp_min(1e-8)
    both = torch.cat([-d_a, -d_b], dim=1) / (max(temperature, 1e-6) * scale)
    w = torch.softmax(both, dim=1)
    return w[:, :1], w[:, 1:]


def spread(mask: torch.Tensor) -> torch.Tensor:
    """How far a mask's mass sits from its own centre, as a fraction of the grid's half-diagonal.

    One compact blob scores near 0. Mass split between two corners scores near 1. Charging this
    keeps a concept in one place, which is what stops a third animal appearing: a third object
    pulls one of the two masks apart and pays for it here.
    """
    b, _, h, w = mask.shape
    m = mask / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    ys = torch.linspace(0, 1, h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
    xs = torch.linspace(0, 1, w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
    cy = (m * ys).sum(dim=(2, 3), keepdim=True)
    cx = (m * xs).sum(dim=(2, 3), keepdim=True)
    var = (m * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum(dim=(2, 3))
    return (var.sqrt() / 0.7071).mean()


def _upsample_to(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)


class PluralityReward:
    """CLIP and DINOv2 held open on one device, so a sampler or a trainer can call them per step.

    ``image`` is always the decoded picture in [0, 1] with gradients attached, shape (1, 3, H, W).
    Both backbones are frozen and in eval mode; only the input carries gradient.
    """

    def __init__(self, device: torch.device, *, dtype: torch.dtype = torch.float32,
                 w_identity: float = 1.0, w_distinct: float = 1.0, w_compact: float = 1.0,
                 w_fidelity: float = 0.0, image_size: int = 224, strict_fidelity: bool = False,
                 w_contrast: float = 0.0):
        self.device, self.dtype = device, dtype
        # Training sets this: a fidelity weight that silently scores zero turns a run into a
        # different run than its name says, which is what happened to refl-04-fidelity.
        self.strict_fidelity = strict_fidelity
        self.w_identity, self.w_distinct, self.w_compact = w_identity, w_distinct, w_compact
        self.w_fidelity = w_fidelity
        self.w_contrast = w_contrast
        self._rm = None
        self.image_size = image_size
        self._clip = None
        self._clip_proc = None
        self._dino = None
        self._text_cache: dict[str, torch.Tensor] = {}

    # -- backbones ---------------------------------------------------------
    def _ensure_clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            name = "openai/clip-vit-base-patch32"
            self._clip = CLIPModel.from_pretrained(name).to(self.device).eval()
            self._clip_proc = CLIPProcessor.from_pretrained(name)
            for p in self._clip.parameters():
                p.requires_grad_(False)

    def _ensure_dino(self):
        if self._dino is None:
            self._dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                        trust_repo=True).to(self.device).eval()
            for p in self._dino.parameters():
                p.requires_grad_(False)

    # -- embeddings --------------------------------------------------------
    def _prep(self, image: torch.Tensor, mean, std) -> torch.Tensor:
        x = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear",
                          align_corners=False)
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - m) / s

    def text_embed(self, prompt: str) -> torch.Tensor:
        self._ensure_clip()
        if prompt not in self._text_cache:
            tok = self._clip_proc(text=[prompt], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                f = self._clip.get_text_features(**tok)
            self._text_cache[prompt] = F.normalize(f, dim=-1)
        return self._text_cache[prompt]

    def clip_image_embed(self, image: torch.Tensor) -> torch.Tensor:
        self._ensure_clip()
        x = self._prep(image, (0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711))
        return F.normalize(self._clip.get_image_features(pixel_values=x), dim=-1)

    def dino_embed(self, image: torch.Tensor) -> torch.Tensor:
        self._ensure_dino()
        x = self._prep(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # DINOv2 ViT-S/14 wants a multiple of 14.
        side = (self.image_size // 14) * 14
        if side != x.shape[-1]:
            x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
        return F.normalize(self._dino(x), dim=-1)

    def image_reward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """ImageReward's score for this picture under the joint text, kept differentiable.

        The scale runs about -2 to +2. It is squashed here so one very good or very bad picture
        cannot swamp the other three terms.
        """
        if self._rm is None:
            import ImageReward as RM
            self._rm = RM.load("ImageReward-v1.0", device=str(self.device))
            for q in self._rm.parameters():
                q.requires_grad_(False)
        x = self._prep(image, (0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711))
        text = self._rm.blip.tokenizer(prompt, padding="max_length", truncation=True,
                                       max_length=35, return_tensors="pt").to(self.device)
        # BLIP_Pretrain has no forward(); score_gard runs its two encoders and ImageReward's own
        # mean and std normalisation, which is what puts the score on the -2 to +2 scale.
        r = self._rm.score_gard(text.input_ids, text.attention_mask, x)
        return torch.tanh(r.squeeze() / 2.0)

    # -- the objective -----------------------------------------------------
    def score(self, image: torch.Tensor, *, prompt_a: str, prompt_b: str,
              mask_a: torch.Tensor | None = None, mask_b: torch.Tensor | None = None,
              ) -> tuple[torch.Tensor, Terms]:
        """The objective and its parts. Higher is better; the caller maximises it."""
        image = image.to(self.device, self.dtype).clamp(0, 1)
        ta, tb = self.text_embed(prompt_a), self.text_embed(prompt_b)

        if mask_a is None or mask_b is None:
            region_a = region_b = image
        else:
            ma, mb = _upsample_to(mask_a.to(image), image), _upsample_to(mask_b.to(image), image)
            # Keep the background rather than blacking it out: a hard cut-out is off CLIP's
            # distribution and scores badly for reasons that have nothing to do with the concept.
            region_a = image * ma + image.mean(dim=(2, 3), keepdim=True) * (1 - ma)
            region_b = image * mb + image.mean(dim=(2, 3), keepdim=True) * (1 - mb)

        ea, eb = self.clip_image_embed(region_a), self.clip_image_embed(region_b)
        ia, ib = (ea * ta).sum(), (eb * tb).sum()
        identity = 0.5 * (ia + ib)
        # Each region against the other concept's text as well as its own: a penguin region that
        # has turned into an elephant still scores fairly on "a penguin", but it scores higher on
        # "an elephant", and this margin goes negative. The masks come from where each branch
        # wanted its concept, so the question is what is drawn where the penguin should be.
        contrast = 0.5 * ((ia - (ea * tb).sum()) + (ib - (eb * ta).sum()))

        da, db = self.dino_embed(region_a), self.dino_embed(region_b)
        distinct = 1.0 - (da * db).sum()

        fidelity = torch.zeros((), device=image.device, dtype=image.dtype)
        if self.w_fidelity > 0:
            try:
                fidelity = self.image_reward(image, f"{prompt_a} and {prompt_b}")
            except Exception as exc:                       # the model is optional at render time
                if self.strict_fidelity:
                    raise
                if not getattr(self, "_warned_rm", False):
                    print(f"fidelity term off: {type(exc).__name__}: {exc}", flush=True)
                    self._warned_rm = True

        compact = torch.zeros((), device=image.device, dtype=image.dtype)
        if mask_a is not None and mask_b is not None:
            compact = -0.5 * (spread(mask_a) + spread(mask_b))

        total = (self.w_identity * identity + self.w_distinct * distinct
                 + self.w_compact * compact + self.w_fidelity * fidelity
                 + self.w_contrast * contrast)
        return total, Terms(identity=float(identity.detach()), distinct=float(distinct.detach()),
                            compact=float(compact.detach()), fidelity=float(fidelity.detach()),
                            total=float(total.detach()), contrast=float(contrast.detach()))

# Kinds of composition: working definitions

Read by: [the overlap-dial idea map](IDEA_MAP.md), claim 7.

"Composition" covers several different operations that the literature calls by one word. This file names each one by what it does to the image (or the 2D point), gives the density operator that produces it, and says what a failure of each looks like. Entries marked **[yours]** are definitions this project proposes. Entries marked **[field]** use names that already exist. Every definition is a draft until a toy shows it.

## Words this uses

**Expert.** A model of one density, p_A(x), learned from one training set.

**Region of an image.** A subset S of the pixels (or of the coordinates of a 2D point). x_S is the image restricted to S.

**Product over the same variables.** p_A(x) · p_B(x): both experts score every pixel.

**Product over a partition.** p_A(x_S) · p_B(x_{S'}) with S and S' disjoint: each expert scores only its own region.

## Five kinds of interaction, as a joint prompt draws them [yours]

<a href="../../results/joint-against-poe-by-kind-of-composition/interaction-grids/joint-only-atlas.png"><img src="../../results/joint-against-poe-by-kind-of-composition/interaction-grids/joint-only-atlas.png" width="900" alt="Seven rows of six images: SD3.5 joint prompts 'A and B' for structural, relational, causal, associative and temporal interactions"></a>

Each image is Stable Diffusion 3.5 medium given one prompt, "A and B", at seed 0. Rows are kinds of interaction and columns are different pairs, so a row shows what one kind looks like when a single model is told about both concepts at once. There is no product of experts in the figure. The pairs were chosen over three Codex rounds so that each row isolates its kind, and the selection and rules are in [the pair-selection folder](../../results/joint-against-poe-by-kind-of-composition/pair-selection/brief.md). The five kinds can coexist in one image: a wet dog in rain is structural (two components), causal (rain wets fur) and associative (dogs and rain co-occur in photos).

### 1. Structural: the components keep their identities

**What it is.** How two components occupy and organise one image. The key question is whether each keeps its identity in a shared picture.

**Row 1a, objects.** A violin beside a pineapple, a backpack beside a toaster, a fire hydrant beside a rocking chair: two whole objects, each recognisable on its own, sharing a surface.

**Row 1b, binding.** A red teapot beside a blue bicycle, a striped umbrella beside a polka-dot suitcase, a furry armchair beside a metal lamp: each attribute stays on the object it was named with.

**Failure.** Fusion into one hybrid object, one component dropped, an attribute moved to the wrong object, or the wrong count.

### 2. Relational: the components stand in a stated relation

**What it is.** The two components satisfy a relationship beyond being present: a position, a depth order, or an action one performs on the other. The key question is whether the relation is there, with the right direction and contact.

**Row 2a, spatial.** A fan left of a clock, a balloon above a picnic basket, a boot in front of a teapot, a kite above a beach umbrella.

**Row 2b, action.** A fox chasing a rabbit, a tugboat towing a barge, a hammer over a nail, an eagle diving at a fish.

**Failure.** Both present but the relation missing or reversed: no pursuit, the order flipped, the actor and target not in contact.

### 3. Causal: one component changes the other through a physical process

**What it is.** The state of one component alters the other: heat, water, light or force leaves a visible consequence. The key question is whether the consequence appears and is consistent with its cause.

**Row 3.** A flame curling a paper sheet, a crashing wave around a sandcastle, a blowtorch beside an ice sculpture. The prompts name only the cause and the object, never the effect, so the consequence must come from the model.

**Failure.** The consequence missing, painted on rather than caused, or pointing the wrong way. A model can draw the consequence without representing the process, so a success here shows a consistent picture, not a mechanism.

### 4. Associative: an object in or out of its usual context

**What it is.** Two conditions linked in the learned distribution: a penguin with ice, a boat with water, a chest drain with a pneumothorax. The association helps when it is wanted and leaks when it is not.

**Row 4, out of context.** Each pair puts an object where the training data rarely shows it: a bathtub in a forest clearing, a traffic light on a beach, a chandelier in a canyon, a refrigerator in a meadow. The sailboat (placed on water anyway) and the hospital bed (kept indoors) show the default leaking into the joint prompt itself.

**Failure.** The usual context appearing where the prompt asks for another, or a shortcut such as a drain drawn because a pneumothorax is.

### 5. Temporal: stages of one process consistent in one frame

**What it is.** One moment that carries the trace of an earlier one, and the two must agree. It differs from causal by time: in causal the cause is present and acting (rain falling on a dog), and in temporal the cause has passed and its trace remains (a wet dog among puddles under a clear sky).

**Row 5.** A melting snowman in a puddle, a half-eaten sandwich with crumbs, a nearly bare tree over a pile of its leaves, a dripping cone with drops beneath it, a burnt-out match with a wisp of smoke, a wet dog among puddles under a clear sky.

**Failure.** A present state and a trace that cannot belong together (fresh snow round a melted snowman, a dry dog in the puddles), or the trace dropped.

## Composition on chest radiographs [proposed]

The same kinds apply to findings on a chest radiograph, with one addition the natural images do not need: whether two findings share a cause. Highlights are drawn by hand from each figure's own caption, in the style of [Radiology Masterclass's annotated films](https://www.radiologymasterclass.co.uk/gallery/chest/cardiac_disease/pleural_effusion), on openly licensed images. A coloured mark goes only where the source authors drew an arrow or circle. A finding the caption names but nobody marked is written in the grey strip at the top of the panel instead. The radiology facts behind these entries are quoted with their URLs in [the radiology sources](radiology-sources.md).

### Comorbidity against compound disease

<a href="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/comorbidity-against-compound-disease.png"><img src="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/comorbidity-against-compound-disease.png" width="760" alt="Left: a calcified granuloma ringed where the authors' arrow points. Right: the authors' circles around a tuberculous cavity and a silicotic opacity."></a>

*Left: Ibraheem et al. 2024, Cureus 16(2):e54343, [doi:10.7759/cureus.54343](https://doi.org/10.7759/cureus.54343), Fig. 1, CC BY 4.0. Right: [Pyana Kitenge et al. 2025](https://doi.org/10.5588/ijtldopen.24.0564), IJTLD Open, Fig. 1 Case 2, CC BY; the circles "cv" and "B" are the authors'.*

**Comorbidity.** Two diagnoses on one film, with no shared cause and no effect on each other's appearance. What the left film shows:

- **Grey strip (caption, not marked):** both lungs are over-inflated. That is COPD, in this patient most likely from smoking. On the film it shows as large, dark lungs and flat diaphragms.
- **Orange ring (the authors' blue arrow):** one small white dot in the upper left lung, an old calcified granuloma. It is the scar of an infection that healed years ago, often tuberculosis or a fungal infection.
- **Green arrow (from the source, not labelled here):** an enlarged aortic knob, a third finding the authors put down to high blood pressure.

The two findings got there by separate routes: smoking caused the COPD, and an old infection left the scar. Neither causes the other, and neither changes how the other looks. This is structural composition at the level of disease. Each finding keeps its identity, so a product over a partition is the right operator. This film is a weak example, because the hyperinflation is a subtle change spread over both lungs and the granuloma is only a few pixels; a clearer pair is being sought.

**Superimposed disease.** Two diseases acting on the same tissue, so that each changes how the other looks. What the right film shows, from [Pyana Kitenge et al. 2025](https://doi.org/10.5588/ijtldopen.24.0564): a mineworker's chest with diffuse small round silicotic opacities, a large opacity in the left upper zone, and a cavity in the right upper zone; the authors call the film "suggestive of silico-TB", and a GeneXpert test for tuberculosis was positive.

- **Yellow ring, the authors' "B":** a large silicotic opacity, where small nodules have merged. Tuberculosis "contributes to conglomeration of discrete silicotic nodules" ([Ehrlich et al. 2024](https://doi.org/10.1183/16000617.0168-2024), Pathology).
- **Orange ring, the authors' "cv":** a cavity, the sign of tuberculosis, sitting inside silicotic lung. Its wall and the nodules around it belong to both diseases.
- **Everywhere else:** small nodules that are silicosis, with any tuberculous nodules hidden among them.

A second case shows the same thing on CT. [Martins et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2929613/) describe "a 55-year-old man, who was a former smoker and had chronic silicosis caused by 14 years of work in the granite industry" (Case Presentation), with a previous history of tuberculosis. His radiograph showed "bilateral perihilar conglomerate masses associated with multiple small nodules" (Figure 1), "Culturing was positive for Mycobacterium tuberculosis", and only the CT showed that "the left conglomerate mass presented with a cavitation" (Figure 2). The authors add that a cavity in a conglomerate mass can also come from ischaemic necrosis without infection (Discussion), so the culture, not the cavity, makes the diagnosis.

Why the two compound rather than just coexist: silica damages the lung's macrophages, the cells that defend against both silica and tuberculosis ([Ehrlich et al. 2024](https://doi.org/10.1183/16000617.0168-2024), Co-pathogenesis), so silicosis raises the risk of tuberculosis, "from 2.8 to 39 times higher" than in healthy controls ([Martins et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2929613/), Discussion). Once both are present, the review literature names three effects on the radiograph. Each disease can hide the other. Each can distort the other's pattern. Together they can build a lesion neither builds alone, as when tuberculosis helps silicotic nodules conglomerate into masses ([Ehrlich et al. 2024](https://doi.org/10.1183/16000617.0168-2024), quoted under "From the literature" below). So in the cavity and around it the pixels belong to both diseases at once, which is why fusion here is the pathology rather than a failure (see "When fusion is the target" below).

**Each disease alone, then together.**

<a href="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/tuberculosis-silicosis-silicotuberculosis.png"><img src="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/tuberculosis-silicosis-silicotuberculosis.png" width="900" alt="Left: tuberculosis opacities ringed at the authors' arrows. Middle: silicosis, caption only. Right: a tuberculous cavity ringed in orange inside a lung full of silicotic nodules, with a silicotic mass in yellow."></a>

*Orange marks tuberculosis and yellow marks silicosis. Left: [Ter-Ovanesyan et al. 2026](https://doi.org/10.7759/cureus.112415), Cureus, Fig. 1, CC BY 4.0. Middle: [He et al. 2026](https://doi.org/10.1186/s12890-025-04091-0), BMC Pulmonary Medicine, Fig. 1a, cropped from a four-panel figure, CC BY 4.0. Right: [Pyana Kitenge et al. 2025](https://doi.org/10.5588/ijtldopen.24.0564), IJTLD Open, Fig. 1 Case 2, CC BY. Orange rings on the left sit at the tips of the authors' red arrows; the rings on the right follow the authors' circles "cv" and "B". The silicosis film has no author marks, so its findings are in the grey strip.*

- **Tuberculosis alone.** A patchy opacity in the right upper lobe and nodular opacities in the left upper and mid zones, in a young adult with culture-proven tuberculosis and no silica exposure.
- **Silicosis alone.** Small nodules through both lungs and two upper-zone masses where the nodules have merged (complicated silicosis, progressive massive fibrosis), in a patient from a cohort with active tuberculosis excluded.
- **Silicosis with tuberculosis.** A mineworker's lung full of small silicotic nodules, a large silicotic opacity ("B") in the left upper zone, and a cavity ("cv") in the right upper zone; the authors call the film "suggestive of silico-TB", and a GeneXpert test was positive. The cavity sits inside silicotic lung, so its edges and the nodules around it belong to both diseases. On this film the tuberculosis is visible only through the cavity; everywhere else the silicosis hides it, which is the first of the three effects above.

**More superimposed diseases.** Each pair below is one disease growing in tissue another disease has already changed, so the second disease looks different from how it looks in a normal lung.

<a href="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/aspergilloma-in-old-tuberculous-cavity.png"><img src="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/aspergilloma-in-old-tuberculous-cavity.png" width="900" alt="The same upper lung three times: an old tuberculous cavity, the cavity five years later with a rounded opacity inside, and a CT showing a fungus ball inside the cavity."></a>

*[Pattamapaspong et al. 2024](https://doi.org/10.1093/bjro/tzae031), BJR Open, Fig. 2 panels A, C and D, each cropped from the article's figure, CC BY 4.0. Rings sit where the authors' arrows (A) and arrowheads (C, D) point.*

- **A fungus ball in an old tuberculosis cavity (aspergilloma).** Healed tuberculosis left an empty cavity (left). Five years later a rounded opacity fills it (middle). On CT the ball sits inside the cavity with a thin crescent of air between it and the wall (right). The fungus grows only because the cavity is there, and the cavity now looks solid rather than empty. This is superimposed and temporal at once: one disease's trace becomes the other's home.

<a href="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/fibrosis-before-and-during-acute-exacerbation.png"><img src="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/fibrosis-before-and-during-acute-exacerbation.png" width="520" alt="CT of a fibrotic lung with honeycombing, then the same lung with new ground glass over it."></a>

*[Hobbs et al. 2021](https://doi.org/10.1148/ryct.2021200279), Radiology: Cardiothoracic Imaging, Figs. 17a and 17b, CC BY 4.0. No author marks, so the findings are in the grey strips.*

- **An acute exacerbation on top of pulmonary fibrosis.** The first CT shows honeycomb cysts in an otherwise dark lung. In the second, new ground glass turns the whole lung grey, so the cysts now sit in grey lung instead of black. The paper notes this pattern can mimic heart failure or infection; infection was clinically excluded.

<a href="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/disease-hidden-by-a-changed-lung.png"><img src="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/disease-hidden-by-a-changed-lung.png" width="900" alt="Left: CT of pneumonia in an emphysematous right lower lobe. Right: CT of a cancer growing within honeycombed fibrosis."></a>

*Left: [Ragnoli et al. 2026](https://doi.org/10.3390/diagnostics16091329), Diagnostics, Fig. 2A (cropped), CC BY 4.0; the caption places the consolidation in panel A and the text says 2B, and panel A shows it. Right: [Kim et al. 2026](https://doi.org/10.3390/jcm15176863), Journal of Clinical Medicine, Fig. 2a (cropped from the article PDF), CC BY 4.0. Neither has author marks.*

- **Pneumonia in emphysema.** The destroyed airspaces cannot fill with fluid, so the consolidation is a net of white strands around black holes rather than a solid white patch. (The usual radiology name for this look, "Swiss cheese", is general knowledge, not this paper's.)
- **Cancer in fibrosis.** The tumour grew around honeycomb cysts and dilated airways, so it looked like a consolidation full of air spaces. The first reader took it for pneumonia; biopsy showed squamous cell carcinoma.

**About the two names.** "Comorbidity" is a standard clinical term: Feinstein (1970) defined it ([review in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6878355); [Cambridge: "What exactly is comorbidity?"](https://www.cambridge.org/core/product/CA42801A3BBE5DDBCF328BEA9CDAFF4A/core-reader)) as any distinct additional clinical entity alongside an index disease. It does not require that the two leave each other's appearance alone; that restriction is this project's. "Compound disease" is not a standard term in radiology or in deep learning for radiology. Radiology says "superimposed" (as in a large heart with superimposed infection, quoted in [the radiology sources](radiology-sources.md)), "concomitant" or "coexisting", and names specific combinations (silicotuberculosis). Deep-learning papers on chest X-rays say "co-occurring findings" or "label co-occurrence".

### Two findings from one cause, against two findings from none

<a href="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/one-shared-cause-against-none.png"><img src="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/one-shared-cause-against-none.png" width="760" alt="Left: heart failure film, caption only. Right: the comorbidity film."></a>

*Left: Palmeira et al. 2016, BMC Research Notes 9:391, [doi:10.1186/s13104-016-2181-5](https://doi.org/10.1186/s13104-016-2181-5), Fig. 3, CC BY 4.0. Right: Ibraheem et al. 2024, as above.*

**A diagnosis built from findings.** Cardiomegaly and bilateral pleural effusions sit in mostly different regions, which is structural. Neither causes the other: the big heart and the effusions both come from one underlying problem, the failing heart.

**Comorbidity, for contrast.** Again two findings in two places, but here they got there by separate routes: smoking for the COPD, an old infection for the scar. The two films look alike in kind, two findings in two regions, so whether findings share a cause is not something the pixels show. It is knowledge about the disease, which a label ("heart failure") can carry and a product of two finding experts cannot.

### Before and after a drain

<a href="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/pneumothorax-before-and-after-drain.png"><img src="../../results/joint-against-poe-by-kind-of-composition/radiograph-highlights/pneumothorax-before-and-after-drain.png" width="760" alt="Left: the authors' arrows on the pleural line, ringed. Right: the same patient with the pigtail drain ringed at the authors' arrow."></a>

*Both films are of one patient, a 74-year-old man with COPD: Beste et al. 2025, Cureus 17(6):e85525, [doi:10.7759/cureus.85525](https://doi.org/10.7759/cureus.85525), Fig. 2 (before) and Fig. 3 (after the drain, a portable AP film), CC BY 4.0. The purple rings sit at the tips of the authors' three arrows on the pleural line, and the yellow ring follows their arrow to the drain. The re-expansion is in the caption only. None of the marks is a radiologist's segmentation.*

**Drain and pneumothorax.** Three kinds at once:

- **Associative:** drains appear because pneumothoraces are treated, so the two are correlated in the data.
- **Temporal:** a film with a drain is a moment after treatment, often with the collapse resolving.
- **Relational:** the drain tip must sit in the pleural space.

This is why a drain is a shortcut for a classifier and why "pneumothorax without a drain" is negation against a strong association.

### The three pairs

| Pair | Kinds | Why |
|---|---|---|
| Drain + pneumothorax | associative, temporal, relational | treatment follows the diagnosis; the film is after treatment; the tip has a required position |
| Cardiomegaly + effusion | structural, common cause | different regions; both from heart failure; merge only at the heart border |
| Silicosis + tuberculosis | causal, same tissue | silicosis raises tuberculosis risk; both change the same parenchyma into one pattern |

### Composition is hierarchical

Each level composes the one below with its own kind of interaction:

1. **Pixels into findings.** An enlarged cardiac outline, a blunted costophrenic angle, a pleural line. Mostly structural and relational.
2. **Findings into a diagnosis.** Cardiomegaly + effusion + oedema read as heart failure. A common cause, many findings to one diagnosis.
3. **Diagnoses into a patient.** Heart failure with an unrelated granuloma is comorbidity. Silicosis with tuberculosis is a compound disease.
4. **A patient across time.** Before and after a drain. Temporal.

For modelling this splits the question in two: compose finding experts upward, or train one diagnosis expert and ask whether it equals the product of its findings. CheXpert's labels already carry a parent-child structure (lung opacity above oedema, consolidation and pneumonia; recalled, citation pending from route 2).

### When fusion is the target

<a href="../../results/joint-against-poe-by-kind-of-composition/interaction-grids/when-fusion-is-the-target.png"><img src="../../results/joint-against-poe-by-kind-of-composition/interaction-grids/when-fusion-is-the-target.png" width="900" alt="Four rows (colour or material, style and content, shape and material, intended hybrid) of four pairs; each cell is a joint prompt beside the product of experts"></a>

Each cell is Stable Diffusion 3.5 medium at seed 0: left, a joint prompt (the first row uses the plain "A and B"; the other rows describe the hybrid), and right, the product of experts of the two parts. The product fuses well where one concept is a style or surface over the other (every style-and-content pair, the golden and pumpkin teapots, the avocado chair). It fails where the target is a new object or a mixed value: one red ball with a blue patch rather than purple, a plain apple and cat, hybrids that lose one part.

The five kinds treat fusion as failure. It is the right answer when both concepts constrain the same variable and the target really has both values. That is intersection, the product over the same variables:

- **Colour or material mixing:** "blue" and "red" on one region give purple; a glass apple.
- **Style and content:** a dog in oil paint.
- **Shape and material:** an avocado-shaped chair.
- **Intended hybrids:** SuperDiff's equal-density blends.
- **Compound disease:** silicotuberculosis, where the merged appearance is the pathology.

English "and" hides three readings: two objects ("a blue cube and a red cube"), one object split into regions ("a red and blue ball"), and one blended value (purple). A joint prompt picks whichever reading its captions taught it. A same-variable product can only give the blend. For silicotuberculosis this has a limit: a product of a silicosis expert and a tuberculosis expert gives where their supports overlap, not the true compound pattern, which only joint data teaches.

## Co-occurrence [yours]

**What it is.** Two concepts appear fully and distinctly in one scene, each in its own region, possibly interacting spatially (a cat beside a dog; a square inside a circle; a chest drain beside a pneumothorax edge).

**The operator.** The composition operator with a shared background, C[p_b, p_A, p_B](x) = p_b(x) · [p_A(x)/p_b(x)] · [p_B(x)/p_b(x)] (Bradley et al. 2025a, Definition 5.1, p.6). Its target is the partition product with a layout, p(layout) · p_A(x_S | layout) · p_B(x_{S'} | layout), and the operator reaches that target exactly when the experts are factorized conditionals: each one matches the background outside its own region (Definition 5.2).

**The background is a choice.** An empty-scene background composes, and the unconditional model as background fails (Bradley et al. 2025a, Figure 3). In the 2D quadrant toy the background is uniform, so dividing by it changes the product only by a constant, and the plain product p_A · p_B tests co-occurrence there. On images the background is not uniform, so the plain and the background-divided products have to be compared.

**What it needs from training data.** Each expert must copy the background outside its own region, and its learned score must be local (Bradley et al. 2025a, Definition 5.2 condition 3; Bradley 2025b). Seeing each concept at many positions over varied backgrounds is one way to get there, not the condition itself. In the 2D slot toy, rotating 5% of each training set into the other slot lifts co-occurrence mass from `0.002` to `0.66` ([the kinds notebook](kinds-of-composition-on-the-papers-toys.ipynb), Figure 2). The two concepts need not be seen together, because the regions are disjoint.

**Failure: contention [yours].** Both experts claim the same pixels or the same features, so a product over the same variables is used where a product over a partition was meant. It shows in three forms:

- **Fusion:** the product's high-density points are compromises each expert finds moderately likely, a chimera (cat pixels and dog pixels fused).
- **Dominance:** one concept missing or faint, CO3's mode collision.
- **Leakage:** an attribute moving to the other object, Bounded Attention's semantic leakage.

The overlap can be in features rather than pixels. A dog with a horse fails because both control the subject, while a dog with a red hat composes on the same pixels (Bradley et al. 2025a, Figure 7). Field names for the symptoms: concept bleeding, attribute leakage, and catastrophic neglect when one concept vanishes (Chefer et al. 2023, *Attend-and-Excite*; recalled, not opened). Region-restricted products exist: MultiDiffusion (Bar-Tal et al. 2023, confident on the method).

**2D toy.** A constrains x1, B constrains x2 (the quadrant setup). Each coordinate is a "region".

**From the literature: the field's names.** Projective composition with coordinate restrictions, and factorized conditionals as the condition that makes it exact (Bradley et al. 2025a, [arXiv 2502.04549](https://arxiv.org/abs/2502.04549), Definitions 4.1 and 5.2, p.5 to 6; confident, quoted). Conditional projective composition, equivalent to local conditional scores (Bradley 2025b, [arXiv 2509.16447](https://arxiv.org/abs/2509.16447), p.4 to 5; confident, quoted). Region-based generation by fusing per-region denoising steps (MultiDiffusion, [arXiv 2302.08113](https://arxiv.org/abs/2302.08113), p.1; confident, quoted). Factor-wise routing of each pixel to one expert (FactorDiff, [arXiv 2607.11758](https://arxiv.org/abs/2607.11758); abstract only). Many copies of one concept is called length generalization (Bradley et al. 2025a p.2) or numeracy (T2I-CompBench++, [arXiv 2307.06350](https://arxiv.org/abs/2307.06350), p.2).

**From the literature: where it agrees.** The disjoint-regions definition matches. Models trained on single-object images compose into multi-object images that are out of distribution for every one of them (Bradley et al. 2025a, Figure 2, p.2). The target in Theorem 5.3, p_b(x_{M_b}) ∏ p_i(x_{M_i}), is this entry's partition product with the background filling the rest (p.6).

**From the literature: where it differs.** The field's operator divides by a shared background, C[p_b, p_1, p_2](x) = p_b(x) ∏ p_i(x)/p_b(x) (Definition 5.1, p.6), and the choice of background decides success. An empty-scene background composes up to several objects. The unconditional model as background "entirely fails" and sometimes produces no objects (Figure 3, settings A and B, p.3; confident, quoted). Under factorized conditionals, composing commutes with noising, so plain reverse diffusion with summed scores samples the composition exactly (Theorem 5.3, p.6 to 7). The layout need not be supplied: in Bradley 2025b the regions come from the locality of the learned score. Shadows, occlusion and overlap between objects break the factorization and degrade the result as the object count grows (Bradley et al. 2025a p.7 and p.16).

**From the literature: names for contention.** "Semantic leakage" between subjects, caused by attention mixing their features and worst for semantically or visually similar subjects (Bounded Attention, [arXiv 2403.16990](https://arxiv.org/abs/2403.16990), p.1 and p.3 to 4; confident, quoted). "Mode collision": the joint-prompt model lands in a mode dominated by one concept, so one concept is "missing, faint, or colliding awkwardly with another" (CO3, [arXiv 2509.25940](https://arxiv.org/abs/2509.25940), p.1 to 2; confident, quoted), followed up by TILT ([arXiv 2607.21606](https://arxiv.org/abs/2607.21606), abstract only). Composing two concepts that control the same feature ("photo of a dog" with "photo of a horse") gives "unexpected results", while dog with "red hat" composes (Bradley et al. 2025a, Figure 7, p.10; confident, quoted). None of the sources read uses the word chimera.

## Intersection [yours, the simple product]

**What it is.** One thing satisfies both concepts at once, on the same pixels: a shape that is both square-ish and circle-ish. The field uses "AND" for three different objects (the Bayes product, the simple product and the equal-density set). This entry is the simple product, the one PoE computes.

**The operator.** A product over the same variables, p_A(x) · p_B(x). This is what PoE literally computes.

**What it needs from training data.** Each expert must give the intersecting region nonzero density. A product intersects supports and creates none.

**Failure: empty intersection.** The two supports do not meet, so the product has (near) zero mass and samples go wherever the sampler's errors take them.

**2D toy.** The notebook's ring-of-Gaussians times bar: both constrain the same two coordinates.

**From the literature: the field's names.** The simple product p_1(x)p_2(x) (Bradley et al. 2025a, [arXiv 2502.04549](https://arxiv.org/abs/2502.04549), section 3.1, p.4; confident, quoted). Product of experts over heterogeneous models, sampled by annealed MCMC with sequential Monte Carlo resampling (Zhang et al. 2025, [arXiv 2506.08894](https://arxiv.org/abs/2506.08894); abstract only). The product path p_t^prod = q¹_t q²_t / Z_t (Feynman-Kac Correctors, [arXiv 2503.02819](https://arxiv.org/abs/2503.02819), eq. 15, p.4; confident, quoted). "AND" names three different objects: the Bayes product p(x)∏p(x|c_i)/p(x) (Liu et al. 2022, [arXiv 2206.01714](https://arxiv.org/abs/2206.01714), eqs. 9 and 10, p.7), the simple product (Bradley et al. 2025a p.3), and the set where every model gives equal density (SuperDiff, [arXiv 2412.17762](https://arxiv.org/abs/2412.17762), p.2). All confident, quoted.

**From the literature: where it agrees.** "A product intersects supports and creates none" is stated as a theorem-level fact: the simple product is zero wherever either factor is zero, so it can never be out of distribution for either expert (Bradley et al. 2025a p.4; confident, quoted).

**From the literature: why summed scores miss the product.** Reverse diffusion needs the score of the noised product, and adding the two scores gives the score of the product of the noised densities. These differ for any t > 0 (Du et al. 2023, [arXiv 2302.11552](https://arxiv.org/abs/2302.11552), eqs. 11 and 12, p.4; confident, quoted). Feynman-Kac Correctors derive the per-sample weight that makes the product-of-noised path exact and resample by it (p.3 to 4). Bradley et al. 2025a show that the mismatch vanishes for factorized conditionals (Theorem 5.3), so the sampler error belongs to experts that overlap on the same coordinates. Even where the composed density is well defined, its annealing path can change abruptly while each expert's path is smooth, which makes it hard for any annealing sampler (Lemma 7.2, p.9). A Metropolis-Hastings acceptance step can be built from the score alone by integrating it along a line, so MALA-style correction does not need an energy model (Sjöberg et al., [arXiv 2307.14012](https://arxiv.org/abs/2307.14012); abstract only).

**From the literature: sampler error against missing data.** Soiffer et al. 2026 ([arXiv 2606.23920](https://arxiv.org/abs/2606.23920), Table 1, p.10; confident, quoted) cross two error sources with two data settings on 2D Gaussians. The fit is measured as sliced Wasserstein-2 distance to the true composition (lower is better, 5000 samples, mean of 30 training runs). With the target inside the training data, Feynman-Kac correction helps. With it outside, and with learned scores from 10,000 samples, the fit goes from 0.39 with 1 particle to 2.82 with 256 (factorized case). With exact analytic scores the same correction improves the out-of-distribution fit from 0.64 to 0.08 (non-factorized case). Out of distribution, the score error dominates and no sampler removes it.

## Interaction [yours]

**What it is.** The concepts change each other: rain plus a dog gives a wet dog; tuberculosis plus silicosis gives a pattern neither shows alone (silicotuberculosis is a recognised clinical entity, confident).

**The operator.** By Bayes' rule, p(x | A, B) ∝ [p(x | A) p(x | B) / p(x)] · [p(A, B | x) / (p(A | x) p(B | x))], where the constant hidden by ∝ is p(A) p(B) / p(A, B). The second factor is the interaction term. It is 1 (zero in log form) when A and B are conditionally independent given x, which includes every case where each label is a fixed function of the image.

**A test for it.** Two causes acting on overlapping mechanisms interact. Causes acting on separate mechanisms give exactly the Bayes product (Soiffer et al. 2026, Appendix B). Both definitions agree that a product of separately trained experts cannot produce an interaction without joint data.

**Not a relation.** Interaction here means a change in appearance. Benchmarks such as T2I-CompBench++ use the word for relation verbs (holding, wearing), which this file calls a relation.

**What it needs from training data.** Either joint samples showing the interaction, or an expert whose density already contains the changed version (wet dogs among the dog images). A product cannot invent the change. It appears once the training data shows A as it looks under B: in the 2D toy, 2% overlap was enough ([the kinds notebook](kinds-of-composition-on-the-papers-toys.ipynb), Figure 2).

**Failure: missing interaction.** The output is a co-occurrence or an intersection where an interaction was meant (a dry dog standing in rain).

**2D toy.** Inside the A∩B quadrant, shift the x1 distribution (for example, A's mean moves from −0.5 to −0.8 when B is present). At 0% overlap a product of the separate experts cannot produce the shift.

**From the literature: the same argument, published.** "Because the two events c₁ and c₂ are conditionally independent given x (since they are deterministic functions of x), we can compute p̂ in terms of the individual conditionals: p(x|c₁,c₂) ∝ p(x|c₁)p(x|c₂)/p(x)" (Bradley et al. 2025a, [arXiv 2502.04549](https://arxiv.org/abs/2502.04549), p.4; confident, quoted). The paper calls this the Bayes composition and shows it cannot place mass where the training distribution had none. Training on cluttered scenes lets "the models learn how multiple objects should interact and even overlap correctly" (p.17). That is the data condition this entry names.

**From the literature: interaction as non-separable causes.** When two causes act on different hidden mechanisms ("causally separable", non-overlapping intervention targets), the joint distribution is exactly P(x|do(a),do(a′)) = P_a(x) P_a′(x) / P_0(x), the Bayes form above (Soiffer et al. 2026, [arXiv 2606.23920](https://arxiv.org/abs/2606.23920), p.2 and Appendix B, p.20 to 21; confident, quoted). Appendix B says this generalizes factorized conditionals. When the targets overlap, the product form is wrong. That case is the field's interaction, and the paper cites a method for finding such pairs from data (Xu et al. 2024, *Automated discovery of pairwise interactions from unstructured data*; recalled through the citation, not opened). Tuberculosis and silicosis, both acting on lung parenchyma, are an overlapping-target pair under this definition (inferred).

**From the literature: does any source add a term beyond the product.** No. All 17 shortlist texts were searched for "interaction", "epistasis", "synergy" and "non-additive", and the matching passages read; none adds an interaction factor to a product of single-concept experts at sampling time. A joint effect comes from joint training data (Bradley et al. 2025a p.17) or from a model of the joint. CO3 keeps the joint-prompt model p(x|C) as its main term and divides out the single-concept models, p(x|C)/∏p(x|c_i) ([arXiv 2509.25940](https://arxiv.org/abs/2509.25940), eq. 1, p.2; confident, quoted). Binding an attribute to the wrong object ("a red bear") is reported as a failure of AND, with no term proposed (Liu et al. 2022, [arXiv 2206.01714](https://arxiv.org/abs/2206.01714), p.13; confident, quoted).

**From the literature: a clash of names.** T2I-CompBench++ uses "interaction" for a relation verb between two objects: "Non-spatial relationships usually describe the interactions between two objects" (hold, wear, look at) ([arXiv 2307.06350](https://arxiv.org/abs/2307.06350), p.5; confident, quoted). A person holding a mat is two objects in a relation, which this file would call co-occurrence with a layout. The project's word means concepts changing each other's appearance.

## Composition over features [field]

**What it is.** Two concepts share the same pixels but control different features of them: an oil painting of a particular dog, a dog wearing a red hat, an object of one colour composed with an object of another.

**The operator.** Projective composition in a feature space: if some invertible map A sends the experts to factorized conditionals, the same background-divided product C[p_b, p_1, p_2] computed in pixel space is a projective composition in that feature space, with no need to know A (Bradley et al. 2025a, [arXiv 2502.04549](https://arxiv.org/abs/2502.04549), Theorem 6.1 and Lemma 6.2, p.8; confident, quoted).

**What it needs from training data.** Concepts that are disentangled in some feature space. A necessary check is that the mean-difference vectors μ_i − μ_b of the concepts are orthogonal (Lemma 8.1, p.10). CLIP embeddings group dog, horse and cat together, and watercolour with oil painting, which predicts that within-group pairs fail (Figure 8, p.11).

**Failure.** Sampling is guaranteed only when A is orthogonal (Lemma 7.1, p.9). Yellow objects compose with blue, cyan and magenta but not with brown, grey, green or red, and the paper explains this through how the blue channel is distributed (Figure 5, p.5).

**Why it matters here.** The project's split puts "same pixels" under intersection, and this kind sits on the same pixels with a partition operator (inferred).

## Disjunction, or mixture [field]

**What it is.** A sample that is A or B: the mixture of the two densities. SuperDiff calls it OR ([arXiv 2412.17762](https://arxiv.org/abs/2412.17762), p.2), and Du et al. 2023 show it beside the product ([arXiv 2302.11552](https://arxiv.org/abs/2302.11552), Figure 2, p.5). Confident, quoted.

## Equal-density blend [field]

**What it is.** A sample that both models find equally likely. SuperDiff calls this its AND (p.2), and its Figure 1 shows the outputs as single hybrid objects ("a flamingo and a candy cane", p.2; confident, quoted). Concept blending treats such hybrids as the goal (Olearo et al. 2025, [arXiv 2506.23630](https://arxiv.org/abs/2506.23630); abstract only). What this file calls a chimera under contention is the intended output here.

## Negation [field]

**What it is.** A and not B. Liu et al. 2022 define two operators, conjunction (AND) and negation (NOT) ([arXiv 2206.01714](https://arxiv.org/abs/2206.01714), p.7; confident, quoted). The project's medical cells (pneumothorax without a drain) are this kind.

## Relations [field]

**What it is.** Two objects in a stated spatial or action relation: "a book on the left of a bird", "a cat in front of a chair", "a woman is holding a yoga mat". T2I-CompBench++ splits these into 2D-spatial, 3D-spatial and non-spatial relationships, beside attribute binding, numeracy and complex compositions ([arXiv 2307.06350](https://arxiv.org/abs/2307.06350), p.1 to 2 and p.5; confident, quoted). These are categories of prompt, not density operators.

## Four questions that place any composition [proposed]

Every kind above is a set of answers to four separate questions. Asking them one at a time stops "co-occurring", "interacting" and "hybrid" from blurring into each other.

| Question | Answers |
|---|---|
| Where do the two sit? | separate regions · the same pixels |
| What does each control? | different features (shape and colour, content and style) · the same features |
| Does one depend on the other? | no · in placement or pose (a relation) · in appearance (an interaction) |
| What comes out? | two objects · one object carrying both properties · one fused object (a hybrid) |

**How to say it.** "The cat and the dog co-occur" says only where they are. "They interact" says one changes how the other looks. "A hybrid" says one object came out where two were meant, or where a blend was the goal. A cat sleeping beside a dog co-occurs with no interaction. A dog chasing a cat co-occurs and is in a relation. A wet dog in rain is an interaction, and it is still a dog, so it is not a hybrid.

## The kinds, with examples [proposed]

| Kind | The four answers | Everyday example | Chest X-ray example |
|---|---|---|---|
| spatial co-occurrence | separate · different · no · two objects | a cat beside a dog | a chest drain beside a pneumothorax edge |
| attribute binding | same · different · no · one object, both properties | a red cube | a *large left-sided* effusion (size and side bound to the effusion) |
| style and content | same · different · no · one object, both properties | an oil painting of your dog | the same lungs as a portable AP film rather than a PA film |
| relation | separate · different · placement or pose · two objects | a dog chasing a cat; an astronaut riding a horse, not a horse riding an astronaut | a drain tip in the pleural space rather than in the fissure; a feeding tube in the stomach rather than a bronchus |
| physical interaction | either · either · appearance · two objects, changed | a wet dog in rain; a shadow cast across a table | a pneumothorax in an emphysematous lung, where bullae can hide or mimic the pleural line |
| clinical interaction (compound disease) | same · same · appearance · a new pattern | none | silicotuberculosis |
| contention | same · same · no · a fused object or one concept lost, by failure | "a dog and a horse" giving one dog-horse animal | a generated drain that merges into a rib |
| equal-density blend (hybrid on purpose) | same · same · no · one fused object, intended | a flamingo made of candy cane | none sensible |
| counting | separate · different · no · several copies | four swans | two drains; multiple nodules |
| negation | either · either · no · A with B absent | a beach with no people | a pneumothorax with no drain |
| implicit composition | separate · different · no · something unasked-for added | "a boat" also gives water | "pneumothorax" also gives a drain (the NIH shortcut) |

## Findings, diagnoses and diseases [proposed]

**Finding.** What the image alone settles, PadChest's test: cardiomegaly (an enlarged heart outline), pleural effusion, a pleural line, and also pneumothorax, a diagnosis the film settles by itself. A *descriptor* (Fleischner) names a sign without implying a cause, such as consolidation. The line moves between schemes: pulmonary oedema is an observation in CheXpert and a differential diagnosis in PadChest.

**Diagnosis built from findings.** A condition that causes a set of findings and is read back from them with uncertainty, using information beyond the image. Heart failure causes cardiomegaly, pleural effusion and pulmonary oedema, and the radiograph is not a diagnostic test for it (Pan et al. 2021). So the interaction term is zero for the findings given the diagnosis, and not for the diagnosis given the findings, which is claim 7's label term. How to say it: "heart failure shows as three findings".

**Comorbidity.** Two diagnoses present with no effect on each other's appearance: a pneumothorax and an unrelated calcified granuloma. This is co-occurrence at the level of disease, and it is narrower than the field's word, which covers any two conditions present together whether or not one changes how the other looks. How to say it: "the two co-occur".

**Compound disease.** Two causes acting on the same tissue: silicotuberculosis. The radiograph shows three effects (Ehrlich et al. 2024). Each disease can mask the other, each can distort the other's pattern, and together they can build a lesion neither gives alone, such as conglomerate masses. This is interaction in the causal sense (overlapping mechanisms, Soiffer et al. 2026, Appendix B). How to say it: "silicosis and tuberculosis interact".

**Consequence.** Something that follows from a diagnosis: a drain placed for a pneumothorax, sternotomy wires after cardiac surgery. It is a dependence in the data, and it changes appearance too: a treated pneumothorax re-expands and can lose its pleural line, which is the temporal kind. One device can follow several diagnoses (12.7% of pneumothorax-negative supine films in one cohort had a thoracic tube, Rueckel et al. 2021). It is what a shortcut learns. How to say it: "the drain is a consequence of the pneumothorax, so the two are correlated in the data".

**Labels in the datasets mix these.** NIH ChestX-ray14 labels include findings (effusion, cardiomegaly, pneumothorax) and diagnoses (pneumonia), so the kind of a label pair has to be decided per pair.

### From the literature: findings, diagnoses and diseases

Citations from route 2, with the full shortlist, the drain-rate method and the NIH co-occurrence counts in [the findings and diagnoses return](routes/02-findings-and-diagnoses-literature.return.md). Each citation is flagged quoted, counted here, inferred or abstract only.

**Finding, in the literature.** Radiology separates what the film shows from what causes it, and the datasets draw that line in different places. PadChest defines a radiographic finding as anything "radiologists can assert based solely on the interpretation of a chest x-ray image" ([Bustos et al. 2020](https://arxiv.org/abs/1901.07441), p.10) [quoted]. This puts pneumothorax among the findings, because the image alone settles it, and cardiomegaly and pleural effusion there too (p.27) [quoted]. The Fleischner glossary labels terms as anatomy, physiology or "an abnormal finding", and marks some as descriptors "to differentiate them from terms that imply a radiologic diagnosis"; consolidation, for example, "does not indicate the pathologic nature of the condition" ([Bankier et al. 2024](https://doi.org/10.1148/radiol.232558), Introduction and Consolidation) [quoted]. CheXpert calls its labels "observations" and says it kept pneumonia "despite being a clinical diagnosis" ([Irvin et al. 2019](https://arxiv.org/abs/1901.07031), p.2) [quoted]. ChestX-ray8 calls every label a "thoracic disease" and merges the UMLS types "Diseases or Syndromes" and "Findings" ([Wang et al. 2017](https://arxiv.org/abs/1705.02315), p.3) [quoted]. Pulmonary oedema shows the disagreement: it is an observation in CheXpert and a differential diagnosis in PadChest (p.29) [quoted].

**Diagnosis built from findings, in the literature.** A diagnosis is read from findings, and the findings do not fix it. PadChest keeps diagnoses in their own tree because they need "the patient's clinical information and/or additional studies" beyond the image, and it files heart insufficiency there (Bustos et al. 2020, p.10 and p.29) [quoted]. For heart failure the signs on the film are the ones the project names: as pulmonary pressure rises, "pulmonary venous congestion, interstitial oedema, alveolar oedema and pleural effusions" appear ([Pan et al. 2021](https://doi.org/10.1007/s00392-021-01836-9), Introduction) [quoted]. The same paper states that "The CXR is not a diagnostic test for HF" and that not every acute heart failure patient shows congestion. Among 975 admitted patients, venous congestion appeared on 78% of films, Kerley B lines on 71%, effusions on 67% and alveolar oedema on 64% (Abstract) [quoted]. When heart failure is adjudicated from the whole record, a radiographic congestion score separates it with an AUC of 0.84 ([Kobayashi et al. 2020](https://doi.org/10.1371/journal.pmed.1003419), Results) [quoted]. The findings are evidence for the diagnosis and the diagnosis causes the findings, so the composition runs one way in generation and the other way, with uncertainty, in reading [inferred]. The label hierarchies in the field are a different relation, subsumption: "Cardiomegaly implies the presence of Enlarged Cardiomediastinum" ([Pham et al.](https://arxiv.org/abs/1911.06475), p.8) [quoted].

**Comorbidity, in the literature.** The field uses "comorbidity" and "co-occurrence" for any two conditions present together and says nothing about whether one alters the other's appearance. [Kobayashi et al. (2020](https://doi.org/10.1371/journal.pmed.1003419), Results) call obesity and COPD comorbidities of heart failure [quoted]. [Lin et al. (2026)](https://arxiv.org/abs/2609.21541) use the word for correlated labels in chest X-ray classifiers [abstract only]. Ehrlich et al. (2024, Plain chest radiography) speak of the "co-occurrence of silicosis with TB" for a pair that does alter appearance [quoted]. ChestX-ray8 describes label co-occurrence as "the interactions and relationships among objects" (Wang et al. 2017, p.3) [quoted]. The project's narrower meaning (two diagnoses with no effect on each other's look) therefore needs stating as its own definition. One piece of evidence fits it: a radiographic heart failure score performed about the same with or without COPD and pneumonia (Kobayashi et al. 2020, section "Radiographic congestion score index as a diagnostic tool for acute heart failure") [quoted].

**Compound disease, in the literature.** Silicotuberculosis is "the combination of silicosis and active TB and/or post-tuberculous lung disease" (Ehrlich et al. 2024, Definitions) [quoted]. The two share a mechanism, because silica damages the macrophage, "the frontline of innate defence against silica and M. tuberculosis" (Co-pathogenesis) [quoted]. The review describes three effects on the radiograph. Each disease can hide the other: "early nodular changes of TB may be difficult to identify, especially if silicosis is evident", and "silicosis may be missed in the presence of tuberculous abnormalities" [quoted]. Each can change the other's pattern, as when TB modifies the symmetrical "angel wing" spread of silicotic nodules and silica exposure gives "a more profuse tuberculous nodulation" (Plain chest radiography) [quoted]. Together they can build a lesion neither builds alone, as when the tuberculous process "contributes to conglomeration of discrete silicotic nodules" into masses (Pathology) [quoted]. The review's summary is that combined disease is "a phenotype in which clear distinction between the presentation and effects of the two diseases may not always be possible" (Aetiology) [quoted].

**Consequence, in the literature.** A drain follows a pneumothorax because it treats it. [Oakden-Rayner et al.](https://arxiv.org/abs/1909.12475) call it "a non-causal image feature in the setting of pneumothorax" and "a healthcare process variable" (2019, p.3 to 4) [quoted]. In NIH ChestX-ray14, 80% of pneumothorax-positive test images carry a drain (p.3) [quoted]. Among 3,543 other pneumothorax-labelled NIH images, 1,637 (46.2%) do, by non-expert labels ([Cheplygina et al. 2025](https://arxiv.org/abs/2309.02244), labels from their Zenodo release) [counted here]. A classifier's pneumothorax AUC is 0.94 on images with drains and 0.77 on images without (Oakden-Rayner et al. 2019, p.3) [quoted]. The consequence also changes the image of the cause. Most NIH pneumothorax cases "were already treated with chest tubes and often did not show radiographic evidence of ectopic pleural gas" ([Oakden-Rayner 2019](https://arxiv.org/abs/1907.12720), p.7) [quoted], so a drain image can carry the label with no pleural line to see. Drains are a consequence of more than one diagnosis: 12.7% of pneumothorax-negative supine films in one hospital cohort had a thoracic tube ([Rueckel et al. 2021](https://doi.org/10.1007/s00330-021-07833-w), Table 1) [quoted]. Generators learn these dependencies as well. In RoentMod's edits, "cardiomegaly often appeared regardless of the user prompt" ([Cooke et al. 2025](https://arxiv.org/abs/2509.08640), p.7) [quoted].

## Which kind a pair of concepts is

Open. The first guess for the project's pairs:

The column "Where they separate" decides the kind: pixels give co-occurrence, features give composition over features, and neither gives intersection or interaction.

| Pair | Where they separate | Kind | Why |
|---|---|---|---|
| chest drain, pneumothorax | pixels | co-occurrence, plus associative and temporal | a tube and a lung edge in mostly different pixels; correlated in the data (NIH: 46% to 80% of pneumothorax images carry a drain, by subset); treated lungs re-expand |
| tuberculosis, silicosis | neither | interaction | both alter the same lung parenchyma |
| cat, dog (joint prompt) | pixels | co-occurrence | two objects, two regions |
| a dog, oil paint | features | composition over features | content and style on the same pixels |
| square, triangle under PoE | neither | intersection | one shape scored by both experts |

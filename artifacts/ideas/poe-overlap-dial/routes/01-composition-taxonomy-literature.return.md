# Return: composition taxonomy in the literature

Route 1 of [the overlap-dial idea map](../IDEA_MAP.md), anchored at claim 7. The prompt is [the taxonomy route prompt](01-composition-taxonomy-literature.prompt.md). The passages and comparisons behind every line here are in the [combined reading on kinds of composition](../../../drips/combine-kinds-of-composition/COMBINE_MAP.md), and the citations are written into [the composition terms](../composition-terms.md).

**What ran.** One search pass over the arXiv API (18 queries, about 60 candidates; Semantic Scholar was rate-limited and not used). A shortlist of 17 was downloaded and its text extracted. Four combined-reading walks, (a) to (d), each compared four sources with a differences table. No number in this return was computed here. Every number is quoted from a paper.

## 1. The shortlist

Ranked by whether the paper defines or separates kinds of composition, or explains when a product fails. "Read" means the passages named were opened in the full text. "Abstract only" means the paper was ranked on its abstract.

| # | arXiv | Paper | Why it ranks here | Read |
|---|---|---|---|---|
| 1 | 2502.04549 | Bradley et al. 2025, *Mechanisms of Projective Composition of Diffusion Models* | defines composition (projective composition), contrasts simple product and Bayes composition, proves when summed scores are exact, gives a failure heuristic | p.1 to 11 |
| 2 | 2606.23920 | Soiffer et al. 2026, *Catastrophic Compositional Generation* | separates sampler error from score error with numbers; defines interaction as non-separable causes | p.1 to 2, 8, 10, 20 to 21 |
| 3 | 2509.16447 | Bradley 2025, *Local Mechanisms of Compositional Generalization in Conditional Diffusion* | region composition is equivalent to local conditional scores | p.1, 4 to 5 |
| 4 | 2509.25940 | Dutta et al. 2025, *Steer Away From Mode Collisions* (CO3), seed | names mode collision; uses a joint model beside the product | p.1 to 4 |
| 5 | 2302.11552 | Du et al. 2023, *Reduce, Reuse, Recycle* (already on the shelf) | why summed scores miss the product | p.4 to 5 |
| 6 | 2503.02819 | Skreta et al. 2025, *Feynman-Kac Correctors in Diffusion* | exact reweighting for the product path | p.1 to 4 |
| 7 | 2412.17762 | Skreta et al. 2024, *The Superposition of Diffusion Models* (SuperDiff) | a different AND (equal density) and OR (mixture) | p.1 to 2, 5 |
| 8 | 2206.01714 | Liu et al. 2022, *Composable Diffusion Models* | AND and NOT, the conditional-independence assumption, binding failures | p.7, 12 to 13 |
| 9 | 2307.06350 | Huang et al., *T2I-CompBench++* | the benchmark category scheme; "interaction" as a relation verb | p.1 to 2, 5 |
| 10 | 2403.16990 | Dahary et al. 2024, *Bounded Attention* | semantic leakage, worst for similar subjects | p.1, 3 to 4 |
| 11 | 2302.08113 | Bar-Tal et al. 2023, *MultiDiffusion* | region-restricted composition at the sampler | p.1 |
| 12 | 2607.21606 | Dutta et al. 2026, *TILT* | follow-up formalising mode collision | abstract only |
| 13 | 2607.11758 | Huang et al. 2026, *FactorDiff*, seed | per-factor routing to experts, a learned partition | abstract only |
| 14 | 2307.14012 | Sjöberg et al. 2023, *MCMC-Correction of Score-Based Diffusion Models*, seed | MH acceptance from the score alone, relevant to the project's MALA code | abstract only |
| 15 | 2506.08894 | Zhang et al. 2025, *Product of Experts for Visual Generation*, seed | PoE over heterogeneous experts, annealed sampling | abstract only |
| 16 | 2605.07078 | Wang et al. 2026, *Test-Time Compositional Generalization via Concept Discovery*, seed | PoE of discovered density modes | abstract only |
| 17 | 2310.09336 | Okawa et al. 2023, *Compositional Abilities Emerge Multiplicatively* | data structure decides when composition emerges | abstract only |

Seen and not shortlisted: Kamb and Ganguli 2024 (2412.20292, local patch-mosaic creativity), Liang et al. 2024 (2408.13256, factorized representations), Du and Kaelbling 2024 (2402.01103, position paper), ConceptMix (2408.14339), SANEval (2602.00249), concept blending (2407.14280, 2506.23630), FOCUS (2510.02315), SDO (2608.13820). No paper found proposes a taxonomy of compositional generation as its subject. Bradley et al. 2025a is the nearest: it compares three definitions.

## 2. Per kind: supports, splits or contradicts

**Co-occurrence: supported, with a different operator.** Disjoint-region composition is the best-understood case in the field (projective composition with coordinate restrictions, Bradley et al. 2025a). It needs no joint data, and plain reverse diffusion samples it exactly. The field writes the operator as a background-divided product, p_b ∏ p_i/p_b, and the background choice decides success. The quadrant toy's p_A · p_B has no background division, so it may not be testing this kind.

**Contention: supported and split three ways.** The field sees fusion, dominance (CO3's mode collision) and leakage (Bounded Attention). It also places the cause in feature space as well as pixels: dog with horse fails, dog with red hat works (Bradley et al. 2025a, Figure 7). Nobody says "chimera". SuperDiff's AND and the concept-blending papers produce the fused object on purpose.

**Intersection: supported, and the field's word is overloaded.** The simple product cannot leave either expert's support (Bradley et al. 2025a p.4), as the entry says. "AND" means three different objects across Liu et al., Bradley et al. and SuperDiff. The summed-score sampler error applies to experts that overlap on the same coordinates and vanishes for factorized ones. The project's nested-shapes example is a co-occurrence by its own definition.

**Interaction: supported in conclusion, split in definition.** The Bayes route matches the project's exactly (Bradley et al. 2025a p.4). A second route, causal separability, gives the same product form when causes act on separate mechanisms, and names the overlapping case as interaction (Soiffer et al. 2026, Appendix B). T2I-CompBench++ uses "interaction" for something else (relation verbs).

**Kinds the project has not named.** Composition over features, which sits on the same pixels with a partition operator. Disjunction (mixture). The equal-density blend. Negation. Relations as a prompt category. Each is now a [field] entry in [the composition terms](../composition-terms.md).

## 3. Does any source treat interaction as needing a term beyond the product?

No. All 17 texts were searched for interaction, epistasis, synergy and non-additive effects, and no source adds an interaction factor to a product of single-concept experts at sampling time. Each places the joint effect in the data or in a model trained on the joint. Bradley et al. 2025a p.17 say cluttered training lets models "learn how multiple objects should interact". CO3 keeps the joint-prompt model as its main term. Soiffer et al. 2026 treat non-separable causes as the case where the product form fails, with no correction term. This supports claim 7's proposed repair: two operators (partition, same variables) plus data conditions, with interaction as a data condition.

The literature adds two things to claim 7. Its exact argument is already published, as the Bayes composition (Bradley et al. 2025a p.4), so it is a known result for the project to cite. The causal version gives interaction a testable definition (overlapping intervention targets). That leaves one open question for the medical case: whether tuberculosis and silicosis labels are better treated as functions of the image or as causes of it.

## 4. Proposed changes to the [yours] entries

Eleven proposals, numbered under "Proposed changes from the literature" at the end of [the composition terms](../composition-terms.md). In short:

1. Co-occurrence: write the operator as the background-divided product and state the factorized-conditionals requirement.
2. Co-occurrence: name the background as a choice; check whether the quadrant toy's p_A · p_B tests intersection instead.
3. Co-occurrence: replace "seen at many positions" with the locality condition.
4. Contention: three observed forms (fusion, dominance, leakage).
5. Contention: the overlap can be in features, not only pixels.
6. Contention: mark the *Attend-and-Excite* citation as recalled.
7. Intersection: move the nested-shapes example to co-occurrence.
8. Intersection: say which AND (the simple product).
9. Interaction: add the causal definition beside the Bayes one.
10. Interaction: guard the word against the benchmark meaning.
11. Pair table: add a column for the space where the concepts separate.

## For the idea map, not applied

- The idea map's Sources row cites Du et al. 2023's summed-score argument as "Section 3". The PDF puts it at p.4, eqs. 11 and 12.
- Claim 4's check (sampler against missing data on analytic experts) has a published precedent with numbers in Soiffer et al. 2026, Table 1.
- Claim 2's check runs on the quadrant setup. Proposal 2 above asks whether that setup, as built, is a co-occurrence toy.

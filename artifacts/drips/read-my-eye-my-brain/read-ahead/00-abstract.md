# Read-ahead: 1. Abstract

Printed section 1, `paper/proposal.md` lines 85 to 91. Three paragraphs, eight sentences, walked
one sentence per chunk. No code exists for any chunk.

### 1 · sentence · abstract paragraph 1

**As printed** (quoted, line 87)

This study proposes the design, development, and validation of an integrated, low-cost artificial intelligence (AI) enabled retinal screening system --- working title "My Eye, My Brain" --- intended to enable early detection of vision-threatening and systemically linked eye disease in underserved South African communities.

**What it claims**: nothing testable. It names the thing to be built and who it is for.

**Pays off, or is paid off by**: section 4.1 (the aim), section 8 (the method).

**On the object**: nothing. State on reading: read.

**Prediction to ask**: none.

**Slips**: "low-cost" is never given a figure anywhere in the document (ran: search for currency).

### 2 · sentence · abstract paragraph 1

**As printed** (quoted, line 87)

The research addresses two closely coupled problems: first, the near-total absence of AI diagnostic models trained and validated specifically on African patient populations, despite growing evidence that models trained on non-African data may not generalise reliably to local populations, imaging hardware, and clinical workflows; and second, the lack of an affordable, portable retinal imaging device suited to deployment in rural and peri-urban primary care and pharmacy-based clinics.

**What it claims**: three things. Models trained and validated on African patients are nearly
absent. Models trained elsewhere may lose accuracy here. No affordable portable camera suits
these clinics.

**Pays off, or is paid off by**: section 3 gaps 1 and 3; section 6.2 (the evidence on
generalising); section 6.3 (the cameras).

**On the object, stage 3**: one model, two tables, one per population. The claim "may not
generalise" is a claim that the two sensitivities differ by more than chance would give.

**The piece**: `two_tables` · lane: typed · about 8 lines · answer key written on approval of
the object.

**Prediction to ask**: a model finds 16 of 20 diseased patients at one clinic and 14 of 20 at
another. Is that a model that works worse at the second clinic?

**Slips**: section 6.2 (line 155) reports a Singapore-trained model reaching "clinically
acceptable sensitivity and specificity" on a Zambian screening population, and section 6.3
(line 159) says multiple low-cost smartphone designs already exist. Both sit against this
sentence's "may not generalise" and "the lack of an affordable ... device". quoted.

### 3 · sentence · abstract paragraph 2

**As printed** (quoted, line 89)

The study is structured as two interdependent research streams.

**What it claims**: nothing: a connecting chunk.

**Pays off, or is paid off by**: sentences 4 and 5; section 8.

**On the object**: nothing. State on reading: read.

**Slips**: section 8 organises the work into three phases, and section 11 (line 209) refers to
"the two research streams described in Section 8". quoted.

### 4 · sentence · abstract paragraph 2

**As printed** (quoted, line 89)

The first involves the curation, fine-tuning, and clinical validation of a deep learning model --- referred to throughout as "Sam" --- for the detection of diabetic retinopathy, glaucoma, and related retinal pathology, using a combination of established public benchmark datasets and newly collected South African patient data.

**What it claims**: a model will be fine-tuned on public data plus new South African data, and
validated clinically.

**Pays off, or is paid off by**: objectives 1 and 2 (lines 121 and 123); section 8.1 (line 179).

**On the object, stage 3**: two models on the same 100 patients, the baseline and the fine-tuned
one. Because the patients are the same, the comparison is patient by patient, which needs fewer
patients than comparing two separate clinics.

**The piece**: `paired_change` · lane: typed · about 8 lines · answer key written on approval.

**Prediction to ask**: the baseline finds 14 of the 20 diseased patients and the fine-tuned
model finds 16. How many patients did the fine-tuning actually change the answer for: 2, or
could it be more?

**Slips**: "Sam" shares its name with Segment Anything (SAM), a well-known vision model
(recalled). The proposal does not say what architecture "Sam" is; the primary question calls it
"lightweight" (line 135) and section 6.1 names convolutional networks and vision transformers
without choosing. quoted.

### 5 · sentence · abstract paragraph 2

**As printed** (quoted, line 89)

The second involves the design and prototyping of a smartphone-based, low-cost fundus imaging device suitable for operation by non-specialist clinical staff.

**What it claims**: a phone-based camera will be designed and built, usable without a trained
photographer.

**Pays off, or is paid off by**: objective 3 (line 125); section 8.1 paragraph 2 (line 181);
section 6.3 (line 159); secondary question 2 (line 141).

**On the object, stage 3, in part**: the object cannot reach the optics. It reaches the one
consequence that shows up in the table: how many of the 100 photographs are too poor to grade,
and what happens to the two fractions when those patients are dropped or counted as referrals.

**The piece**: `ungradable` · lane: typed · about 8 lines · answer key written on approval.

**Prediction to ask**: 15 of the 100 photographs are too blurred to grade. If they are left out,
does the model's reported sensitivity go up, go down, or is it impossible to say?

**Slips**: none in the sentence. The proposal gives no design parameter (field of view,
resolution, cost) for the device anywhere (ran: search).

### 6 · sentence · abstract paragraph 2

**As printed** (quoted, line 89)

The two streams are validated jointly through an integrated field study that tests the model\'s performance on images captured by the study\'s own prototype device under real rural clinical conditions, directly addressing a well-documented failure mode in which AI systems validated on curated datasets underperform once deployed with different imaging workflows and hardware.

**What it claims**: the model is tested on the prototype's own photographs in real clinics, and
this is the answer to a known failure where a model does well on curated data and worse in the
field.

**Pays off, or is paid off by**: objective 4 (line 127); section 8.2 (line 185); section 3 gap 2
(line 107); secondary question 3 (line 143); section 6.2's last sentence (line 155).

**On the object, stage 3**: the same 100 patients photographed twice, once with a conventional
camera and once with the prototype. Two tables. The drop from one to the other is the quantity
this sentence is about.

**The piece**: `two_tables`, reused from chunk 2.

**Prediction to ask**: the field study photographs patients with the prototype only. Without the
second camera, what would a low sensitivity tell you: that the model is poor on these patients,
or that the camera is?

**Slips**: section 8.2 compares against conventional photography only "where feasible"
(line 185). Without it, a field result cannot be split into the camera's share and the
population's share. The failure the sentence calls well-documented is cited in the body only
through Katende 2025 (line 155); Beede et al. 2020 is in the references (line 257) and never
cited by name. quoted and ran.

### 7 · sentence · abstract paragraph 3

**As printed** (quoted, line 91)

The study further maps its research outputs against South Africa\'s emerging regulatory framework for AI/ML-enabled medical devices, positioning the resulting evidence base to support future regulatory authorisation and translation into primary healthcare and retail pharmacy screening channels.

**What it claims**: the evidence is shaped so a regulator could later use it.

**Pays off, or is paid off by**: objective 5 (line 129); section 6.5 (line 167); section 8.3
(line 189); section 12 (line 225).

**On the object, stage 4**: the regulator's requirement, as the proposal reports it, is
"clinical evidence with subgroup performance analysis" (line 167). On the object that is the 20
diseased patients split into subgroups, and the range each subgroup's fraction could take.

**The piece**: `interval` · lane: verified by hand · answer key written on approval.

**Prediction to ask**: split the 20 diseased patients into four subgroups of 5. The model finds
4 of 5 in one of them. How sure can you be of that subgroup's sensitivity?

**Slips**: the regulatory submission is in the timeline's right-hand column in Year 3
(line 218) while sections 8.3 and 12 put it outside the doctoral scope. The table's heading
says "context only", so the two agree; the reader may still ask who does that work. quoted.

### 8 · sentence · abstract paragraph 3

**As printed** (quoted, line 91)

Expected contributions include a validated, population-appropriate diagnostic model, an openly documented low-cost device design, and a body of clinical evidence that directly addresses a recognised gap in African-validated ophthalmic AI systems.

**What it claims**: three outputs: a model, a device design, and clinical evidence.

**Pays off, or is paid off by**: section 7 (line 171); section 13 (lines 229 to 233).

**On the object**: nothing new. Its state follows chunks 4, 5 and 6.

**Prediction to ask**: none.

**Slips**: "openly documented" appears only here (ran: search). Section 12 (line 225) plans
commercialisation "through a university technology transfer office or spin-out structure",
which sits against an open design unless the proposal says what is open and what is not.
quoted.

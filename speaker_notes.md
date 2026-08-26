## Opening

**Slide 1 — Title**  
"Hi everyone, I'm Clive Binu, a PhD student at Rochester Institute of Technology, working with Jeyhan Kartaltepe."  
"Today I'll talk about SpecPT, a transformer-based approach for spectroscopic redshift estimation, and especially how we adapt it to lower-resolution grism data."  
**Transition:** "I want to start from the problem statement, because this project really begins with the scale of data we now face."

**Slide 2 — The Telescopes Behind the Data**  
"These are the telescopes that produced the data we work with — DESI, HST, Roman, Euclid, and JWST."  
**Transition:** "Each of these surveys produces massive amounts of spectral data, and that scale is what motivates the approach I'll describe today."

**Slide 3 — The Scale of Modern Survey Data**  
"To put this in perspective — HST's total archive is about 184 terabytes, which is already a lot of data."  
"JWST adds roughly 66 terabytes per year, accumulating to about 364 terabytes over its 5.5-year mission."  
"Euclid, over its 6-year mission, will produce about 170 petabytes of data — roughly a thousand times HST's archive."  
"And Roman is even larger — about 4 petabytes per year, totaling around 20 petabytes over its 5-year prime mission."  
"The point is: these telescopes are not just generating more data than before, they are generating data on a fundamentally different scale."  
**Transition:** "This is the scale that makes traditional redshift measurement methods increasingly impractical."

## Motivation

**Slide 4 — Traditional Methods Break at Scale**  
"Traditionally, redshift is measured either by visual inspection or with template-based methods like cross-correlation."  
"These methods are useful, but they become much harder to scale when datasets get very large or when the spectra are noisy and low resolution."  
"With facilities like Roman, Euclid, and JWST producing millions of spectra, these methods are increasingly impractical."  
**Transition:** "So this is where machine learning becomes useful — not because astronomy changed, but because the scale of the data changed."

**Slide 5 — SpecPT: A Foundation Model for Spectra**  
"SpecPT is a transformer-based model — the same architecture behind large language models, but applied to spectral flux values instead of text."  
"The key idea is self-attention: the model learns which spectral regions are predictive of redshift, much like how astronomers identify emission lines and absorption features by eye."  
"It captures both line features and continuum shape simultaneously."  
**Transition:** "So the first question is: where does the model learn these spectral representations?"

## Method

**Slide 6 — SpecPT Architecture Walkthrough** (6 sub-steps, arrow keys advance, ~3 min total)

**Sub-step 0 (Input):** "Here's one spectrum — 7781 flux values. Spikes are emission lines; the wiggle along the trace is noise. Flux is z-scored before entering the model."

**Sub-step 1 (Convolution):** "A pattern of 41 weights slides along the spectrum. Sixty-four filters run at once — the heatmap below fills in as the window travels. Stride 2, so half the length."

**Sub-step 2 (Three convs + pool):** "Each convolution halves the length and widens the description: 64 → 128 → 256 channels, then a max-pool leaves 487 positions."

**Sub-step 3 (Transformer):** "The convolutional stack is flattened and projected to a 512-number vector, then three transformer encoder layers refine it into a 512-d latent representation — this shared encoder block is exactly what both models reuse."

**Sub-step 4 (Decoder):** "A transformer decoder refines the latent, then linear layers expand it back to 7781 values. The reconstruction lands on top of the input — lines and continuum survive the round trip."

**Sub-step 5 (Redshift head):** "For redshift, the same encoder output flows through self-attention and five residual MLP blocks, then a small prediction head with a Softplus outputs one continuous z, trained with an NMAD loss." [Press 'r' to replay]

**Transition:** "That pretrained encoder is what we carry into HST grism data."

**Slide 6 — Pre-training on DESI**  
"SpecPT is pre-trained on half a million spectra from the DESI survey."  
"At this stage, the model learns a rich representation of spectral features that are predictive of redshift."  
"This gives us a pretrained foundation that we can then adapt to harder domains instead of starting from scratch."  
**Transition:** "Once we have that pretrained model, the real challenge is moving it into a much harder domain: 3D-HST grism data."

**Slide 7 — Fine-tuning to 3D-HST Grism**  
"This is where transfer learning becomes essential. The pre-trained model learns reusable spectral features, and fine-tuning adapts those features to a new instrument."  
"3D-HST G141 grism operates at R~130, which is much lower resolution than DESI."  
"After a quality cut of SNR >= 2.5, the usable sample contains about 8,000 spectra."  
"We applied data augmentation techniques — redshift-shift augmentation and balanced sampling — to increase the effective dataset to about 100,000 spectra."  
**Transition:** "And when we do that, we get encouraging results — but we also uncover a very clear failure mode."

**Slide 8 — Transfer Learning: Pretrain → Adapt → Specialize**  
"The pre-training on DESI learns general spectral representations."  
"Fine-tuning adapts these representations to new instrument domains."  
"This enables effective learning even with limited target data — only about 8,000 usable HST spectra."  
"Data augmentation expands the effective sample to about 100,000 spectra."  
**Transition:** "Now let's see how well this transfer approach actually performs."

## Results

**Slide 9 — Results on HST Grism Data**  
"So this first result is promising, because it shows that transfer learning is working on grism data."  
"The diagonal structures in the plot tell us the model is making systematic mistakes rather than just random noise."  
"Our interpretation is that the model is sometimes confusing single emission lines and therefore landing on the wrong redshift branch."  
**Transition:** "That led us to look more closely at the failure mode."

**Slide 12 — Examining Outliers**  
"Looking at the outliers in more detail, we can see the structured nature of the failures."  
"These are not random — they follow specific patterns related to emission line confusion."  
**Transition:** "That led us to ask what other information could help break that ambiguity."

**Slide 13 — Breaking Degeneracy with Photometry**  
"The solution we explored was adding photometric information, because broadband colors provide complementary redshift information when the spectrum alone is ambiguous."  
"The 3D-HST COSMOS field is ideal for this because it provides both the grism spectra and rich multi-wavelength photometry — 30+ bands from the COSMOS2020 catalog."  
"So instead of asking the spectral model to solve everything on its own, we let photometry provide extra context."  
"This is really the multimodal step in the project."  
**Transition:** "Once we add that information, we can test whether those structured failures actually improve."

**Slide 14 — Photometry Results**  
"After adding photometry, those diagonal artifacts are reduced, which supports the idea that the extra information is helping resolve the line-confusion problem."  
"The catastrophic outlier fraction drops from 9.11% to 2.21%."  
**Transition:** "That improvement is encouraging, but we also wanted to check whether the model is making these predictions for the right physical reasons."

**Slide 10 — GradCAM (HST spectrum)**  
"We used GradCAM for interpretability, because we wanted to check whether the model is attending to meaningful spectral regions."  
"This shows the model's attention on an individual HST spectrum."  
**Transition:** "And here's the integrated gradient view."

**Slide 11 — GradCAM (IG)**  
"The integrated gradient analysis confirms that the model is focusing on emission lines and continuum features that make physical sense."  
"That part matters scientifically, because in astronomy we care not only about performance, but also about whether the model's behavior makes physical sense."  
**Transition:** "So to finish, I'll end with where we want to take this next."

## Closing

**Slide 15 — Next Steps**  
"The next steps are to apply the fine-tuned SpecPT to data from telescopes like Euclid, combine the photometry CNN and arbiter model into a single unified model, and extend SpecPT to other predictions like SFR estimation and AGN classification."  
**Transition:** "So the main takeaway is..."

**Slide 16 — Summary**  
"SpecPT transfers effectively from DESI to HST grism data — the catastrophic outlier fraction drops from 52% to 37%."  
"Photometric data further improves the final predictions to 6.24%."  
"GradCAM highlights spectral regions that drive predictions, confirming the model learns physically meaningful features."  
**Transition:** "Thank you."

**Slide 17 — Questions**

**Slide 22 — How SpecPT Processes a Spectrum**  
"Let me walk through what the data looks like at each stage of SpecPT."  
"Panel 1 shows the input — a galaxy's spectrum with about 3600 flux measurements across the wavelength axis. The highlighted peak is a strong emission line."  
"Panel 2: SpecPT applies three convolutional filters of different sizes — k=41, k=21, and k=11. Think of these as different-sized 'reading windows' that slide across the spectrum. Each one captures different things: the k=41 window finds the broad features like the overall continuum shape; the k=21 window picks out the prominent emission peaks; the k=11 window catches the fine detail and narrow features. At the bottom you see the spectrum gets broken into 30 patches, like a sentence broken into words."  
"Panel 3: Now the model looks at all 30 patches together — that's the transformer. It figures out which patches contain the most useful information. Watch as it highlights certain patches as more important."  
"The important patches are then combined into a single summary — a list of 384 features that capture the spectrum's essential fingerprint."  
"Panel 4: Here are the 384 learned features, but we can only show a few. Each one captures something meaningful — like 'strong emission line', 'line ratios', or 'continuum shape'. These are the model's vocabulary for describing a spectrum. Watch them pulse as the model uses them."  
"Panel 5: Finally, the redshift head combines all those features and produces a single deterministic prediction. The model has been trained to output one specific redshift value for each input spectrum. The final answer: z equals 0.84."
**Transition:** "This shows how the raw spectrum is transformed at every step — from light to a single number."

**Slide 23 — How SpecPT Reconstructs a Spectrum**  
"Now I'll walk through the autoencoder pathway in SpecPT — the same architecture that's used during pretraining to learn good representations."  
"Panel 1 shows the input — a galaxy's spectrum with about 3600 flux measurements. The highlighted peak is a strong emission line."  
"Panel 2: The spectrum goes through three convolutional filters of different sizes — k=41, k=21, and k=11. The k=41 window finds broad features like the overall continuum; k=21 picks out the prominent emission peaks; k=11 catches the fine detail. At the bottom you see the spectrum gets broken into 30 patches, like a sentence broken into words."  
"Panel 3: Now the model looks at all 30 patches together — the transformer encoder. It applies self-attention across 3 layers with 8 heads each, and figures out which patches contain the most useful information. The model compresses these into a 512-dimensional latent representation."  
"Panel 4: This is the Transformer Decoder — the inverse of the encoder. Watch how the flow is reversed: the 512-d summary at the top expands down into the 5 important feature patches, which then reconstruct all 30 patches at the bottom. The magnifier in the middle represents the self-attention layer. The 3 layers with 8 heads each reconstruct the full spectrum from the summary."  
"Panel 5: The decoder then takes this 512-d latent and reconstructs the original spectrum. The dashed line at the top is the original input for reference, and the solid cyan line is the reconstructed output. The checkmark shows the reconstruction matches the input — the autoencoder has faithfully learned to reproduce the spectrum. The model is trained with an L₁ reconstruction loss, which measures the absolute difference between input and output."  
"Transition: This reconstruction ability — learning to compress and decompress the spectrum — is what gives SpecPT its strong representations for downstream tasks like redshift prediction."

## Backup lines

- "So far I've talked about the science motivation; now I want to shift to the modeling approach."
- "This is where the earlier SpecPT work connects to the part I've been focusing on."
- "At first this looked like a normal scatter plot issue, but the pattern told us there was a real systematic failure."
- "That failure mode is what motivated adding photometry."
- "And beyond performance, we also wanted a way to understand what the model was using."

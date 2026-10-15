# HDSD evaluation protocol — final_v3

This protocol fixes evaluation of the completed unaugmented models on the existing partition. Freeze its machine-readable companion, `runs/final_v3/protocol.json`, and SHA256 before generating evaluation predictions. The JSON records every recording/reference, model configuration, selected checkpoint hash, software environment, and code identity. This document is part of that frozen record.

## Version 2 correction recorded before its inference

Version 1 was frozen and executed first. During its audit, the saved full checkpoints' suppression list was found to omit task-token IDs 50358 and 50359 that the pinned base model suppresses. The installed Transformers 4.57.1 Whisper implementation uses that list to construct its logits processor ([official source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/generation_whisper.py)). Version 2 loads the complete generation configuration from the same pinned base snapshot for every model, then applies the shared Hindi/task/greedy settings. Assert equality of the resolved decoder configuration across all models before inference and record its hash with each output. Rerun all seven model instances; do not silently relabel version 1 predictions. Preserve the complete first pass under `runs/final_v1/`, with its original protocol and document, as a disclosed preliminary decoding check.

This correction was motivated by decoder configuration inspection, not score optimization. It changes no labels, splits, weights, learning rates, seeds, or development-selected checkpoints. Version 1 scores had already been observed; both versions remain exploratory. The checkpoint selections retain their recorded development procedure and are not reconsidered using either evaluation pass.

## Version 3: forward the configuration through the adapter

A further adapter audit found that generic PEFT forwards `generate()` to the underlying base model ([official PEFT 0.17.1 source](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py)). Assigning a new generation-configuration object on the outer wrapper alone is insufficient proof that the base decoder uses it. Version 3 explicitly passes `generation_config=model.generation_config` in every generation call; its exact contents are checked against the frozen shared configuration first. A real PEFT forwarding check validates delivery of that argument to the base model without running training or speech generation. Version 2 was stopped during its first LoRA checkpoint; its completed groups and notebook/source records are preserved.

Only the twelve completed version-2 pretrained/full groups are eligible for compatible reuse. Those models already called Whisper directly with the same checked configuration object; explicit argument forwarding does not change that policy. Freeze their exact metadata/metric/prediction hashes in the version-3 JSON, verify matching model/data/group/generation identities, and copy identical predictions with explicit source-protocol provenance and `new_inference=False`. Their original generation times remain associated with their predictions. Do not reuse version-2 LoRA predictions. All three LoRA checkpoints are evaluated freshly under version 3. On a fresh reproduction without these compatible artifacts, the visible notebook code evaluates all seven model instances normally.

No old prediction or execution metadata is overwritten. Compatible reuse is not a new inference run or an independent replication. No score, label, model, split, seed, or epoch was selected using this amendment. All versions and comparisons remain exploratory.

## Question and status

Compare pretrained Whisper-small, full fine-tuning, and ordinary LoRA in recognition accuracy, held-out speaker/prompt behavior, and adaptation cost. All results on this partition are **exploratory fixed-holdout results**: the legacy test was inspected in earlier work. Newly generated results and a protocol recorded now do not make the original split an untouched confirmatory test. The comparisons are descriptive, not clinical assessments. No model is assumed to win.

## Data and inclusion

Use the frozen 1,882-pair manifest SHA256 `ab0acc16d98c3f0e979ed75fde88ea582b9e920ecc26e2863a5709a11ff92b6d`, M2 close-talk audio at 16 kHz, and the supplied Roman WORD annotations with the documented Hindi conversion. Preserve repetitions and omissions. The user accepted this annotation version; no new listening verification is claimed. Keep original review statuses and original HDSD files unchanged. No augmentation, clipping, label changes, or result-dependent exclusions are permitted in this version.

| Case | Manifest split | Recordings | Speakers | Prompts | Audio minutes |
|---|---|---:|---:|---:|---:|
| Both speaker and prompt unfamiliar; primary comparison | `test` | 45 | 9 | 5 | 2.895 |
| Unfamiliar speaker, familiar prompt; secondary | `test_speaker_train_prompt` | 189 | 9 | 21 | 9.508 |
| Familiar speaker, unfamiliar prompt; secondary | `train_speaker_test_prompt` | 213 | 41 | 5 | 13.073 |

These are 447 distinct M2 recordings. Training remains the existing 899 recordings, 41 speakers, and 21 prompts. Check exact recording/reference/audio identities, waveform separation from both training microphones, and the claimed speaker/prompt roles before inference. A paired microphone or repeated session is not an independent speaker. Do not pool the three cases into one headline score: speakers and sentence content differ. The two mixed development/test-role groups are outside this evaluation.

## Models and selection fixed before evaluation

- Pretrained `openai/whisper-small`, revision `973afd24965f72e36ca33b3055d56a652f456b4d`; evaluate once.
- Full fine-tuning, selected learning rate 5e-6; seeds 42, 123, 2026.
- LoRA, selected learning rate 5e-5; seeds 42, 123, 2026; rank 16, alpha 32, dropout 0.05, q/v projections.
- Both adaptations already completed five epochs with batch 4, accumulation 4, and BF16 training. Keep all other saved configuration fields fixed.
- Each seed uses its own best primary-development-WER checkpoint; equal WER retains the earlier epoch. Full best epochs are 1/1/2; LoRA best epochs are 3/1/5 for seeds 42/123/2026. Preserve those selections irrespective of evaluation scores.
- Learning-rate selection used primary development only; ties chose the smaller rate. Record selected checkpoint and training metadata hashes. Do not choose the best evaluation seed, retune, or retrain in this evaluation step.

## Transcription

Use all seven model instances on identical recording IDs in each case. Set Hindi transcription, no timestamps, deterministic greedy decoding (`do_sample=False`, `num_beams=1`), `max_new_tokens=96`, batch 8, FP32 model inference, no autocast, matmul TF32 disabled and cuDNN TF32 enabled. Set the correct Hindi/task/no-timestamps decoder prefix and clear `forced_decoder_ids` as in the validated development implementation. Use model evaluation mode and inference mode; create no optimizer and update no weights.

Pin the current cluster environment in the JSON. Report generation-cap counts; do not silently discard capped or poor predictions. An execution failure stops that group. A deliberate change to decoding, inclusion, annotations, or a checkpoint requires a separately recorded protocol version and a clear exploratory disclosure. Cache reuse requires identical protocol, model, membership, and prediction hashes. Complete each group atomically; partial groups do not count as completed evidence.

## Scores and model summaries

Score the exact saved CSV, not transient in-memory outputs. Apply NFC, replace Unicode punctuation with spaces, and collapse whitespace identically on references and predictions. Preserve meaningful Hindi spelling distinctions. Word error rate (WER) is 100 × (substitutions + deletions + insertions) / reference words, summed across recordings. Character error rate (CER) uses Unicode code points including spaces, not grapheme clusters. Save raw scores as sensitivity outputs. Edit alignment uses minimum distance with tie priority substitution, deletion, insertion; individual error-type allocations can depend on this declared tie rule.

Primary endpoint: normalized corpus WER on `test`. Report CER, word error counts, speaker-macro WER, per-speaker/per-prompt values, and utterance-level errors as supporting evidence. Report every seed. For each adapted method, average its three corpus WER/CER values and report sample standard deviation across seeds. Evaluate the pretrained baseline once; do not invent baseline seed variability or treat repeated copies as independent. Do not pool three seed predictions as three independent participant samples.

## Comparisons, uncertainty, and prompt sensitivity

Report full-minus-LoRA, pretrained-minus-full, and pretrained-minus-LoRA WER differences separately in every case. A positive difference favors the second model. These are exploratory comparisons with no multiplicity adjustment or confirmatory significance claim.

Use 20,000 paired speaker bootstrap samples, RNG seed 20261015, with replacement and as many sampled speakers as the original case. Average word-error counts over each method's three fixed trained seeds first. Resample identical speakers for both methods and retain all recordings/prompts from each sampled speaker; recompute the corpus-WER difference with the corresponding reference-word denominator. Report 2.5th/97.5th percentile bounds and the number of speakers helped, worsened, and tied based on seed-mean errors. Never resample individual utterances as independent speakers.

These intervals condition on this prompt set, split, and trained seed set; they do not jointly quantify training randomness, new prompts, all HDSD participants, or clinical populations. Show training-seed SD separately. Report leave-one-prompt-out WER differences to expose sentence sensitivity; these are descriptive sensitivity values, not confidence intervals. Cases share participants/prompts and are not independent replications. Stronger evidence needs separately predefined partitions and new adaptation runs, outside final_v3.

## Efficiency, artifacts, and ownership

Report trainable parameters, each run's recorded end-to-end adaptation time and peak allocated GPU memory, selected checkpoint bytes, and separately measured evaluation inference time. Adaptation times include model/data preparation, development generation, saving, and any recorded overhead; they are not pure training throughput or a controlled speed benchmark. A LoRA adapter requires the pinned base model; its bytes are not the complete deployable model size. Actual allocation is A100 MIG 3g.40gb, not the physical card's full 80 GB.

Write new outputs only under `runs/final_v3/`: one directory per model and case containing predictions, metrics, and execution metadata; `tables/` contains per-run/model, speaker, prompt, utterance-error, paired-comparison, prompt-sensitivity, efficiency, and source-hash records. Preserve all prior data, training, development predictions, histories, and checkpoints. Keep recordings, transcript-containing artifacts, and model sharing decisions subject to the supplied academic-use/non-redistribution agreement; this evaluation does not authorize publication of private data.

The existing three notebooks contain the full evaluation code. Their opt-in flag defaults to `False`. Notebook 01 evaluates the baseline; 02 the full checkpoints; 03 the LoRA checkpoints. Table export waits until all seven models are available. Record actual execution and failures in the project research log; code existing is not evidence of an experiment having run.

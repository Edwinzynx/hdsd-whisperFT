# Completed exploratory evaluation: final_v3

The completed comparison uses pretrained Whisper-small once and full/LoRA adaptation with seeds 42, 123, and 2026. Each model instance has predictions for the same 447 M2 recordings: 3,129 hypotheses across 21 model/case groups. Scores were independently checked against the privately retained predictions. Exporting this repository ran no training or new model inference.

WER is a percentage; lower is better. Adapted values are the mean ± sample standard deviation across the three training seeds. Seed standard deviation is not a confidence interval. The baseline has one evaluation and no invented seed variability.

| Evaluation case | Recordings | Pretrained WER | Full WER | LoRA WER |
|---|---:|---:|---:|---:|
| Both speaker and prompt unfamiliar | 45 | 71.51 | 63.04 ± 6.97 | 63.31 ± 3.18 |
| Unfamiliar speaker, familiar prompt | 189 | 93.03 | 17.65 ± 4.32 | 44.79 ± 12.63 |
| Familiar speaker, unfamiliar prompt | 213 | 75.17 | 65.53 ± 3.31 | 62.78 ± 1.74 |

The primary full-minus-LoRA difference is −0.28 percentage points, with paired speaker-bootstrap bounds [−10.67, +9.08]. This supports neither a clear primary-case ranking nor equivalence. Full has lower seed-mean WER for all nine unfamiliar speakers reading familiar prompts; its mean advantage there is 27.14 points. For familiar speakers reading unfamiliar prompts, LoRA's mean advantage is 2.75 points, with bounds that also span zero.

The paired bootstrap uses 20,000 resamples, RNG seed 20261015, identical sampled speakers for both methods, and word-error counts averaged over each method's fixed trained seeds. Its intervals condition on these prompts, this partition, and these fitted models. They do not jointly measure uncertainty from training randomness and new participants/prompts. Removing one primary prompt at a time changes the full-minus-LoRA difference from −6.51 to +3.44 points; this is prompt sensitivity, not a confidence interval.

## Selection and resources

Learning-rate searches used primary development WER, selecting 5e-6 for full adaptation and 5e-5 for LoRA. Both methods completed five epochs per selected seed. Each seed's checkpoint was selected using development WER, retaining the earlier epoch on ties. Full selected epochs 1/1/2 and LoRA 3/1/5 for seeds 42/123/2026.

Full adaptation updates 240,582,912 parameters; LoRA updates 1,769,472, approximately 0.74% as many. Across the three selected runs, recorded mean end-to-end adaptation time was 7.06 minutes for full and 7.27 for LoRA; mean peak allocated memory was 4.60 and 1.89 GiB, respectively. Timing includes preparation, development generation, and saving, and is not a controlled throughput benchmark. LoRA checkpoint bytes exclude the required pretrained base model. The actual allocation was A100 MIG 3g.40gb, not a full 80 GB device.

## Published numerical evidence

These CSVs are exact copies of the accepted aggregate tables; their hashes are recorded in [release.json](release.json).

| File | Contents |
|---|---|
| [model_summary.csv](results/model_summary.csv) | Nine method/case summaries: WER/CER mean and seed SD, and mean error counts. |
| [per_run.csv](results/per_run.csv) | All 21 model/case scores, seeds, substitution/deletion/insertion counts, and generation-cap counts. |
| [paired_comparisons.csv](results/paired_comparisons.csv) | Nine paired method comparisons, conditional speaker intervals, and counts of speakers helped/worsened/tied. |
| [efficiency.csv](results/efficiency.csv) | Parameters, recorded adaptation time/memory, selected snapshot bytes, and final inference time for seven model instances. |

The tables contain no utterance transcripts, recording paths, or participant-level rows. Exact references, predictions, checkpoints, manifests, and the machine-readable protocol are retained privately and are required for independent rescoring.

## Decoder audit and limitations

The first evaluation inherited different special-token suppression defaults from saved full checkpoints. Version 2 pinned the base generation configuration. A subsequent PEFT audit showed that the shared configuration also needed explicit forwarding through the adapter wrapper. Version 3 implements and checks that forwarding. Twelve correct direct baseline/full groups were reused with frozen version-2 provenance; all nine LoRA groups were generated freshly. No training, label, split, seed, learning rate, or selected epoch changed. Exact comparison found zero changed hypotheses across the 3,129 first-pass and accepted predictions. Earlier passes remain private audit evidence, not independent replications.

The legacy holdout had already been inspected, so all results are exploratory. The primary case has only nine speakers and five prompts. The three cases differ in participants and content; their differences are descriptive rather than a matched causal experiment. No listening verification or clinical severity labels were added. No data augmentation is included. Stronger generalization claims need additional predefined speaker/prompt partitions or external data.

Read [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) for the exact procedure and [README.md](README.md) for the distinction between running the public training workflow and replaying the private archived evaluation.

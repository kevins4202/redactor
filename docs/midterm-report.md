# Redactor: midterm report draft

Prepared 11 October 2026 for the proposal's 20 November milestone. This is a
working submission draft based on completed experiments, not a record that the
midterm was submitted or that a TA meeting occurred. No course-specific length
or formatting rubric has been supplied.

## Objective and approach

Redactor lets a user specify an object once, then propagates a mask and hides it
throughout a video. The proposal prioritizes avoiding exposed target pixels,
followed by keeping redaction tight. Manual outlines, class detectors, and text
descriptions share an instance prompt consisting of an ID, frame, and coarse
region. This makes segmentation and propagation replaceable and measurable.

The original baseline combines GrabCut with median Lucas–Kanade translation.
Its mask cannot adapt to scale and has no reliable mechanism for finding an
object after occlusion or a cut. We retained that implementation for comparison.

## Three implementation iterations

1. **Scale-aware motion and refinement.** Forward/backward-checked optical flow
   feeds a RANSAC similarity transform; periodic GrabCut refines the warped mask.
   This addresses shrinking subjects without changing the initial prompt.
2. **Recovery and monitoring.** A frozen histogram and multiscale appearance
   template search for the first-frame target when motion becomes uncertain.
   Cut monitoring, predicted motion, ambiguity rejection, and explicit review
   states replace silent mask freezing. An optional follow-up substitutes frozen
   ResNet features for the recovery search.
3. **Learned propagation and complete prompting.** Official SAM 2.1 Tiny is an
   alternative segment/propagate backend. Grounding DINO supplies face, plate,
   and text prompts; independent IDs support multiple objects, late arrivals,
   corrections, and manually confirmed target endings.

## Evaluation and findings

The starting repository did not contain the synthetic script referenced in the
proposal. We reconstructed four 90-frame scenarios: translation, shrinking to
45% scale, pillar occlusion, and a cut halfway through the clip. The fixture is
not claimed to reproduce the proposal's original Table 1. All methods use the
same initial prompt slack of 5, 9, or 13 pixels. No later ground-truth mask enters
propagation. IoU scores the raw predicted mask; leakage scores visible target
pixels outside the mask after the proposal's 9×9 dilation.

| Method | Synthetic mean IoU | Synthetic mean leakage |
|---|---:|---:|
| Translation baseline | 0.495 | 24.98% |
| Similarity motion + refinement | 0.639 | 24.36% |
| Classical + recovery | 0.812 | 4.68% |
| SAM 2.1 Tiny | 0.776 | 16.14% |

Means average all four scenarios and all three slack values. Similarity motion
helps scale change, but recovery is needed for the occlusion and cut scenarios.
SAM 2 performs poorly on the artificial occlusion fixture, so synthetic results
alone would give a misleading ranking.

Real-video evaluation includes eight convenience-selected DAVIS clips with
twelve annotated objects, plus all 150 crosswalk frames tracked and six
approximately traced frames scored. At five-pixel prompt slack:

| Method | DAVIS IoU | DAVIS leakage | Crosswalk IoU | Crosswalk leakage |
|---|---:|---:|---:|---:|
| Translation baseline | 0.298 | 53.98% | 0.402 | 39.02% |
| Classical + recovery | 0.310 | 47.67% | 0.246 | 70.96% |
| SAM 2.1 Tiny | 0.835 | 2.21% | 0.779 | 1.98% |

SAM 2 is the stronger real-footage option, but average overlap does not establish
privacy coverage. Its confidence monitor raises no flags on the crosswalk even
though the sparse labels show leakage. The classical tracker often requests
full-frame coverage; its resulting low guarded leakage comes at a large loss
of visible context. Propagation speed in the original DAVIS runs was a median
75.9 fps for translation, 7.2 fps for classical recovery, and 6.3 fps for SAM 2
on an RTX 4050 Laptop GPU. Loading and encoding are excluded from these timings.

## Follow-up experiments

The proposed crowded-frame box/lasso comparison now independently segments each
crosswalk annotation. Outline prompts reach mean IoU 0.806 versus 0.620 for the
original box initialization. A rectangle with the same seed construction as
lasso reaches 0.537 and covers more background, supporting a geometry advantage
in this clip. These prompts are generated from approximate labels; this is not
a human-effort or broad crowd benchmark.

Learned recovery is implemented but does not improve the aggregate DAVIS result:
mean IoU is 0.284 and raw leakage 52.59%, versus 0.310 and 47.67% for classical
recovery. This negative result is retained and the backend is experimental.

A paired coverage sweep leaves SAM 2 predictions unchanged and varies only the
redaction margin. Increasing the margin from 4 to 16 pixels reduces mean DAVIS
leakage from 2.21% to 0.74%, but increases excess frame coverage from 1.27% to
4.32%; the worst scored frame still leaks 25.15%. A larger margin cannot replace
identity recovery or complete review. Full tables, limitations, and reproduction
commands are in the [proposal follow-up](proposal-followup.md).

## Delivered workflow and remaining work

The local studio supports selection, an actual segmentation preview before
export, configurable coverage margins, progress/cancellation, corrective
prompts, target ending, video playback, and downloadable replay/review logs.
Preview is a fresh segmentation of the displayed frame, not propagation of
earlier memory. Export preserves audio when requested and only publishes a
successfully encoded video.

The implementation and experiments cover the proposal's planned mechanisms.
The general select-once, zero-leak objective is not achieved. Remaining research
includes reliable identity recovery, calibrated coverage warnings, independent
dense annotations, and domain-specific detector recall. The six crosswalk
labels cannot establish exposure duration, and the DAVIS subset is not a
randomized held-out evaluation. Course meetings, presentations, and submission
remain separate milestones for the team.

## Sources and reproducibility

- [Original proposal](../581proposal.pdf)
- [Final report and original results](final-report.md)
- [Benchmark commands and metric definitions](evaluation.md)
- [Follow-up results and raw data links](proposal-followup.md)
- [SAM 2 source](https://github.com/facebookresearch/sam2)
- [DAVIS benchmark](https://davischallenge.org/davis2017/code.html)
- [Grounding DINO source](https://github.com/IDEA-Research/GroundingDINO)
- [TorchVision ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

Development assistance: Codex was used for implementation, evaluation tooling,
analysis, and drafting this report. Team review and responsibility for the
submission remain necessary; this note is separate from Git authorship.

# Development iterations

This is a retrospective implementation record, not a claim that scheduled
midterm meetings or course presentations have occurred.

1. **Recover a measurable baseline.** Preserve original lasso/GrabCut/median
   translation, restore the historical flow-gate comparison, reconstruct the
   four synthetic stress tests, and distinguish IoU from leakage and excess
   redaction. Result: scale, occlusion, and cut failures can be measured rather
   than judged only by the appearance of a blurred video.
2. **Improve propagation and failure handling.** Add similarity RANSAC,
   periodic refinement, frozen appearance recovery, scene-cut monitoring,
   and explicit uncertainty. Result: large gains on controlled scale/cut tests,
   partial occlusion recovery, but poor generalization to many real clips.
3. **Compare learned tracking and complete the workflow.** Add official SAM 2,
   Grounding DINO, shared instance prompts, local review UI, corrections and
   target ending, checked audio/video export, and real-video benchmarks.
   Result: substantially better real-video coverage with SAM 2, while remaining
   edge leakage and unflagged errors justify mandatory human review.

Final measurements are in [results.md](results.md). The final report explains
the difference between implemented proposal deliverables and the unsolved
general no-leak objective.

## Proposal follow-up

Added the proposed frozen learned-feature recovery experiment, a paired
crowded-frame box/lasso ablation with an initialization control, actual browser
segmentation preview, and a coverage-margin sweep. Learned recovery worsened
aggregate DAVIS accuracy and remains experimental. Wider margins reduced
boundary leakage while covering more surroundings; they did not eliminate
worst-frame exposure. The [follow-up report](proposal-followup.md) preserves
these negative results and distinguishes completed implementation from the
open reliability objective. A [midterm draft](midterm-report.md) and
[presentation runbook](demo-runbook.md) are ready for team review.

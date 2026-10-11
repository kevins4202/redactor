# Project demonstration runbook

Prepared for rehearsal; the proposal's Final Project Day presentation has not
been performed or submitted by this workflow. Allow about four minutes.

## Before presenting

Run `python app.py`, open http://127.0.0.1:8765, and check that SAM 2 is available.
Keep the [12-second demo](../videos/redactor-demo/redactor-demo.mp4) and the
[follow-up figure](images/proposal-followup.png) available locally. A full export
can take over a minute on the measured laptop; use the prepared video if a live
export would exceed the presentation slot. The currently hosted player is
https://redactor-demo-production.up.railway.app and contains only the demo.

## Demonstration sequence

1. **Problem, 30 seconds.** Explain selecting a subject once and prioritizing
   missed pixels over mask tightness. Show the original crosswalk subject.
2. **Selection and preview, 45 seconds.** Outline the woman in the white skirt
   or draw a box around her. Choose SAM 2 and click Preview mask. Toggle Hide/Show
   mask. Explain that the strong green area is initial segmentation and the
   faint edge is extra coverage. It does not guarantee subsequent tracking.
3. **Result and correction, 60 seconds.** Play the prepared before/after video.
   Show the studio timeline and explain that drawing again at a later frame
   corrects the same target. End this target here records a reviewed departure;
   New target introduces another instance. Review flags are aids, not proof of
   zero leakage.
4. **Evidence, 60 seconds.** Show the original real-video comparison and the
   follow-up figure. SAM 2 improves mean DAVIS IoU to 0.835. Lasso improves crowded
   initial segmentation on the six labeled crosswalk frames. Learned recovery
   did not improve aggregate DAVIS accuracy and remains experimental.
5. **Limits, 45 seconds.** Show the margin trade-off: 4→16px lowers mean DAVIS
   leakage from 2.21% to 0.74%, while increasing excess coverage. A worst frame
   still leaks 25.15%. State that identity recovery, detector recall, and reliable
   warning calibration remain unresolved. Do not claim certified anonymity or
   a universally successful select-once workflow.

## Materials for the team

- [Midterm draft](midterm-report.md)
- [Final report](final-report.md)
- [Proposal status and follow-up results](proposal-followup.md)
- [Reproduction commands](evaluation.md)
- [Editable demo source and rendering notes](../videos/redactor-demo/README.md)

The team still needs to conduct the TA meeting, adapt the reports to any course
rubric, rehearse and give the presentation, and submit the requested artifacts.

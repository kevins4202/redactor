---
workflow: general-video
flow: automation
storyboard: no
message: "Select once, track locally, and review the redaction before sharing."
audience: "Computer vision course project reviewers"
destination: "Local MP4 demo accompanying the final report"
aspect: "16:9"
language: "English"
length: "30 seconds"
---

## Intent

The user asked to complete the project described in 581proposal.pdf, including its
final demo. Build an evidence-based technical comparison using the existing
crosswalk clip, real exported masks, and the project's measured results.

## Decisions

Autonomous production follows the user's instruction to complete the project.
The duration, landscape format, quiet presentation, and sage editorial design
are implementation choices. No narration or music is needed for the classroom
walkthrough; explanatory on-screen copy carries the story.

## Assets

Repository crosswalk footage, actual SAM 2 output, and real local studio capture.
Do not simulate an algorithmic result. Keep source and output aligned in time.

## Notes

Show manual selection, synchronized original/output playback, and the review
workflow. Close with the measured limitations: no guarantee of zero leakage,
and confidence does not detect every mistake. Use local assets and rendering.

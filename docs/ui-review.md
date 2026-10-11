# UI review

Reviewed locally in Chrome, desktop and 390 × 844 mobile viewport.

| Severity | Location | Before | After | Why |
|---|---|---|---|---|
| Resolved | Frame seeking | Drawing could use a previous image during a fast seek | Drawing requires the loaded image to match the selected frame | Keeps geometry tied to the correct frame |
| Resolved | Automatic-target review | New manual IDs could duplicate detected objects | Saved detector IDs populate target choices; corrections replace later detector prompts for that ID | Corrects the intended instance |
| Resolved | Target lifecycle | No explicit way to stop tracking a confirmed departure | End-target action with timeline validation | Avoids forcing a stale target through the rest of the video |
| Resolved | Outline download | JSON could refer to inaccessible PNG sidecars | Browser downloads the original polygon geometry | Makes saved selections portable |
| Resolved | Progress polling | Restarted sessions could retry forever | A missing job ends polling with a readable explanation | Gives a clear recovery path |
| Not verified | Browser file chooser automation | Extension denied file URL access | Left permissions unchanged; multipart upload passes integration tests | Separates an automation limitation from server validation |

Verified: rectangle selection, stepping, target ending/removal, export start,
cancellation, manual/automatic controls, full 150-frame SAM 2 solid export,
result playback, download endpoints, responsive layout without horizontal
overflow, source-pixel coordinate mapping, focus indicators, reduced-motion
styles, and minimum-size primary controls. A screenshot is in
[studio-desktop.jpg](images/studio-desktop.jpg); mobile evidence is in
[studio-mobile.jpg](images/studio-mobile.jpg).

The automatic-to-manual correction data flow was reviewed in code; a full
browser replay of that specific transition was not separately automated.
Model integration and multi-instance corrections are covered separately by
the backend tests. The pushed revision `ab4a9e8` passed Windows/Linux CI;
other browsers remain unverified.

Approve for the local research/demo scope. This approval is not a claim of
privacy certification or zero leaked pixels.

## Proposal follow-up review

Verified the new interface in Chrome against the real local backends:

- Two manual selections previewed through classical segmentation.
- A manual box previewed through SAM 2, including a 16px coverage margin.
- Grounding DINO text selection ("the woman in a white skirt") followed by SAM 2
  segmentation produced a one-target preview on frame 1.
- Hide/Show mask toggles the overlay; changing frame clears it and disables
  manual preview until a selection exists on that frame.
- Changing backend, text, or coverage margin invalidates the prior result;
  response-version checks discard previews superseded by edits.
- Export is disabled during preview. API tests verify concurrent requests are
  rejected and the model lock is released after a failed preview.
- At 390×844, the controls and preview status wrap without horizontal overflow
  (document scroll width 375px, viewport width 390px).

Browser testing exposed an existing selection-coordinate bug: focusing a
partially scrolled canvas could scroll it during pointer-down. Canvas focus now
uses `preventScroll`; its position was measured before and after drawing and
remained unchanged. The final screenshot shows the mask aligned with the woman
in the white skirt rather than the area above her.

Evidence: [SAM 2 preview](images/segmentation-preview.jpg) and
[mobile controls](images/segmentation-preview-mobile.jpg). The local suite passed
26 tests with optional model tests enabled, including exact SAM 2 first-frame
preview/export mask equality. A later-frame preview intentionally performs fresh
segmentation rather than reconstructing the previous tracking state.

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
the backend tests. Remote CI and other browsers remain not verified.

Approve for the local research/demo scope. This approval is not a claim of
privacy certification or zero leaked pixels.

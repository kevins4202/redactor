# Single-model pipeline

This replaces the earlier multi-model design proposal. The first version has one
manually selected target and one SAM 2.1 Base+ model.

1. Decode the clip and prepare quality-100 JPEG frames. Detect large shot changes.
2. Validate that all selections and corrections refer to one target.
3. For each shot with a selection, create a fresh model state and insert all of
   that shot's selections before propagation. Run forward from the earliest
   selection and backward to the shot start. Do not transfer state across cuts.
4. Save raw masks and boundary stability per frame. For unselected shots, save
   empty masks and request review rather than guess the target's identity.
5. Check empty masks, excessive mask area, unstable boundaries, and abrupt changes
   in size. These are review heuristics, not a calibrated identity verifier.
6. Expand masks slightly and render a solid mask over their foreground. Empty
   predictions never trigger full-frame cover. Keep source audio.
7. Publish a final video only if there are no flagged frames, or after explicit
   review of the preview. A correction starts a fresh analysis so future anchors
   can repair earlier predictions rather than retain stale tracking memory.

The browser exposes a box tool, selection preview, processing, correction,
review, and download. It does not expose model choice, detector choice, class/text
presets, recovery rules, coverage settings, or failure policies.

`sam_backend.py` owns model inference. `pipeline.py` owns analysis and publication.
`render.py` applies coverage. `video.py` handles decoding, shot-change checks, and
atomic video encoding. The server and browser only collect selections and present
results. Dataset readers and metrics remain available for validation. The old
classical trackers, learned recovery, detector association, and their obsolete
tests have been removed. Historical evaluation reports remain in Git.

There is no custom re-identification system. Same-model tracking can still switch
between similar objects or lose a target during long occlusion. The implementation
must earn any stronger claim through actual model runs and independent review;
application tests with test doubles only verify software behavior.

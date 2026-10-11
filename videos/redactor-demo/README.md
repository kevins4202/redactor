# Redactor demo

[Play the 30-second MP4](redactor-demo.mp4). The editable composition is
`index.html` plus the three files in `compositions/`.

```sh
npm run check
npm run dev
npm run render -- --quality delivery --output redactor-demo.mp4 --workers 2
python verify.py
```

The package scripts pin HyperFrames 0.8.149. GSAP 3.14.2 is bundled locally with
its license header. Source Serif 4 and Source Sans 3 are bundled with their open
font licenses. No music
or narration was added: the video is designed to accompany a live explanation.

`assets/studio.jpg` is a real capture of the local application. `original.mp4`
and `redacted.mp4` are synchronized left-side crops (1200 × 1080 source pixels,
resized to 800 × 720) from the repository's crosswalk clip and an actual SAM 2
solid-cover export. Both play at 0.5×, explicitly labeled on screen. Source
colors are unchanged. No simulated result or synthetic privacy claim is shown.

To reproduce the underlying redaction:

```sh
python redact.py data/crosswalk-short.mp4 --backend sam2 --prompts videos/redactor-demo/assets/prompts.json --style solid -o output/demo-solid.mp4
```

The browser capture used the hand-drawn box saved in `assets/prompts.json`.
The source/export hashes and prompt provenance are in `assets/provenance.json`.

Validation: HyperFrames check passed with no lint, runtime, layout, or contrast
findings; all scene midpoints were inspected. The animation-map helper reports
duplicate nested-timeline collision candidates; the actual timeline and
snapshots show independent scene slots and one progress bar, with no held
overlaps. The final encoded video is checked separately for duration, frame
count, and source/output synchronization.

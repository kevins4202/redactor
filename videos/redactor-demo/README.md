# Redactor demo

[Play the 12-second MP4](redactor-demo.mp4). The complete editable composition
is `index.html`.

```sh
npm run check
npm run dev
npm run render -- --quality delivery --output redactor-demo.mp4 --workers 2
python verify.py
```

The simplified cut uses one locally bundled font, Syne, with its open font
license. It contains only a wordmark, Before/After labels, and the closing line
Select. Track. Review. No UI screenshots, small captions, paragraphs, progress
bars, letter effects, music, or narration.

The package scripts pin HyperFrames 0.8.149. GSAP 3.14.2 is bundled with its
license header. The product-launch-video skill guided this revision; its
installation is recorded in the repository's `skills-lock.json`.

## Actual footage

The original five-second crosswalk plays first, followed by the same five
seconds from a real SAM 2 solid-cover export. Both play at native 1x speed.
The identical crops cover source x=0, y=540, width=1200, height=540, resized to
1680 x 756. Source colors and model output are unchanged.

To reproduce the underlying redaction:

```sh
python redact.py data/crosswalk-short.mp4 --backend sam2 --prompts videos/redactor-demo/assets/prompts.json --style solid -o output/demo-solid.mp4
```

For each original/export input, derive the demonstration asset with:

```sh
ffmpeg -i INPUT.mp4 -vf 'crop=1200:540:0:540,scale=1680:756' -an -c:v libx264 -crf 16 -g 30 -keyint_min 30 -movflags +faststart ASSET.mp4
```

The exact hand-drawn selection is saved in `assets/prompts.json`. Source,
export, font, and asset hashes are in `assets/provenance.json`. This example is
not a universal privacy guarantee; see the project's final report for measured
limitations.

## Verification

HyperFrames checks cover runtime, layout, and text contrast. Midpoints and
both sides of each cut are visually inspected before rendering. `verify.py`
checks the encoded duration, dimensions, every source/output frame in sequence,
and the stability of the title area. It writes `verification.json` and the
encoded contact sheet in `docs/images/demo-encoded.jpg`.

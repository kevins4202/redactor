from pathlib import Path

import cv2
import numpy as np


def make_frame(index, width=320, height=240):
    rng = np.random.default_rng(11)
    background = rng.integers(20, 70, (height, width, 3), dtype=np.uint8)
    gradient = np.linspace(0, 70, width, dtype=np.uint8)
    background[:, :, 1] = np.clip(
        background[:, :, 1].astype(np.uint16) + gradient[None, :],
        0,
        255,
    ).astype(np.uint8)
    center_x = 75 + 2 * index
    center_y = 115 + int(18 * np.sin(index / 12))
    cv2.circle(background, (center_x, center_y), 38, (35, 80, 235), -1)
    cv2.circle(background, (center_x - 12, center_y - 8), 6, (245, 245, 245), -1)
    cv2.circle(background, (center_x + 12, center_y - 8), 6, (245, 245, 245), -1)
    cv2.line(background, (center_x - 16, center_y + 14), (center_x + 16, center_y + 14), (20, 20, 20), 4)
    cv2.rectangle(background, (230, 45), (285, 100), (190, 90, 30), -1)
    return background


def main():
    output = Path("data/demo.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (320, 240),
    )
    for index in range(90):
        writer.write(make_frame(index))
    writer.release()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

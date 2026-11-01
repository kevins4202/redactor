"""Text selects a box; a separate video model supplies temporal masks."""
import cv2
from PIL import Image


def ground_target(frame, text):
    if not text.strip():
        raise ValueError("Describe the object to remove")
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError as error:
        raise RuntimeError('Install text selection with pip install -e .') from error
    name = 'IDEA-Research/grounding-dino-tiny'
    revision = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(name, revision=revision)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name, revision=revision).to(device).eval()
    query = text.strip().lower().rstrip('.')+'.'
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=query, return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=.30, text_threshold=.25,
        target_sizes=[image.size[::-1]])[0]
    if not len(result['boxes']):
        raise ValueError('No object matched the description; use a clearer description or a box')
    detections = [{'box': [float(x) for x in box], 'score': float(score)}
                  for box, score in zip(result['boxes'].cpu(), result['scores'].cpu())]
    detections.sort(key=lambda detection: detection['score'], reverse=True)
    return {'text': text, 'model': name, 'revision': getattr(model.config, '_commit_hash', None),
            'policy': 'highest scoring first-frame candidate; inspect selection preview',
            'selected': detections[0], 'candidates': detections}

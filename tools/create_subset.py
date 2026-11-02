from pathlib import Path
import random

source = Path.home() / "data/imagenet/train"
target = Path.home() / "data/imagenet-10k/train"
rng = random.Random(42)

classes = sorted(p for p in source.iterdir() if p.is_dir())
assert len(classes) == 1000, f"Expected 1000 classes, found {len(classes)}"

for cls in classes:
    images = sorted(p for p in cls.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert len(images) >= 10, f"Not enough images in {cls.name}"
    destination = target / cls.name
    destination.mkdir(parents=True, exist_ok=True)
    for img in rng.sample(images, 10):
        link = destination / img.name
        if not link.exists():
            link.symlink_to(img)

print("Created 10,000-image subset across 1,000 classes.")

"""Combine per-model per_block_energy plots into a single vertical stack (1 model per row)."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def combine_vertical(image_paths, labels, output_path):
    """Stack images vertically, one per row, with a label on the left."""
    imgs = [Image.open(p) for p in image_paths]

    # Uniform width, keep aspect ratio
    target_w = max(img.size[0] for img in imgs)
    label_w = 220  # left margin for model label

    # Resize all to same width
    resized = []
    for img in imgs:
        w, h = img.size
        if w != target_w:
            new_h = int(h * target_w / w)
            img = img.resize((target_w, new_h), Image.LANCZOS)
        resized.append(img)

    total_h = sum(img.size[1] for img in resized)
    canvas = Image.new("RGB", (label_w + target_w, total_h), "white")

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    y_offset = 0
    for img, label in zip(resized, labels):
        h = img.size[1]
        # Draw rotated label on the left
        text_y = y_offset + h // 2
        # Draw label lines (split iter pattern for readability)
        lines = label.split("\n")
        line_h = 24
        start_y = text_y - (len(lines) * line_h) // 2
        for j, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((label_w - tw) // 2, start_y + j * line_h), line, fill="black", font=font)

        canvas.paste(img, (label_w, y_offset))
        # Draw separator line
        draw.line([(0, y_offset), (label_w + target_w, y_offset)], fill="gray", width=1)
        y_offset += h

    canvas.save(output_path, dpi=(150, 150))
    print(f"  Saved: {output_path}  ({canvas.size[0]}x{canvas.size[1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="tools/bs/energy_landscape")
    args = parser.parse_args()

    base = Path(args.base_dir)
    plot_file = "per_block_energy_wikitext.png"

    for group_name in ["4_blocks", "4_blocks_dropout", "5_blocks", "5_blocks_dropout", "6_blocks", "8_blocks"]:
        group_dir = base / group_name
        if not group_dir.is_dir():
            continue

        model_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir() and "EGPT" in d.name])
        if not model_dirs:
            continue

        print(f"\n=== {group_name} ({len(model_dirs)} models) ===")

        paths = []
        labels = []
        for d in model_dirs:
            p = d / plot_file
            if not p.exists():
                print(f"  Missing: {p}")
                continue
            paths.append(p)
            # Extract short label
            name = d.name.replace("EGPT_", "").replace("_30k", "")
            if "ml_" in name:
                name = name.split("ml_")[1]
            elif name.startswith("4b_") or name.startswith("5b_"):
                name = name[3:]  # strip "4b_" or "5b_"
            else:
                name = "_".join(name.split("_")[2:])
            labels.append(name)

        if paths:
            combine_vertical(paths, labels, group_dir / f"combined_{plot_file}")


if __name__ == "__main__":
    main()

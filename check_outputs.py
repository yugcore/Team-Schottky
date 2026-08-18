import glob
import os
import numpy as np

files = glob.glob("./outputs/*.npy")

print(f"Output files: {len(files)}")

if not files:
    raise RuntimeError("No .npy files found in outputs/")

shapes = {}
global_min = float("inf")
global_max = float("-inf")
bad = []

for path in files:
    x = np.load(path)

    shapes[x.shape] = shapes.get(x.shape, 0) + 1

    if not np.isfinite(x).all():
        bad.append((os.path.basename(path), "NaN/Inf"))

    global_min = min(global_min, float(x.min()))
    global_max = max(global_max, float(x.max()))

print("\nShapes:")
for shape, count in shapes.items():
    print(f"  {shape}: {count}")

print(f"\nGlobal range: [{global_min}, {global_max}]")
print(f"Non-finite files: {len(bad)}")

if bad:
    print("\nBAD FILES:")
    for item in bad[:20]:
        print(" ", item)

print("\nFirst 5 outputs:")
for path in files[:5]:
    x = np.load(path)
    print(
        f"  {os.path.basename(path)} "
        f"shape={x.shape} "
        f"dtype={x.dtype} "
        f"range=[{x.min()}, {x.max()}]"
    )

if bad:
    raise SystemExit("\nFAILED: outputs contain NaN/Inf.")

print("\nPASSED: outputs are numerically valid.")
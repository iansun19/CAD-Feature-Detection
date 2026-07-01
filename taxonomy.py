"""
Canonical 25→12 machining-feature taxonomy for MFCAD++.

All downstream code must import mappings, names, colors, and descriptions from here.
New class ids are zero-indexed (0–11) for use with F.cross_entropy(logits, batch.y).
"""

from __future__ import annotations

NUM_CLASSES = 12

OLD_TO_NEW: dict[int, int] = {
    0: 9,   # chamfer
    1: 0,   # through_hole
    2: 1,   # poly_through_passage
    3: 1,
    4: 1,
    5: 2,   # through_slot
    6: 2,
    7: 2,
    8: 3,   # through_step
    9: 3,
    10: 3,
    11: 4,  # o_ring
    12: 5,  # blind_hole
    13: 6,  # blind_pocket
    14: 6,
    15: 6,
    16: 6,
    17: 7,  # blind_slot
    18: 7,
    19: 7,
    20: 8,  # blind_step
    21: 8,
    22: 8,
    23: 10,  # round_fillet
    24: 11,  # stock
}

NEW_NAMES: dict[int, str] = {
    0: "through_hole",
    1: "poly_through_passage",
    2: "through_slot",
    3: "through_step",
    4: "o_ring",
    5: "blind_hole",
    6: "blind_pocket",
    7: "blind_slot",
    8: "blind_step",
    9: "chamfer",
    10: "round_fillet",
    11: "stock",
}

NEW_DESCRIPTIONS: dict[int, str] = {
    0: (
        "Through hole: a single cylindrical face passing entirely through the part, "
        "open at both ends."
    ),
    1: (
        "Poly through passage: through opening with a polygonal cross-section "
        "(triangular, rectangular, or hexagonal planar walls), open at both ends."
    ),
    2: (
        "Through slot: channel cut fully across the part with triangular, rectangular, "
        "or circular profile, open at both ends and the top."
    ),
    3: (
        "Through step: L-shaped shoulder running fully across the part — including "
        "two-sided and slanted variants — open at both ends."
    ),
    4: (
        "O-ring: a circular ring groove (toroidal/cylindrical channel) recessed "
        "into a face."
    ),
    5: (
        "Blind hole: cylindrical hole that does NOT pass through — a cylinder wall "
        "plus a flat bottom."
    ),
    6: (
        "Blind pocket: closed pocket with triangular, rectangular, hexagonal, or "
        "rounded floor profile, open only at the top."
    ),
    7: (
        "Blind slot: slot closed at one end (floor and walls), open at the other "
        "end and the top; may terminate in a circular end face."
    ),
    8: (
        "Blind step: step closed at one end with triangular, circular, or "
        "rectangular profile."
    ),
    9: (
        "Chamfer: narrow planar bevel joining two faces at an oblique angle (~45°)."
    ),
    10: (
        "Round fillet: a rounded (cylindrical/toroidal) face blending two faces "
        "across a smooth/tangent edge."
    ),
    11: (
        "Stock: original raw-material outer surface — large planar faces forming "
        "the part's outer bounding box, typically convex neighbors."
    ),
}

# Saturated palette tuned for white 3D viewer background (WCAG-friendly contrast).
NEW_COLORS: dict[int, str] = {
    0: "#16A34A",   # through_hole — green
    1: "#2563EB",   # poly_through_passage — blue
    2: "#9333EA",   # through_slot — purple
    3: "#CA8A04",   # through_step — amber
    4: "#DB2777",   # o_ring — pink
    5: "#92400E",   # blind_hole — brown
    6: "#EA580C",   # blind_pocket — orange
    7: "#0891B2",   # blind_slot — cyan
    8: "#0369A1",   # blind_step — sky blue
    9: "#DC2626",   # chamfer — red
    10: "#B45309",  # round_fillet — bronze
    11: "#854D0E",  # stock — dark bronze
}


def old_to_new(old_id: int) -> int:
    """Map a legacy MFCAD++ label (0–24) to the collapsed class (0–11)."""
    if not 0 <= old_id <= 24:
        raise ValueError(f"old_id must be in 0–24, got {old_id}")
    return OLD_TO_NEW[old_id]


def new_name(new_id: int) -> str:
    """Return the canonical name for a collapsed class (0–11)."""
    return NEW_NAMES[new_id]


def validate() -> None:
    """Assert mapping completeness and that the image is exactly {0..11}."""
    expected_old = set(range(25))
    expected_new = set(range(NUM_CLASSES))

    assert set(OLD_TO_NEW.keys()) == expected_old, (
        f"OLD_TO_NEW keys {set(OLD_TO_NEW.keys())} != {expected_old}"
    )
    image = set(OLD_TO_NEW.values())
    assert image == expected_new, f"OLD_TO_NEW image {image} != {expected_new}"

    assert set(NEW_NAMES.keys()) == expected_new
    assert set(NEW_DESCRIPTIONS.keys()) == expected_new
    assert set(NEW_COLORS.keys()) == expected_new

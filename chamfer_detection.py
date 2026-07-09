"""Chamfer (edge-break) recognizer (Stage 1 cascade, additive).

A chamfer is a small FLAT (planar) face that bevels a sharp corner between two
larger faces -- geometrically distinct from a fillet, which is a CURVED (sphere/
torus/cylinder) blend that meets its neighbours tangentially. So the discriminator,
following the fillet recognizer's template-match style (surface type + edge
convexity/angle), is:

  * the face is planar and smaller than the faces it bridges;
  * it shares CONVEX edges with >= 2 distinct larger neighbours;
  * those edges sit in a bevel angle band (angle between the two face normals, the
    same signal step_ingest persists as cos on edge_attr) -- roughly 25deg..65deg for
    a typical 30/45/60 chamfer. This band deliberately excludes both ~0deg tangent
    blend edges (fillets) and ~90deg step edges (walls), and in particular the
    ~78.5deg planar facets inside 96260B's filleted lobes (which meet spheres/tori,
    not two flat faces at a bevel).

CAVEATS carried forward (cf. the fillet_radius_mm max-vs-min note):
  * Only CONVEX (external, edge-break) chamfers are detected. Concave/internal bevels
    are out of scope -- no test geometry exercises them.
  * `chamfer_size_mm` is a coarse footprint proxy (sqrt of face area), NOT the true
    chamfer leg length (edge lengths are not available to this pass). The bevel angle
    is exact; the size is approximate.
  * UNVERIFIED against a real chamfered part: 96260B has no chamfers (this recognizer
    correctly finds zero there). Positive detection is proven only by a synthetic
    fixture until a chamfered part exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# Bevel band on the angle between the two adjacent face normals (degrees). A 45deg
# chamfer on a 90deg corner puts each of its edges at 45deg; 30/60 chamfers stay
# inside 25..65. Below the band -> tangent blend (fillet); above -> step wall.
CHAMFER_MIN_ANGLE_DEG = 25.0
CHAMFER_MAX_ANGLE_DEG = 65.0
# A chamfer must bevel a corner, i.e. share bevel edges with at least this many
# distinct neighbour faces.
CHAMFER_MIN_BRIDGED_FACES = 2


@dataclass
class ChamferFeature:
    """One recognized chamfer face."""

    face_index: int
    chamfer_angle_deg: float
    area_mm2: float
    bridged_faces: list[int]

    @property
    def face_indices(self) -> frozenset[int]:
        return frozenset({self.face_index})

    @property
    def chamfer_size_mm(self) -> float:
        # Coarse footprint proxy; NOT the true chamfer leg length (see module note).
        return round(math.sqrt(self.area_mm2), 3) if self.area_mm2 > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.face_index,
            "chamfer_angle_deg": round(self.chamfer_angle_deg, 2),
            "chamfer_size_mm": self.chamfer_size_mm,
            "area_mm2": round(self.area_mm2, 4),
            "bridged_faces": sorted(self.bridged_faces),
            "has_chamfer": True,
            "toolpath_class": "chamfer",
        }


@dataclass
class ChamferResult:
    features: list[ChamferFeature] = field(default_factory=list)

    @property
    def claimed_faces(self) -> frozenset[int]:
        return frozenset(f.face_index for f in self.features)


def _external_edges(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> dict[int, list[tuple[int, float, bool]]]:
    """face_index -> [(neighbour_index, angle_between_normals_deg, is_convex)]."""
    ei = np.asarray(edge_index)
    ea = np.asarray(edge_attr)
    cos_col = ea[:, 3] if ea.shape[1] >= 4 else ea[:, -1]
    by_face: dict[int, list[tuple[int, float, bool]]] = {}
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i == j:  # boundary / self-loop
            continue
        ang = math.degrees(math.acos(float(np.clip(cos_col[k], -1.0, 1.0))))
        is_convex = int(np.argmax(ea[k, :3])) == 1  # cols: [concave, convex, smooth]
        by_face.setdefault(i, []).append((j, ang, is_convex))
    return by_face


def detect_chamfers(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    exclude_faces: frozenset[int] = frozenset(),
    min_angle_deg: float = CHAMFER_MIN_ANGLE_DEG,
    max_angle_deg: float = CHAMFER_MAX_ANGLE_DEG,
) -> ChamferResult:
    """Recognize convex edge-break chamfers among unclaimed planar faces."""
    area_by_index = {int(f.index): float(getattr(f, "area", 0.0) or 0.0) for f in faces}
    by_face = _external_edges(edge_index, edge_attr)

    result = ChamferResult()
    for f in faces:
        idx = int(f.index)
        if idx in exclude_faces or getattr(f, "surface_type", None) != "plane":
            continue

        bevel = [
            (j, ang)
            for (j, ang, convex) in by_face.get(idx, [])
            if convex and min_angle_deg <= ang <= max_angle_deg
        ]
        bridged = sorted({j for j, _ in bevel})
        if len(bridged) < CHAMFER_MIN_BRIDGED_FACES:
            continue

        # A chamfer strip is smaller than the faces it bevels.
        bridged_areas = [area_by_index.get(j, 0.0) for j in bridged]
        if not bridged_areas or area_by_index.get(idx, 0.0) >= min(bridged_areas):
            continue

        result.features.append(
            ChamferFeature(
                face_index=idx,
                chamfer_angle_deg=float(np.mean([ang for _, ang in bevel])),
                area_mm2=area_by_index.get(idx, 0.0),
                bridged_faces=bridged,
            )
        )
    return result

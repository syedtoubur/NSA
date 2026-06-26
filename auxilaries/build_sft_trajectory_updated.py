"""
sft_trajectory_builder.py
==========================
Fixed / hardened version of `build_sft_trajectory`.

This module turns a single (input_grid, output_grid, trans_dict) triple -
produced by `sample_and_apply` in grid_transformation.py - into a textual
"Thinking with Visual Primitives" trajectory that can be used as:
  1. Supervised cold-start (SFT) training data.
  2. A ground-truth replay used to *validate* that the textual trajectory
     actually reproduces the target grid (see `verified` field below).

-----------------------------------------------------------------------
BUGS FOUND IN THE ORIGINAL `build_sft_trajectory` AND HOW THEY'RE FIXED
-----------------------------------------------------------------------

1. Malformed closing tag in the grid-based branch:
   `</|box|Layout>` -> should be `</|box|>`. This silently broke the
   primitive syntax for *every* grid-based transformation (connect,
   magnet, mirror_grid, upscale_grid, crop, beam, rotate_grid, recolor,
   shift, truncate, rotate_duplicate, fill, arbitrary_duplicate) -
   roughly half of all possible transformations.

2. Dead / confusing ternary when computing the initial bounding box:
       max(p[1] for p in sp) if 'sp' in locals() else max(p[1] for p in sub_pixels)
   `sp` is a variable from a *later* section of the function and is
   never bound yet at this point, so this always silently fell through
   to the `else` branch. It "worked" by accident but is fragile and
   confusing. Replaced with a direct, explicit min/max computation.

3. **Most important correctness bug**: `adjust_to_bounding_box` (called
   `canvas_changed` in the original) was *inferred* by comparing
   `grid.shape != end_grid.shape`. But the actual NSA pipeline
   (`modify_grid` in grid_transformation.py) decides this purely from
   *which transformation op is used* (see `TRANSFORMATIONS_NO_ADJUST`
   below) - completely independent of whether the shapes happen to
   differ. Because `undo_abstraction(adjust_to_bounding_box=...)`
   dispatches to two structurally different reconstruction routines
   (`undo_abstraction1` crops to the bounding box of all surviving
   pixels; `undo_abstraction2` re-paints onto the original canvas),
   picking the wrong one can silently produce a grid that does *not*
   match `end_grid`, even when dimensions coincidentally match.
   Fixed by reusing the same `TRANSFORMATIONS_NO_ADJUST` rule the data
   generator itself uses.

4. `extract` and `duplicate` were called inside the generic
   "one node at a time" loop:
       transformation_fn(info["node_key"], **transformation_params)
   But `ARCGraph.extract(self, node, ...)` expects `node` to be the
   *entire list* of surviving node keys (it is called once, not per
   node), and `ARCGraph.duplicate(self, axis=..., ...)` takes **no**
   node argument at all - it is a whole-graph operation. Calling them
   per-node either corrupts the graph (extract) or raises a
   `TypeError` / silently mis-binds `axis` (duplicate). Both are now
   special-cased exactly like `modify_grid` does.

5. Enum parameters (`Direction.LEFT`, `Rotation.CW`, ...) were
   interpolated into the primitive text with the default `f"{v}"`
   formatting, which for a plain `Enum` renders `"Direction.LEFT"`,
   not the clean `"LEFT"` shown in the desired schema. A dedicated
   `_format_param_value` now renders enums via `.name`, so the
   produced text is `direction=LEFT`, and - critically - this is also
   what `execution_engine.py` expects when parsing it back.

6. Non-deterministic ordering of newly-created nodes: the original
   code iterated `for nk in final_nodes_in_graph` where
   `final_nodes_in_graph = set(...)`. Set iteration order is not part
   of the language guarantee for arbitrary tuple keys, so two runs of
   the *same* mutation could print "Object_New_1" / "Object_New_2" in
   a different order. Fixed by sorting new nodes by bounding box
   before printing (consistent with how the initial objects are
   ordered).

7. Object IDs were assigned in spatial (bounding-box) order, but the
   filter+mutation loop in the *real* pipeline iterates nodes in the
   abstraction's native graph order (e.g. color-major for `nbccg`),
   not spatial order. For order-sensitive transforms (anything that
   uses `check_collision`, e.g. `move_node_max`, `move_node`,
   `extend_node`), mutating nodes in a different order than the
   canonical pipeline can - in principle - produce a different final
   grid for the same `trans_dict`. This version mutates nodes in the
   graph's native order (faithful replay) while still *presenting*
   them with spatially-ordered IDs (left-to-right, matching the
   paper's box-ordering convention) for readability.

8. Multicolor nodes (abstraction `na`, `mcccg`) store `color` as a
   *list*, one entry per pixel. The original f-string would dump the
   raw Python list inline (`Color: [1, 1, 2, 2, 2, ...]`). Replaced
   with `_summarize_color`, which collapses this to a clean summary.

9. Removed a stray `print(selected_nodes)` debug statement.

10. Added a `verified` field: after reconstructing `output_grid`, it is
    compared cell-by-cell against the caller-supplied `end_grid`. This
    mirrors the paper's own "rigorous verification mechanism... to
    eliminate noise during cold-start training" (Sec 2.4.1). Any
    sample where `verified=False` should be dropped by the data
    generator rather than silently kept.

-----------------------------------------------------------------------
EXTENSION: graph-grounded object tracking for grid-based transformations
-----------------------------------------------------------------------
11. Branch A (the 13 whole-grid functions in GRID_BASED_TRANSFORMATIONS)
    previously emitted only a single Input/Output Canvas box -- no
    per-object visual primitives at all, unlike Branch B. This meant
    roughly half of all transformations carried no real "Thinking with
    Visual Primitives" signal. Branch A now:
      - segments the input grid into objects via the SAME graph-based
        abstraction Branch B uses ('nbccg': single-color, 4-connected,
        background-excluded components from ARCGraph/Image), giving each
        one a stable Object_N id (Step 1), and
      - independently re-segments the actual `output_grid` the same way,
        then solves a shape-correspondence problem (translation + the 8
        D4 symmetries + uniform integer upscale, with color compared
        separately) to recognize which output object "is" which input
        object -- even if it moved, rotated/mirrored, was upscaled,
        recolored, or duplicated into several copies (Step 4).
    This intentionally segments BOTH grids fresh rather than threading
    bookkeeping through 13 unrelated black-box pixel functions: it is the
    generic, transformation-agnostic analogue of how a human re-parses an
    image rather than running a hard-coded rule for "what counts as an
    object" (see Sec. 1 of the paper on the human, top-down/bottom-up
    nature of object segmentation). It is best-effort, not an oracle:
    objects that a transformation fuses or erases beyond recognition are
    reported as [DELETED] / [CREATED], which is itself an honest
    description, not a tracking failure. Crucially, none of this touches
    how `output_grid` is computed -- it is still the exact same call
    `modify_grid` makes -- so `verified` cannot regress because of it.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from image import Image
from ARCGraph import ARCGraph

# These are real, lightweight, dependency-free re-implementations of the
# two helpers that the original snippet pulled in from
# `auxilaries.grid_transformation`. Keeping them local means this module
# has zero dependency on the heavy plotting / LLM-prompt imports that
# grid_transformation.py drags in (matplotlib, PIL, llm.selector_prompt,
# plots, etc.) which are irrelevant to trajectory construction.
try:
    # Prefer the user's real project layout if it is on the path.
    from auxilaries.grid_transformation import (
        graph_to_grid,
        connect_grid_based, magnet_grid_based, mirror_grid_based,
        upscale_grid_based, crop_grid_based, beam_grid_based,
        rotate_grid_based, recolor_grid_based, shift_grid_based,
        truncate_grid_based, rotate_duplicate_grid_based,
        fill_grid_based, arbitrary_duplicate_grid_based,
    )
except ImportError:
    try:
        from grid_transformation import (
            graph_to_grid,
            connect_grid_based, magnet_grid_based, mirror_grid_based,
            upscale_grid_based, crop_grid_based, beam_grid_based,
            rotate_grid_based, recolor_grid_based, shift_grid_based,
            truncate_grid_based, rotate_duplicate_grid_based,
            fill_grid_based, arbitrary_duplicate_grid_based,
        )
    except ImportError:
        from extended_transformations.connect_grid import connect_grid_based
        from extended_transformations.magnet_grid import magnet_grid_based
        from extended_transformations.mirror_grid import mirror_grid_based
        from extended_transformations.upscale_grid import upscale_grid_based
        from extended_transformations.crop_grid import crop_grid_based
        from extended_transformations.beam_grid import beam_grid_based
        from extended_transformations.rotate_grid import rotate_grid_based
        from extended_transformations.recolor_grid import recolor_grid_based
        from extended_transformations.shift_grid import shift_grid_based
        from extended_transformations.truncate_grid import truncate_grid_based
        from extended_transformations.rotate_duplicate import rotate_duplicate_grid_based
        from extended_transformations.fill_grid import fill_grid_based
        from extended_transformations.arbitrary_duplicate_grid import arbitrary_duplicate_grid_based

        def graph_to_grid(graph, width, height, background_color):
            grid = [[background_color for _ in range(width)] for _ in range(height)]
            for node, data in graph.nodes(data=True):
                nodes = data.get("nodes", [node])
                color = data["color"]
                if not isinstance(color, list):
                    color = [color] * len(nodes)
                for (y, x), c in zip(nodes, color):
                    if 0 <= y < height and 0 <= x < width:
                        grid[y][x] = c
            return grid


GRID_BASED_TRANSFORMATIONS = {
    "connect": connect_grid_based, "magnet": magnet_grid_based, "mirror_grid": mirror_grid_based,
    "upscale_grid": upscale_grid_based, "crop": crop_grid_based, "beam": beam_grid_based,
    "rotate_grid": rotate_grid_based, "recolor": recolor_grid_based, "shift": shift_grid_based,
    "truncate": truncate_grid_based, "rotate_duplicate": rotate_duplicate_grid_based,
    "fill": fill_grid_based, "arbitrary_duplicate": arbitrary_duplicate_grid_based,
}

# Exactly mirrors `transformations_no_adjust` inside `modify_grid` in
# grid_transformation.py. This is the SINGLE SOURCE OF TRUTH for whether
# undo_abstraction should crop-to-bounding-box (True) or repaint onto the
# original canvas (False) -- it must stay in sync with that list.
TRANSFORMATIONS_NO_ADJUST = {
    "move_node", "move_node_max", "update_color", "extend_node",
    "rotate_node", "add_border", "fill_rectangle", "hollow_rectangle",
    "mirror", "flip", "insert", "remove_node", "extract",
}

# Whole-graph ops that take no per-node argument at all.
GRAPH_LEVEL_NODE_OPS = {"duplicate"}
# Ops that operate on the *entire filtered set* in a single call rather
# than being looped over one node at a time.
BATCH_NODE_OPS = {"extract"}


def _format_param_value(v: Any) -> str:
    """Render a parameter value the way it should appear inside
    <transform>op(key=value, ...)</transform>, and the way
    execution_engine.py expects to parse it back."""
    if isinstance(v, Enum):
        return v.name
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_format_param_value(x) if x is not None else "None" for x in v) + ")"
    return str(v)


def _param_str(params: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={_format_param_value(v)}" for k, v in params.items())


def _summarize_color(color: Any) -> str:
    """Multicolor nodes (abstraction 'na' / 'mcccg') store color as a
    per-pixel list. Collapse that into something readable instead of
    dumping the raw list."""
    if isinstance(color, list):
        unique = sorted(set(color))
        if len(unique) == 0:
            return "none"
        if len(unique) == 1:
            return str(unique[0])
        return "multi" + str(unique)
    return str(color)


def _node_box(node_data: dict, node_key) -> Tuple[int, int, int, int]:
    """Returns (xmin, ymin, xmax, ymax) for a node's current pixels."""
    pixels = node_data.get("nodes", [node_key])
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return (min(xs), min(ys), max(xs), max(ys))


# =====================================================================
# EXTENSION: graph-grounded object tracking for grid-based ("Branch A")
# transformations.
#
# Why this exists: hard-coding "object = whatever a particular ARCGraph
# abstraction op happens to carve out" is not how humans look at an ARC
# grid. People re-segment the *pixels* fresh every time and then track
# shapes across the before/after pair -- a bottom-up grouping that gets
# corrected/confirmed by whatever top-down hypothesis they're holding.
# The 13 functions in GRID_BASED_TRANSFORMATIONS (see
# extended_grid_based_transformations.py) are opaque whole-grid pixel
# functions with no notion of "node" at all -- there is nothing for
# `modify_grid` to hand us. So instead of threading transformation-specific
# bookkeeping through every one of those 13 functions, we do the human
# thing: segment BOTH the raw input grid and the raw output grid
# independently -- using the *same* graph-based abstraction (ARCGraph
# 'nbccg': single-color, 4-connected components, background=0 excluded)
# that Branch B already relies on for genuine node-based transformations
# -- and then solve a shape-correspondence problem between the two
# segmentations to recover which output object "is" which input object,
# even if it moved, rotated/mirrored, was uniformly upscaled, was
# recolored, or was duplicated one-to-many. This is purely descriptive
# narration layered ON TOP of an already-correct replay: `output_grid`
# itself is still produced by the exact same call `modify_grid` makes, so
# nothing here can ever cause `verified` to go from True to False.
#
# Limitation (documented rather than hidden): this is a best-effort,
# generic shape-matcher, not a transformation-aware oracle. Operations
# that fundamentally fuse/erase pixel groups (e.g. `connect`, `beam`,
# `fill`, `magnet`, `truncate`, `crop`) can legitimately produce content
# that this matcher cannot trace back to a single input object -- in
# those cases the object is reported as [DELETED] (consumed) on the input
# side and/or [CREATED] (newly drawn) on the output side, which is itself
# an honest and human-plausible description of what happened.
# =====================================================================

# Safety valve: above this many tracked objects on either side, the
# all-pairs shape-correspondence search below is both too slow and far
# too verbose to be a useful training signal, so we silently fall back to
# the original canvas-only narration instead of attempting per-object
# tracking.
MAX_OBJECTS_FOR_GRID_TRACKING = 40

# The 8 symmetries of the square (dihedral group D4), expressed as
# transforms over a translation-normalized pixel shape (top-left at 0,0).
_D4_OPS = [
    "identity", "rot90", "rot180", "rot270",
    "flip_v", "flip_h", "flip_diag", "flip_antidiag",
]
_D4_LABELS = {
    "identity": None,  # no extra wording needed for a pure translation
    "rot90": "rotated 90\u00b0 clockwise",
    "rot180": "rotated 180\u00b0",
    "rot270": "rotated 90\u00b0 counter-clockwise",
    "flip_v": "mirrored horizontally",
    "flip_h": "mirrored vertically",
    "flip_diag": "mirrored along the main diagonal",
    "flip_antidiag": "mirrored along the anti-diagonal",
}
# Generous upper bound on uniform block-upscale factor a single object can
# be matched under (covers upscale_grid's factor=2..4 with margin).
_MAX_UPSCALE_FACTOR = 5


def _normalize_shape(pixels) -> frozenset:
    """Pixel coordinates -> shape with translation removed (bbox top-left
    re-anchored to (0, 0)). Two objects have "the same shape" iff their
    normalized shapes are equal."""
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    min_y, min_x = min(ys), min(xs)
    return frozenset((y - min_y, x - min_x) for y, x in pixels)


def _d4_transform(shape: frozenset, op: str) -> frozenset:
    """Apply one of the 8 D4 symmetries to a normalized shape, then
    re-normalize the result back to a (0, 0)-anchored frame."""
    if op == "identity":
        pts = shape
    elif op == "rot90":            # whole-shape 90deg CLOCKWISE rotation
        pts = {(dx, -dy) for dy, dx in shape}
    elif op == "rot180":
        pts = {(-dy, -dx) for dy, dx in shape}
    elif op == "rot270":           # == 90deg COUNTER-clockwise
        pts = {(-dx, dy) for dy, dx in shape}
    elif op == "flip_v":           # left-right mirror
        pts = {(dy, -dx) for dy, dx in shape}
    elif op == "flip_h":           # top-bottom mirror
        pts = {(-dy, dx) for dy, dx in shape}
    elif op == "flip_diag":        # transpose (main diagonal)
        pts = {(dx, dy) for dy, dx in shape}
    elif op == "flip_antidiag":
        pts = {(-dx, -dy) for dy, dx in shape}
    else:
        raise ValueError(f"Unknown D4 op: {op}")
    min_y = min(p[0] for p in pts)
    min_x = min(p[1] for p in pts)
    return frozenset((y - min_y, x - min_x) for y, x in pts)


def _scale_shape(shape: frozenset, k: int) -> frozenset:
    """Uniformly upscale a normalized shape by integer factor k (each
    pixel becomes a k x k block) -- mirrors what `upscale_grid` does to a
    single object's pixels."""
    if k == 1:
        return shape
    pts = set()
    for dy, dx in shape:
        base_y, base_x = dy * k, dx * k
        for i in range(k):
            for j in range(k):
                pts.add((base_y + i, base_x + j))
    return frozenset(pts)


def _match_shape(in_shape: frozenset, out_shape: frozenset) -> Optional[Tuple[str, int]]:
    """Return the (op, scale) pair that maps `in_shape` onto `out_shape`
    (translation already removed from both) via a D4 symmetry plus an
    optional uniform integer upscale, or None if no such relationship
    exists. Since scale is isotropic, in_size * scale**2 == out_size has
    AT MOST one positive-integer solution, so there is at most one scale
    worth testing -- we solve for it directly rather than searching.
    When multiple D4 ops match at that scale (possible for symmetric
    shapes), 'identity' is preferred over rotations/mirrors for the
    simplest possible narration; this preference never affects whether a
    match exists, only which explanation is reported."""
    in_size, out_size = len(in_shape), len(out_shape)
    if in_size == 0 or out_size % in_size != 0:
        return None
    ratio = out_size // in_size
    scale = round(ratio ** 0.5)
    if scale < 1 or scale > _MAX_UPSCALE_FACTOR or scale * scale != ratio:
        return None

    best = None
    for op in _D4_OPS:
        transformed = _d4_transform(in_shape, op)
        if scale > 1:
            transformed = _scale_shape(transformed, scale)
        if transformed == out_shape:
            if best is None or _D4_OPS.index(op) < _D4_OPS.index(best):
                best = op
    if best is None:
        return None
    return (best, scale)


def _segment_objects(grid: List[List[int]]) -> List[Dict[str, Any]]:
    """Segment a raw grid into foreground objects using the SAME
    graph-based abstraction machinery (ARCGraph / Image, 'nbccg') that
    Branch B uses for real node-based transformations -- i.e. single-color,
    4-connected components, with background (color 0) excluded.

    This is deliberately independent of whatever `trans_dict['abstraction']`
    happens to be: in `sample_and_apply`, the abstraction is sampled
    *before* it is known whether a grid-based or a node-based
    transformation will be chosen, so for grid-based ops that field is
    unrelated noise, not a meaningful signal about how to segment objects.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    if height == 0 or width == 0:
        return []
    img = Image(task=None, grid=grid, width=width, height=height)
    seg_graph = getattr(img, Image.abstraction_ops["nbccg"])()
    objects = []
    for node, data in seg_graph.graph.nodes(data=True):
        pixels = data.get("nodes", [node])
        if not pixels:
            continue
        xmin, ymin, xmax, ymax = _node_box(data, node)
        objects.append({
            "pixels": frozenset(pixels),
            "shape": _normalize_shape(pixels),
            "color": data.get("color", 0),
            "box": (xmin, ymin, xmax, ymax),
        })
    return objects


def _match_objects(
    input_objects: List[Dict[str, Any]], output_objects: List[Dict[str, Any]]
) -> Tuple[Dict[int, Dict[str, Any]], "set[int]"]:
    """Solve the input<->output object correspondence problem so that
    moved / rotated / mirrored / upscaled / recolored / duplicated objects
    are recognized as "the same object" -- per the requirement that even
    if an object lands elsewhere on the output grid, we still know it's
    the same object -- while still allowing ONE input object to
    legitimately explain MULTIPLE output objects (duplication).

    An output object may match at most one input object; an input object
    may match zero, one, or many output objects.

    Returns:
      assignments: {output_index: {"input_index", "op", "scale", "same_color"}}
        for every output object that was matched to some input object.
        Output objects absent from this dict have no explaining input
        object (i.e. newly created content).
      unmatched_input_indices: set of input indices matched to ZERO output
        objects (i.e. deleted / consumed content).
    """
    candidates_per_output: Dict[int, List[Dict[str, Any]]] = {}
    for oi, out_obj in enumerate(output_objects):
        cand_list = []
        for ii, in_obj in enumerate(input_objects):
            match = _match_shape(in_obj["shape"], out_obj["shape"])
            if match is None:
                continue
            op, scale = match
            in_box, out_box = in_obj["box"], out_obj["box"]
            in_cx = (in_box[0] + in_box[2]) / 2.0
            in_cy = (in_box[1] + in_box[3]) / 2.0
            out_cx = (out_box[0] + out_box[2]) / 2.0
            out_cy = (out_box[1] + out_box[3]) / 2.0
            dist = ((in_cx - out_cx) ** 2 + (in_cy - out_cy) ** 2) ** 0.5
            cand_list.append({
                "input_index": ii,
                "op": op,
                "scale": scale,
                "same_color": in_obj["color"] == out_obj["color"],
                "dist": dist,
            })
        candidates_per_output[oi] = cand_list

    # Deterministic, left-to-right / top-to-bottom processing order so
    # that ties are broken reproducibly run-to-run (same convention as
    # the paper's left-to-right box-listing order).
    output_order = sorted(range(len(output_objects)), key=lambda oi: output_objects[oi]["box"])

    assignments: Dict[int, Dict[str, Any]] = {}
    used_input_counts: Dict[int, int] = defaultdict(int)
    for oi in output_order:
        cand_list = candidates_per_output[oi]
        if not cand_list:
            continue  # genuinely new content -> reported as [CREATED]
        # Preference order: same color first, then the simplest geometric
        # relation (smallest upscale factor, then identity over any
        # rotation/mirror), then nearest spatial match to disambiguate
        # ties between multiple identical-shape candidates (e.g. several
        # interchangeable copies of the same sprite).
        cand_list.sort(key=lambda c: (
            0 if c["same_color"] else 1,
            c["scale"],
            _D4_OPS.index(c["op"]),
            c["dist"],
        ))
        best = cand_list[0]
        assignments[oi] = best
        used_input_counts[best["input_index"]] += 1

    unmatched_input_indices = {
        ii for ii in range(len(input_objects)) if used_input_counts[ii] == 0
    }
    return assignments, unmatched_input_indices


def _describe_relation(op: str, scale: int, same_color: bool, out_color: int) -> List[str]:
    """Human-readable fragments describing how a matched output object
    relates to its source input object, beyond plain translation."""
    parts = []
    if scale > 1:
        parts.append(f"upscaled \u00d7{scale}")
    label = _D4_LABELS.get(op)
    if label:
        parts.append(label)
    if not same_color:
        parts.append(f"recolored to {out_color}")
    return parts


def grids_equal(g1: List[List[int]], g2: List[List[int]]) -> bool:
    if len(g1) != len(g2):
        return False
    if len(g1) and len(g1[0]) != len(g2[0]):
        return False
    return all(g1[r][c] == g2[r][c] for r in range(len(g1)) for c in range(len(g1[0])))


def build_sft_trajectory(grid, end_grid, trans_dict) -> Dict[str, Any]:
    """
    Executes a transformation program over an input grid and builds a
    step-by-step "Thinking with Visual Primitives" trajectory.

    Returns a dict with keys:
      - trajectory:        the full <user>/<assistant> text block
      - output_grid:       the grid actually reconstructed by replaying
                            the transformation (should equal `end_grid`)
      - nodes_transformed:  number of nodes the transformation touched
      - verified:           True iff output_grid == end_grid exactly
    """
    transformation_name = trans_dict["transformation"]
    transformation_params = dict(trans_dict.get("transformation_params", {}))
    param_str = _param_str(transformation_params)

    # ---------------------------------------------------------------
    # BRANCH A: GLOBAL GRID-BASED TRANSFORMATIONS
    # ---------------------------------------------------------------
    if transformation_name in GRID_BASED_TRANSFORMATIONS:
        output_grid = GRID_BASED_TRANSFORMATIONS[transformation_name](grid, **transformation_params)

        if not output_grid or not isinstance(output_grid[0], (list, tuple)):
            # Defensive guard: a handful of grid-based ops can return None
            # or otherwise malformed output for certain parameter
            # combinations (e.g. upscale_grid_based's "standard" branch in
            # extended_grid_based_transformations.py ends with a bare
            # `return` instead of `return upscaled_grid` for some inputs --
            # a pre-existing bug in that module, not introduced here).
            # Rather than crashing the whole generation run on one bad
            # sample, fall back to the unchanged input grid; `verified`
            # will then correctly come out False below and the caller
            # should drop the sample -- mirroring how `modify_grid` itself
            # already falls back to `grid` on failure.
            output_grid = [row[:] for row in grid]

        in_h, in_w = len(grid), len(grid[0])
        out_h = len(output_grid)
        out_w = len(output_grid[0]) if out_h else 0
        dims_changed = (in_h, in_w) != (out_h, out_w)

        # --- Graph-grounded object segmentation (see EXTENSION note above
        #     `grids_equal`). Re-segmenting both grids from scratch, rather
        #     than threading per-transformation bookkeeping through 13
        #     unrelated black-box pixel functions, is what lets Branch A
        #     get the same box-primitive treatment as Branch B. ---
        input_objects = _segment_objects(grid)
        output_objects = _segment_objects(output_grid)
        can_track = (
            0 < len(input_objects) <= MAX_OBJECTS_FOR_GRID_TRACKING
            and len(output_objects) <= MAX_OBJECTS_FOR_GRID_TRACKING
        )

        if not can_track:
            # Fallback: the original canvas-only narration. Triggered when
            # there are no discrete foreground objects to track at all, or
            # there are too many for per-object correspondence to be a
            # useful (and fast) training signal.
            step1_text = (
                "1. Parse the visual primitives present in the input domain:\n"
                f"   - Input Canvas: <|box|>[[0, 0, {in_w - 1}, {in_h - 1}]]</|box|>"
            )
            if dims_changed:
                step2_text = (
                    "2. Evaluate the spatial variations and operational rules:\n"
                    f"   - The canvas is resized from {in_w}x{in_h} to {out_w}x{out_h} "
                    "by a global, macro-structural transformation."
                )
            else:
                step2_text = (
                    "2. Evaluate the spatial variations and operational rules:\n"
                    "   - The canvas dimensions stay fixed; the global grid environment "
                    "triggers a macro-structural transformation across all pixels."
                )
            step3_text = (
                "3. Execute transformations over the tracked functional IDs:\n"
                f"   <transform>{transformation_name}({param_str})</transform>"
            )
            step4_text = (
                "4. Map out the resulting visual primitives for the final output domain configuration:\n"
                f"   - Output Canvas: <|box|>[[0, 0, {out_w - 1}, {out_h - 1}]]</|box|>\n"
                "   - Transformed canvas layout matches the target configuration matrix."
            )
            nodes_transformed = 1

        else:
            # --- Step 1: input objects, presented left-to-right / top-to-
            #     bottom (paper convention), each given a stable Object_N id
            #     that Step 4 will reference even if the object moves,
            #     duplicates, or changes color/orientation. ---
            presentation_in = sorted(
                range(len(input_objects)), key=lambda i: input_objects[i]["box"]
            )
            in_id = {idx: f"Object_{rank}" for rank, idx in enumerate(presentation_in, start=1)}

            step1_lines = [
                "1. Parse the visual primitives present in the input domain:",
                f"   - Input Canvas: <|box|>[[0, 0, {in_w - 1}, {in_h - 1}]]</|box|>",
            ]
            for idx in presentation_in:
                xmin, ymin, xmax, ymax = input_objects[idx]["box"]
                step1_lines.append(
                    f"   - {in_id[idx]}: <|box|>[[{xmin}, {ymin}, {xmax}, {ymax}]]</|box|> "
                    f"(Color: {_summarize_color(input_objects[idx]['color'])})"
                )
            step1_text = "\n".join(step1_lines)

            # --- Step 2 ---
            step2_lines = ["2. Evaluate the spatial variations and operational rules:"]
            if dims_changed:
                step2_lines.append(
                    f"   - The canvas is resized from {in_w}x{in_h} to {out_w}x{out_h} "
                    "by a global, macro-structural transformation."
                )
            else:
                step2_lines.append(
                    "   - The canvas dimensions stay fixed; the global grid environment "
                    "triggers a macro-structural transformation across all pixels."
                )
            step2_lines.append(
                f"   - {len(input_objects)} discrete object(s) detected under non-background "
                "connectivity. This is a whole-grid operation -- it is not anchored to any "
                "single tracked object ahead of execution; object-level effects are recovered "
                "by re-parsing the visual primitives present after the transformation runs."
            )
            step2_text = "\n".join(step2_lines)

            # --- Step 3 (unchanged: a single, whole-grid call -- exactly
            #     what `modify_grid` does for these 13 operations) ---
            step3_text = (
                "3. Execute transformations over the tracked functional IDs:\n"
                f"   <transform>{transformation_name}({param_str})</transform>"
            )

            # --- Step 4: re-segment the OUTPUT independently and recover
            #     object correspondence (moved / rotated / mirrored /
            #     upscaled / recolored / duplicated / deleted / created). ---
            assignments, unmatched_input = _match_objects(input_objects, output_objects)

            matches_per_input: Dict[int, List[int]] = defaultdict(list)
            for oi, m in assignments.items():
                matches_per_input[m["input_index"]].append(oi)
            for ii in matches_per_input:
                matches_per_input[ii].sort(key=lambda oi: output_objects[oi]["box"])

            step4_lines = [
                "4. Map out the resulting visual primitives for the final output domain configuration:",
                f"   - Output Canvas: <|box|>[[0, 0, {out_w - 1}, {out_h - 1}]]</|box|>",
            ]

            changed_count = len(unmatched_input)
            for idx in presentation_in:
                obj_id = in_id[idx]
                copies = matches_per_input.get(idx, [])
                if not copies:
                    step4_lines.append(f"   - {obj_id}: [DELETED]")
                    continue
                in_box = input_objects[idx]["box"]
                for copy_rank, oi in enumerate(copies, start=1):
                    m = assignments[oi]
                    out_box = output_objects[oi]["box"]
                    out_color = output_objects[oi]["color"]
                    label = obj_id if copy_rank == 1 else f"{obj_id}_dup{copy_rank}"
                    moved = in_box[:2] != out_box[:2]
                    fragments = _describe_relation(m["op"], m["scale"], m["same_color"], out_color)

                    status_parts = []
                    if copy_rank > 1:
                        status_parts.append(f"Duplicate of {obj_id}")
                    elif moved:
                        status_parts.append(
                            "Moved from <|box|>"
                            f"[[{in_box[0]}, {in_box[1]}, {in_box[2]}, {in_box[3]}]]</|box|>"
                        )
                    status_parts.extend(fragments)
                    status = "; ".join(status_parts) if status_parts else "Unchanged"
                    if status != "Unchanged":
                        changed_count += 1

                    step4_lines.append(
                        f"   - {label}: <|box|>[[{out_box[0]}, {out_box[1]}, {out_box[2]}, {out_box[3]}]]</|box|> "
                        f"({status})"
                    )

            # New, unexplained output content (e.g. lines drawn by `connect`
            # / `beam`, regions painted by `fill`, padding from `crop`, ...).
            created_indices = [oi for oi in range(len(output_objects)) if oi not in assignments]
            created_sorted = sorted(created_indices, key=lambda oi: output_objects[oi]["box"])
            for created_count, oi in enumerate(created_sorted, start=1):
                xmin, ymin, xmax, ymax = output_objects[oi]["box"]
                step4_lines.append(
                    f"   - Object_New_{created_count}: <|box|>[[{xmin}, {ymin}, {xmax}, {ymax}]]</|box|> "
                    f"(Color: {_summarize_color(output_objects[oi]['color'])}) [CREATED]"
                )
            changed_count += len(created_sorted)

            step4_text = "\n".join(step4_lines)
            nodes_transformed = changed_count

    # ---------------------------------------------------------------
    # BRANCH B: LOCALIZED NODE / OBJECT-BASED TRANSFORMATIONS
    # ---------------------------------------------------------------
    else:
        img = Image(task=None, grid=grid, width=len(grid[0]), height=len(grid))
        abstraction_name = trans_dict["abstraction"]

        if abstraction_name not in Image.abstraction_ops:
            raise ValueError(f"Unknown abstraction operation: {abstraction_name}")
        arcgraph: ARCGraph = getattr(img, Image.abstraction_ops[abstraction_name])()

        # Canonical (graph-native) node order -- THIS is the order the
        # real pipeline (modify_grid) iterates filtered nodes in, and the
        # order we must replay mutations in to faithfully reproduce
        # `end_grid` for order-sensitive (collision-based) transforms.
        canonical_order = list(arcgraph.graph.nodes())

        initial_node_info = []
        for node in canonical_order:
            node_data = arcgraph.graph.nodes[node]
            xmin, ymin, xmax, ymax = _node_box(node_data, node)
            initial_node_info.append({
                "node_key": node,
                "box": (xmin, ymin, xmax, ymax),
                "color": node_data.get("color", 0),
            })

        # Presentation order: left-to-right, top-to-bottom (paper
        # convention: "bounding boxes are ordered from left to right").
        # This is *only* for assigning human-readable Object_N IDs and
        # printing the listing -- it is decoupled from canonical_order,
        # which is what's actually used to run filters/transformations.
        presentation_order = sorted(initial_node_info, key=lambda x: x["box"])
        for idx, info in enumerate(presentation_order, start=1):
            info["id"] = f"Object_{idx}"
        node_to_id = {info["node_key"]: info["id"] for info in initial_node_info}

        if not initial_node_info:
            # Degenerate case: abstraction found no objects at all.
            step1_text = (
                "1. Parse the visual primitives present in the input domain:\n"
                f"   - Input Canvas: <|box|>[[0, 0, {len(grid[0]) - 1}, {len(grid) - 1}]]</|box|>\n"
                "   - No discrete objects detected under this abstraction."
            )
            output_grid = [row[:] for row in grid]
            return {
                "trajectory": _assemble_trajectory(
                    grid, output_grid, step1_text,
                    "2. Evaluate the spatial variations and operational rules:\n   - Nothing to transform.",
                    "3. Execute transformations over the tracked functional IDs:\n   (No transformations executed)",
                    "4. Map out the resulting visual primitives for the final output domain configuration:\n"
                    f"   - Output Canvas: <|box|>[[0, 0, {len(output_grid[0]) - 1}, {len(output_grid) - 1}]]</|box|>",
                ),
                "output_grid": output_grid,
                "nodes_transformed": 0,
                "verified": grids_equal(output_grid, end_grid),
            }

        step1_lines = [
            "1. Parse the visual primitives present in the input domain:",
            f"   - Input Canvas: <|box|>[[0, 0, {len(grid[0]) - 1}, {len(grid) - 1}]]</|box|>",
        ]
        for info in presentation_order:
            xmin, ymin, xmax, ymax = info["box"]
            step1_lines.append(
                f"   - {info['id']}: <|box|>[[{xmin}, {ymin}, {xmax}, {ymax}]]</|box|> "
                f"(Color: {_summarize_color(info['color'])})"
            )
        step1_text = "\n".join(step1_lines)

        # --- Filter handling (evaluated once, on the pristine graph,
        #     in canonical order -- exactly like modify_grid does) ---
        filter_name = trans_dict["filter"]
        filter_fn = getattr(arcgraph, filter_name)
        filter_params = trans_dict.get("filter_params", {})

        selected_nodes = [
            info for info in initial_node_info
            if filter_fn(info["node_key"], **filter_params)
        ]
        selected_keys = {info["node_key"] for info in selected_nodes}

        step2_lines = ["2. Evaluate the spatial variations and operational rules between the input and output states:"]
        # Present these lists in spatial (Object_N) order for readability.
        transformed_ids = [info["id"] for info in presentation_order if info["node_key"] in selected_keys]
        static_ids = [info["id"] for info in presentation_order if info["node_key"] not in selected_keys]

        if transformation_name in BATCH_NODE_OPS:
            # "extract": selected nodes are KEPT, the rest are discarded --
            # phrase this as keep/discard rather than the misleading
            # "remain static" (those objects don't survive at all).
            if transformed_ids:
                step2_lines.append(f"   - {', '.join(transformed_ids)} are kept (matched the filter).")
            if static_ids:
                step2_lines.append(f"   - {', '.join(static_ids)} are discarded (did not match the filter).")
        elif transformation_name in GRAPH_LEVEL_NODE_OPS:
            step2_lines.append(
                "   - This is a whole-graph operation; it is not anchored to any single tracked object."
            )
        else:
            if static_ids:
                step2_lines.append(f"   - {', '.join(static_ids)} remain static.")
            if transformed_ids:
                step2_lines.append(f"   - {', '.join(transformed_ids)} undergo transformation.")
            else:
                step2_lines.append("   - All elements remain completely static.")
        step2_text = "\n".join(step2_lines)

        # --- Apply mutations ---
        step3_lines = ["3. Execute transformations over the tracked functional IDs:"]

        if transformation_name in GRAPH_LEVEL_NODE_OPS:
            # e.g. "duplicate": whole-graph operation, no node argument.
            transformation_fn = getattr(arcgraph, transformation_name)
            step3_lines.append(f"   <transform>{transformation_name}({param_str})</transform>")
            transformation_fn(**transformation_params)

        elif transformation_name in BATCH_NODE_OPS:
            # e.g. "extract": takes the *entire* list of surviving node
            # keys in one call, not one node at a time.
            transformation_fn = getattr(arcgraph, transformation_name)
            node_keys = [info["node_key"] for info in selected_nodes]
            ids_str = ", ".join(info["id"] for info in presentation_order if info["node_key"] in selected_keys)
            step3_lines.append(
                f"   <transform>{transformation_name}(ids=[{ids_str}]"
                f"{', ' + param_str if param_str else ''})</transform>"
            )
            if node_keys:
                transformation_fn(node_keys, **transformation_params)

        else:
            # Standard per-node ops (move_node_max, update_color, ...).
            # IMPORTANT: print the <transform> lines in the SAME order
            # they are executed (canonical/graph-native order) -- not
            # presentation order. If print-order and execution-order ever
            # diverged, a model that faithfully learned "replay my actions
            # in the order I wrote them" would reproduce a *different*
            # grid than the one baked into end_grid whenever the
            # transformation is order-sensitive (e.g. move_node_max /
            # move_node / extend_node, which use check_collision and can
            # therefore block each other depending on processing order).
            # Object_N *labels* are still assigned in spatial order above
            # (matches the paper's left-to-right box-listing convention);
            # only the sequence of <transform> lines changes here.
            transformation_fn = getattr(arcgraph, transformation_name)
            for info in selected_nodes:  # canonical order
                step3_lines.append(
                    f"   <transform>{transformation_name}(id={info['id']}"
                    f"{', ' + param_str if param_str else ''})</transform>"
                )
                transformation_fn(info["node_key"], **transformation_params)

        if not selected_nodes and transformation_name not in GRAPH_LEVEL_NODE_OPS:
            step3_lines.append("   (No transformations executed)")
        step3_text = "\n".join(step3_lines)

        # --- Reconstruct the grid ---
        # THE KEY FIX: which undo_abstraction variant to use is decided
        # by the transformation name (matching modify_grid), not by
        # comparing shapes after the fact.
        adjust_to_bounding_box = transformation_name not in TRANSFORMATIONS_NO_ADJUST
        modified_arcgraph = arcgraph.undo_abstraction(adjust_to_bounding_box=adjust_to_bounding_box)
        if modified_arcgraph is None:
            output_grid = [[0 for _ in range(len(end_grid[0]))] for _ in range(len(end_grid))]
        else:
            output_grid = graph_to_grid(
                modified_arcgraph.graph, len(end_grid[0]), len(end_grid), background_color=0
            )

        # --- Describe post-execution lifecycle state ---
        final_nodes_in_graph = set(arcgraph.graph.nodes())

        all_remaining_pixels = []
        for node in final_nodes_in_graph:
            all_remaining_pixels.extend(arcgraph.graph.nodes[node].get("nodes", [node]))

        if all_remaining_pixels and adjust_to_bounding_box:
            global_min_y = min(p[0] for p in all_remaining_pixels)
            global_min_x = min(p[1] for p in all_remaining_pixels)
        else:
            global_min_y = 0
            global_min_x = 0

        final_objects_info = []
        for info in presentation_order:
            nk, obj_id = info["node_key"], info["id"]
            if nk not in final_nodes_in_graph:
                final_objects_info.append(f"   - {obj_id}: [DELETED]")
            else:
                nd = arcgraph.graph.nodes[nk]
                xmin, ymin, xmax, ymax = _node_box(nd, nk)
                is_mutated = (xmin, ymin, xmax, ymax) != info["box"] or nd.get("color", 0) != info["color"]
                status = "Mutated" if is_mutated else "Unchanged"
                fx_min, fy_min = xmin - global_min_x, ymin - global_min_y
                fx_max, fy_max = xmax - global_min_x, ymax - global_min_y
                final_objects_info.append(
                    f"   - {obj_id}: <|box|>[[{fx_min}, {fy_min}, {fx_max}, {fy_max}]]</|box|> ({status})"
                )

        # Newly created nodes (e.g. from `insert`): sort deterministically
        # by bounding box instead of relying on raw set iteration order.
        new_nodes = [nk for nk in final_nodes_in_graph if nk not in node_to_id]
        new_nodes_sorted = sorted(
            new_nodes, key=lambda nk: _node_box(arcgraph.graph.nodes[nk], nk)
        )
        for created_count, nk in enumerate(new_nodes_sorted, start=1):
            nd = arcgraph.graph.nodes[nk]
            xmin, ymin, xmax, ymax = _node_box(nd, nk)
            fx_min, fy_min = xmin - global_min_x, ymin - global_min_y
            fx_max, fy_max = xmax - global_min_x, ymax - global_min_y
            final_objects_info.append(
                f"   - Object_New_{created_count}: <|box|>[[{fx_min}, {fy_min}, {fx_max}, {fy_max}]]</|box|> "
                f"(Color: {_summarize_color(nd.get('color', 0))}) [CREATED]"
            )

        step4_lines = [
            "4. Map out the resulting visual primitives for the final output domain configuration:",
            f"   - Output Canvas: <|box|>[[0, 0, {len(output_grid[0]) - 1}, {len(output_grid) - 1}]]</|box|>",
        ] + final_objects_info
        step4_text = "\n".join(step4_lines)
        nodes_transformed = len(selected_nodes)

    trajectory = _assemble_trajectory(grid, output_grid, step1_text, step2_text, step3_text, step4_text)
    return {
        "trajectory": trajectory,
        "output_grid": output_grid,
        "nodes_transformed": nodes_transformed,
        "verified": grids_equal(output_grid, end_grid),
    }


def _assemble_trajectory(grid, output_grid, step1_text, step2_text, step3_text, step4_text) -> str:
    def format_grid(g):
        return "[" + ",\n ".join(str(row) for row in g) + "]"

    return (
        "<user>\n"
        "Given the input ARC grid and its target output grid, determine the pattern rules and execute the correct cell mutations to produce the output grid.\n"
        f"Input Grid:\n{format_grid(grid)}\n"
        f"Output Grid:\n{format_grid(output_grid)}\n"
        "</user>\n\n"
        "<assistant>\n"
        "Thinking Process:\n"
        f"{step1_text}\n\n"
        f"{step2_text}\n\n"
        f"{step3_text}\n\n"
        f"{step4_text}\n\n"
        "Execution completed. The output matrix matches the target layout.\n"
        "</assistant>"
    )

"""Plan four collision slots from an explicit request or one target-time map scan.

This CPU adapter never loads a model. Times are relative to source frame zero;
the generator's checkpoint supports 1..9 seconds on its 0.5-second token grid.
"""
from __future__ import annotations

import math

import numpy as np

from scan_collision_opportunities import COLLISION_TYPES, RoadGraph, ScanConfig, analyze_frame


def canonical_type(value):
    value = str(value).replace("-", "_")
    if value not in COLLISION_TYPES:
        raise ValueError(f"Unknown collision type: {value}")
    return value.replace("_", "-")


def resolve_mode(mode=None, requested_type=None):
    if requested_type == "any":
        requested_type = None
        if mode not in (None, "any"):
            raise ValueError("requested type 'any' conflicts with manual/auto mode")
        mode = "any"
    mode = mode or ("manual" if requested_type else "any")
    if mode not in ("any", "manual", "auto"):
        raise ValueError(f"Unknown type mode: {mode}")
    if mode != "manual" and requested_type is not None:
        raise ValueError("An explicit collision type requires manual mode")
    return mode, canonical_type(requested_type) if requested_type else None


def plan_scene(scene, mode="any", requested_type=None, collision_time_seconds=None,
               scan_config=None):
    """Return exactly four slots, including slots with no detected opportunity.

    Empty/invalid auto detection is a rejected budget, not an unrestricted retry.
    Manual cut-in obeys the explicit 2026-11-02 intersection exclusion. Other
    manual requests and unrestricted mode retain their original input policy.
    """
    mode, requested_type = resolve_mode(mode, requested_type)
    if mode == "manual" and requested_type is None:
        raise ValueError("Manual mode requires a CLI or manifest collision type")
    ego = scene["tracks"][str(scene["metadata"]["sdc_id"])]["state"]
    positions = np.asarray(ego["position"])
    timestamps = np.asarray(scene["metadata"].get("ts", np.arange(len(positions)) * .1), float)
    if timestamps.ndim != 1 or len(timestamps) != len(positions) or not np.isfinite(timestamps).all():
        raise ValueError("Source timestamps must match the ego trajectory")
    relative = timestamps - timestamps[0]
    # Native WOMD timestamps can drift by 44ms over 91 frames. The existing
    # checkpoint uses frame/10; retain original states and report the real ts.
    if (np.any(np.diff(timestamps) <= 0)
            or not np.allclose(relative, np.arange(len(positions)) * .1, atol=.05, rtol=0)):
        raise ValueError("The checkpoint requires source frames at 10 Hz")
    seconds = (len(positions) - 1) / 10 if collision_time_seconds is None else float(collision_time_seconds)
    if not math.isfinite(seconds) or not 1 <= seconds <= 9 or not math.isclose(seconds * 2, round(seconds * 2), abs_tol=1e-8, rel_tol=0):
        raise ValueError("Collision time must be 1..9 seconds in 0.5-second increments")
    frame = int(round(seconds * 10))
    if frame >= len(positions):
        raise ValueError("Collision time exceeds the source trajectory")
    heading = np.asarray(ego["heading"]).reshape(-1)
    valid = np.asarray(ego["valid"]).reshape(-1)
    if len(heading) != len(positions) or len(valid) != len(positions):
        raise ValueError("Ego state arrays have inconsistent lengths")
    usable = bool(valid[frame]) and bool(np.isfinite(positions[frame, :2]).all()) and bool(np.isfinite(heading[frame]))
    scan = None
    if not usable:
        status, available = "invalid_target_ego", []
    elif mode == "auto":
        scan = analyze_frame(RoadGraph(scene.get("map_features", {})), positions[frame, :2],
                             float(heading[frame]), scan_config or ScanConfig())
        available = [canonical_type(k) for k in COLLISION_TYPES if k in scan["potential_collision_types"]]
        status = "ready" if available else "no_local_collision_opportunity"
    else:
        available = [requested_type] if mode == "manual" else []
        status = "ready"
        config = scan_config or ScanConfig()
        if mode == 'manual' and requested_type == 'cut-in' and config.cut_in_intersection_buffer_m is not None:
            scan = analyze_frame(RoadGraph(scene.get('map_features', {})), positions[frame, :2],
                                 float(heading[frame]), config)
            if scan.get('cut_in_intersection_guard', {}).get('target_blocked', False):
                status = 'cut_in_excluded_near_intersection'
    if mode == "manual":
        available = [requested_type]
    slots = []
    occurrences = {}
    for index in range(4):
        kind = available[index % len(available)] if available else None
        ordinal = occurrences.get(kind, 0)
        occurrences[kind] = ordinal + 1
        slots.append(dict(candidate_slot=index, requested_collision_type=kind,
                          type_candidate_index=ordinal, status=status))
    return dict(scenario_id=str(scene.get("id", scene.get("metadata", {}).get("scenario_id", ""))),
                type_mode=mode, collision_time_seconds=seconds, target_frame=frame,
                source_timestamp_seconds=float(relative[frame]),
                candidate_budget=4, status=status, available_collision_types=available,
                ego_position=positions[frame, :2].tolist(), ego_heading=float(heading[frame]),
                local_check=scan, slots=slots)


def apply_type_requirement(metrics, requested_type):
    """Only unrestricted inputs omit type adherence; keep every physical gate."""
    result = dict(metrics)
    if requested_type is None:
        result["rejection_reasons"] = [r for r in metrics["rejection_reasons"]
                                       if r != "requested_collision_type_mismatch"]
        result["accepted"] = not result["rejection_reasons"]
    result["requested_type"] = requested_type
    result["type_adherence_required"] = requested_type is not None
    return result

"""Function: Scan nine local road structures and potential collision types every 0.5s.
Input: ScenarioNet scene dictionaries/pickles, or a scene CSV and dataset directory.
Output: Standalone visualization-ready JSON timelines; no generation/model side effects.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re

import numpy as np


ROAD_TYPES = {
    1: ("one_way_single_lane", "单向单车道"),
    2: ("one_way_multi_lane", "单向多车道"),
    3: ("two_way_road", "双向道路"),
    4: ("curve", "弯道"),
    5: ("merge_diverge_weave", "汇入／分流／交织区"),
    6: ("lane_gain_loss", "车道增减路段"),
    7: ("intersection", "各类交叉口"),
    8: ("roundabout", "环岛"),
    9: ("u_turn", "掉头区域"),
}
COLLISION_TYPES = ("rear_end", "cut_in", "side", "head_on")


@dataclass(frozen=True)
class ScanConfig:
    interval_s: float = 0.5
    search_radius_m: float = 40.0
    conflict_radius_m: float = 12.0
    ego_lane_distance_m: float = 4.0
    ego_heading_tolerance_deg: float = 70.0
    ego_match_slack_m: float = 0.8
    adjacent_distance_m: float = 6.0
    opposing_distance_m: float = 8.0
    same_direction_deg: float = 30.0
    opposing_direction_deg: float = 150.0
    curve_turn_deg: float = 25.0
    u_turn_deg: float = 150.0
    # None explicitly reproduces the pre-2026-11-02 opportunity policy.
    cut_in_intersection_buffer_m: float | None = 5.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values() if v is not None):
            raise ValueError("scan parameters must be finite and positive")
        if self.conflict_radius_m > self.search_radius_m:
            raise ValueError("conflict radius must not exceed search radius")
        if not self.same_direction_deg < self.opposing_direction_deg <= 180:
            raise ValueError("invalid direction thresholds")


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def angle_deg(a, b):
    return float(abs(np.rad2deg(wrap(a - b))))


def contact_type(angle):
    """Existing realized-contact bins; here applied to candidate local lane tangents."""
    for threshold, kind in ((15, "rear_end"), (60, "cut_in"), (120, "side")):
        if angle <= threshold + 1e-8:
            return kind
    return "head_on"


def feature_ids(value):
    if value is None:
        return []
    if isinstance(value, (dict, str, int, np.integer)):
        value = [value]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return feature_ids(value.item())
    return sorted({str(v.get("feature_id", v.get("id"))) if isinstance(v, dict)
                   else str(v) for v in value
                   if v is not None and (not isinstance(v, dict)
                                         or v.get("feature_id", v.get("id")) is not None)})


class Lane:
    def __init__(self, lane_id, feature):
        self.id = lane_id
        self.feature = feature
        self.xy = np.asarray(feature["polyline"], dtype=float)[:, :2]
        if len(self.xy) < 2 or not np.isfinite(self.xy).all():
            raise ValueError("invalid lane coordinates")
        self.delta = np.diff(self.xy, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.s = np.r_[0., np.cumsum(self.lengths)]
        if self.s[-1] <= 1e-6:
            raise ValueError("zero-length lane")
        self.headings = np.arctan2(self.delta[:, 1], self.delta[:, 0])
        useful = np.flatnonzero(self.lengths > 1e-6)
        for i in np.flatnonzero(self.lengths <= 1e-6):
            self.headings[i] = self.headings[useful[np.argmin(abs(useful - i))]]
        self.entries = set(feature_ids(feature.get("entry_lanes")))
        self.exits = set(feature_ids(feature.get("exit_lanes")))
        self.low, self.high = self.xy.min(0), self.xy.max(0)

    def project(self, point):
        fraction = np.clip(np.sum((point - self.xy[:-1]) * self.delta, axis=1)
                           / np.maximum(self.lengths ** 2, 1e-12), 0., 1.)
        projected = self.xy[:-1] + fraction[:, None] * self.delta
        distances = np.linalg.norm(projected - point, axis=1)
        i = int(np.argmin(distances))
        return dict(point=projected[i], heading=float(self.headings[i]),
                    distance=float(distances[i]), segment=i,
                    s=float(self.s[i] + fraction[i] * self.lengths[i]))

    def sample(self, s):
        s = np.clip(s, 0., self.s[-1])
        i = min(int(np.searchsorted(self.s, s, side="right") - 1), len(self.delta) - 1)
        f = (s - self.s[i]) / max(self.lengths[i], 1e-12)
        return self.xy[i] + f * self.delta[i], float(self.headings[i]), i

    def neighbors(self, segment, side=None):
        """Respect WOMD per-segment neighbor ranges, while accepting nuPlan ID lists."""
        output = set()
        for key in ([side] if side else ["left_neighbor", "right_neighbor"]):
            values = self.feature.get(key, [])
            if values is None:
                values = []
            if isinstance(values, (str, int, dict)):
                values = [values]
            for item in values:
                if isinstance(item, dict):
                    start = int(item.get("self_start_index", 0))
                    end = int(item.get("self_end_index", len(self.xy) - 1))
                    if not start <= segment <= end:
                        continue
                output.update(feature_ids(item))
        return output


class RoadGraph:
    def __init__(self, features):
        self.lanes = {}
        self.skipped_lane_ids = []
        for key, feature in sorted(features.items(), key=lambda item: str(item[0])):
            if "polyline" not in feature or not (
                "LANE" in str(feature.get("type", "")).upper()
                or "entry_lanes" in feature or "exit_lanes" in feature
            ):
                continue
            try:
                self.lanes[str(key)] = Lane(str(key), feature)
            except (ValueError, IndexError, TypeError):
                self.skipped_lane_ids.append(str(key))
        # Accept datasets storing either or both sides of an edge. Never add geometric links.
        for lane in self.lanes.values():
            for key in list(lane.entries):
                if key in self.lanes:
                    self.lanes[key].exits.add(lane.id)
            for key in list(lane.exits):
                if key in self.lanes:
                    self.lanes[key].entries.add(lane.id)

    def near(self, point, radius):
        result = {}
        for key, lane in self.lanes.items():
            if np.linalg.norm(np.maximum(np.maximum(lane.low - point, point - lane.high), 0)) > radius:
                continue
            projection = lane.project(point)
            if projection["distance"] <= radius:
                result[key] = projection
        return result

    def corridor(self, anchor, nearby):
        """Extend only through unambiguous continuations; do not absorb other merge arms."""
        result = {anchor}
        for direction in ("entries", "exits"):
            key = anchor
            for _ in range(24):
                links = getattr(self.lanes[key], direction)
                if len(links) != 1:
                    break
                nxt = next(iter(links))
                if nxt not in nearby or nxt in result:
                    break
                first, second = ((nxt, key) if direction == "entries" else (key, nxt))
                if angle_deg(self.lanes[first].headings[-1], self.lanes[second].headings[0]) > 45:
                    break
                if np.linalg.norm(self.lanes[first].xy[-1] - self.lanes[second].xy[0]) > 3:
                    break
                result.add(nxt)
                key = nxt
        return result

    def ring(self, seeds, nearby):
        """Bounded smooth-cycle heuristic, intentionally not a full roundabout recognizer."""
        for start in sorted(seeds):
            stack = [(start, (start,), self.lanes[start].s[-1])]
            visited = 0
            while stack and visited < 256:
                key, path, length = stack.pop()
                visited += 1
                for nxt in sorted(self.lanes[key].exits):
                    if nxt not in nearby:
                        continue
                    a, b = self.lanes[key], self.lanes[nxt]
                    if np.linalg.norm(a.xy[-1] - b.xy[0]) > 3 or angle_deg(a.headings[-1], b.headings[0]) > 65:
                        continue
                    if nxt == start:
                        headings = np.concatenate([self.lanes[k].headings for k in path])
                        changes = wrap(np.diff(np.r_[headings, headings[0]]))
                        signed, total = abs(float(changes.sum())), float(abs(changes).sum())
                        if (length >= 15 and 300 <= np.rad2deg(signed) <= 420
                                and total <= signed * 1.25 and np.max(abs(changes)) < np.deg2rad(65)):
                            return set(path)
                    elif nxt not in path and len(path) < 24 and length + b.s[-1] <= 300:
                        stack.append((nxt, path + (nxt,), length + b.s[-1]))
        # Some maps encode an entire circular lane as one closed polyline without a self-edge.
        for key in sorted(seeds):
            lane = self.lanes[key]
            turn = np.rad2deg(abs(float(wrap(np.diff(lane.headings)).sum())))
            if lane.s[-1] >= 15 and np.linalg.norm(lane.xy[0] - lane.xy[-1]) < 2 and turn >= 300:
                return {key}
        return set()


def intersections(a, b, center, radius):
    """Find local centerline intersections and their directed tangents in XY only."""
    hits = []
    for i, (p, r) in enumerate(zip(a.xy[:-1], a.delta)):
        if a.lengths[i] < 1e-6:
            continue
        if np.linalg.norm(np.maximum(np.maximum(np.minimum(p, p + r) - center,
                                                center - np.maximum(p, p + r)), 0)) > radius:
            continue
        q, s = b.xy[:-1], b.delta
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        useful = abs(den) > 1e-9
        safe = np.where(useful, den, 1.)
        d = q - p
        t = (d[:, 0] * s[:, 1] - d[:, 1] * s[:, 0]) / safe
        u = (d[:, 0] * r[1] - d[:, 1] * r[0]) / safe
        for j in np.flatnonzero(useful & (t >= -1e-8) & (t <= 1 + 1e-8)
                               & (u >= -1e-8) & (u <= 1 + 1e-8)):
            point = p + np.clip(t[j], 0, 1) * r
            if np.linalg.norm(point - center) <= radius:
                hits.append((point, angle_deg(a.headings[i], b.headings[j])))
    # Adjacent polyline segments can report the same shared vertex twice.
    unique = {}
    for point, angle in hits:
        unique.setdefault(tuple(np.round(point, 3)), (point, angle))
    return sorted(unique.values(), key=lambda hit: np.linalg.norm(hit[0] - center))


def polygon_distance(point, polygon):
    """Euclidean distance to a convex polygon, including its interior."""
    point, polygon = np.asarray(point, float), np.asarray(polygon, float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = point - polygon
    cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
    if np.all(cross >= -1e-9) or np.all(cross <= 1e-9):
        return 0.0
    t = np.clip(np.sum(relative * edges, axis=1) / np.maximum(np.sum(edges ** 2, axis=1), 1e-12), 0, 1)
    return float(np.linalg.norm(point - (polygon + t[:, None] * edges), axis=1).min())


def intersection_cutin_guard(graph, position, nearby, anchors, config):
    """Map-only crossing footprints, then an explicit metric exclusion buffer.

    WOMD does not supply junction polygons. At each genuine centerline crossing,
    intersect the two local tangent lane strips using their recorded left/right
    widths. Missing widths use a disclosed 1.8m half-width solely for this rule.
    Keep footprints separate so two junctions cannot fill the road between them.
    Opposite directions on one axis, adjacent lanes, and mapped merge/seam
    connections do not define a crossing. This remains an XY map heuristic.
    """
    if config.cut_in_intersection_buffer_m is None:
        return dict(enabled=False, target_blocked=False, regions=[], suppressed_opportunities=[])
    corridor = set().union(*(graph.corridor(key, nearby) for key in anchors)) if anchors else set()
    regions, seen = [], set()

    def widths(lane, projection):
        values = np.asarray(lane.feature.get('width', []), float)
        if values.shape == (len(lane.xy), 2):
            value = values[projection['segment']]
            if np.isfinite(value).all() and np.all(value > 0):
                return value, 'native_lane_width'
        return np.array([1.8, 1.8]), 'fallback_1.8m_half_width'

    for a in sorted(corridor):
        lane = graph.lanes[a]
        own = graph.corridor(a, nearby)
        for b in sorted(nearby.keys() - own):
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            other = graph.lanes[b]
            related = (b in lane.entries or b in lane.exits or bool(lane.exits & other.exits)
                       or bool(lane.entries & other.entries))
            if related:
                continue
            for point, angle in intersections(lane, other, position, config.search_radius_m):
                # Folding directed angles into an undirected road axis avoids
                # treating a normal two-way road as a junction.
                if min(angle, 180 - angle) <= 2:
                    continue
                pa, pb = lane.project(point), other.project(point)
                if b in lane.neighbors(pa['segment']) or a in other.neighbors(pb['segment']):
                    continue
                wa, sa = widths(lane, pa)
                wb, sb = widths(other, pb)
                normals = np.array([[-np.sin(pa['heading']), np.cos(pa['heading'])],
                                    [-np.sin(pb['heading']), np.cos(pb['heading'])]])
                offsets = np.array([[-wa[1], -wb[1]], [wa[0], -wb[1]],
                                    [wa[0], wb[0]], [-wa[1], wb[0]]])
                polygon = np.linalg.solve(normals, offsets.T).T + point
                regions.append(dict(lane_ids=[a, b], centerline_crossing=point.tolist(),
                                    polygon=polygon.tolist(), width_sources=[sa, sb],
                                    target_distance_m=polygon_distance(position, polygon)))
    distances = [r['target_distance_m'] for r in regions]
    nearest = min(distances) if distances else None
    return dict(enabled=True, buffer_m=config.cut_in_intersection_buffer_m,
                geometry='union of local crossing lane-strip footprints; XY only',
                nearest_target_distance_m=nearest,
                target_blocked=nearest is not None and nearest <= config.cut_in_intersection_buffer_m + 1e-8,
                regions=regions, suppressed_opportunities=[])


def analyze_frame(graph, position, heading, config):
    nearby = graph.near(position, config.search_radius_m)
    matches = [(p["distance"] + 0.03 * angle_deg(p["heading"], heading), key)
               for key, p in nearby.items() if p["distance"] <= config.ego_lane_distance_m
               and angle_deg(p["heading"], heading) <= config.ego_heading_tolerance_deg]
    matches.sort()
    if not matches:
        return dict(status="no_ego_lane", ego_lane_ids=[], nearby_lane_ids=sorted(nearby),
                    road_type_ids=[], road_types=[], potential_collision_types=[], opportunities=[],
                    road_features=[])
    anchors = [key for cost, key in matches if cost <= matches[0][0] + config.ego_match_slack_m][:3]
    corridor = set().union(*(graph.corridor(key, nearby) for key in anchors))
    local = {key for key in corridor if nearby[key]["distance"] <= config.conflict_radius_m}
    relations = []
    features = []
    labels = set()

    def feature(road_id, lane_ids, **extra):
        labels.add(road_id)
        features.append(dict(road_type_id=road_id, lane_ids=sorted(lane_ids), **extra))

    def relation(kind, a, b, point, angle, detail=None, **extra):
        relations.append(dict(relation=kind, relation_detail=detail or kind,
                              ego_lane_id=a, other_lane_id=b, point=point.tolist(),
                              angle_deg=float(angle), **extra))

    for key in sorted(local):
        lane, p = graph.lanes[key], nearby[key]
        relation("same_lane", anchors[0], key, p["point"], 0.)
        # Continuous curvature uses a local arclength window, never distant tangents.
        lo, hi = max(0., p["s"] - config.conflict_radius_m), min(lane.s[-1], p["s"] + config.conflict_radius_m)
        _, _, first = lane.sample(lo)
        _, _, last = lane.sample(hi)
        turns = wrap(np.diff(lane.headings[first:last + 1]))
        net_turn = float(abs(np.rad2deg(turns.sum())))
        if net_turn >= config.curve_turn_deg:
            feature(4, [key], turn_deg=net_turn)
        if config.u_turn_deg <= net_turn <= 240:
            feature(9, [key], turn_deg=net_turn)

    # Include turns split across multiple uniquely connected lane records.
    for anchor in anchors:
        chain = graph.corridor(anchor, nearby)
        starts = [k for k in chain if not (graph.lanes[k].entries & chain)]
        for start in sorted(starts):
            key, path, tangents = start, [], []
            while key not in path:
                path.append(key)
                lane = graph.lanes[key]
                mid = (lane.xy[:-1] + lane.xy[1:]) / 2
                mask = np.linalg.norm(mid - position, axis=1) <= config.conflict_radius_m
                tangents.extend(lane.headings[mask].tolist())
                following = lane.exits & chain
                if len(following) != 1:
                    break
                key = next(iter(following))
            if len(tangents) > 1:
                turn = abs(float(np.rad2deg(wrap(np.diff(tangents)).sum())))
                if turn >= config.curve_turn_deg:
                    feature(4, path, turn_deg=turn)
                if config.u_turn_deg <= turn <= 240:
                    feature(9, path, turn_deg=turn)

    has_adjacent = False
    has_opposing = False
    for a in sorted(local):
        lane, pa = graph.lanes[a], nearby[a]
        neighbors = lane.neighbors(pa["segment"])
        # Two crossing lanes can both match Ego at the conflict point. Do not let
        # multi-lane matching erase the conflict by treating their union as one route.
        same_corridor = graph.corridor(a, nearby)
        for b in sorted(nearby.keys() - same_corridor):
            other, pb = graph.lanes[b], nearby[b]
            if pb["distance"] > config.conflict_radius_m:
                continue
            # Evaluate the other lane beside the local Ego-lane projection.
            pb = other.project(pa["point"])
            angle = angle_deg(pa["heading"], pb["heading"])
            adjacent = b in neighbors or a in other.neighbors(pb["segment"])
            if adjacent and angle <= config.same_direction_deg and pb["distance"] <= config.adjacent_distance_m:
                has_adjacent = True
                relation("merge", a, b, (pa["point"] + pb["point"]) / 2, angle, "adjacent_same_dir")
            elif angle >= config.opposing_direction_deg and pb["distance"] <= config.opposing_distance_m:
                has_opposing = True
                relation("opposing", a, b, (pa["point"] + pb["point"]) / 2, angle)
            # Shared graph endpoints belong to merge/diverge, not an intersection.
            related = (b in lane.entries or b in lane.exits or bool(lane.exits & other.exits)
                       or bool(lane.entries & other.entries))
            if not related and not adjacent:
                for point, contact_angle in intersections(lane, other, position, config.conflict_radius_m):
                    if 2 < contact_angle < 178:
                        feature(7, [a, b], point=point.tolist())
                        relation("crossing", a, b, point, contact_angle)

    # A fan is relevant only if it touches the Ego corridor at a nearby endpoint.
    fan_kinds = set()
    for key in sorted(nearby):
        lane = graph.lanes[key]
        for links, point, phase in ((lane.entries, lane.xy[0], "merging"),
                                    (lane.exits, lane.xy[-1], "diverging")):
            if np.linalg.norm(point - position) > config.conflict_radius_m:
                continue
            arms = []
            for arm_id in sorted(links & nearby.keys()):
                arm = graph.lanes[arm_id]
                ha, hb = ((arm.headings[-1], lane.headings[0]) if phase == "merging"
                          else (lane.headings[-1], arm.headings[0]))
                if angle_deg(ha, hb) <= 75:
                    arms.append(arm_id)
            if len(arms) < 2 or not ({key, *arms} & corridor):
                continue
            fan_kinds.add(phase)
            feature(5, [key, *arms], phase=phase, point=point.tolist())
            # Diverging branches alone do not imply a lateral incursion opportunity.
            if phase == "merging":
                ego_arm = next((a for a in anchors if a in arms),
                               next((a for a in arms if a in corridor), key))
                for b in arms:
                    if b != ego_arm:
                        relation("merge", ego_arm, b, point,
                                 angle_deg(graph.lanes[ego_arm].project(point)["heading"],
                                           graph.lanes[b].headings[-1]), "merging")
    if fan_kinds == {"merging", "diverging"}:
        feature(5, local, phase="weaving")

    def side_counts(lane, s):
        point, direction, segment = lane.sample(s)
        return tuple(sum(k in graph.lanes
                         and angle_deg(graph.lanes[k].project(point)["heading"], direction) <= config.same_direction_deg
                         for k in lane.neighbors(segment, side))
                     for side in ("left_neighbor", "right_neighbor"))

    for key in sorted(local):
        lane, p = graph.lanes[key], nearby[key]
        before = side_counts(lane, max(0., p["s"] - config.conflict_radius_m))
        after = side_counts(lane, min(lane.s[-1], p["s"] + config.conflict_radius_m))
        # A mapped continuation is needed to infer count changes across record boundaries.
        if lane.s[-1] - p["s"] <= config.conflict_radius_m and len(lane.exits) == 1:
            nxt = next(iter(lane.exits))
            if nxt in nearby and angle_deg(lane.headings[-1], graph.lanes[nxt].headings[0]) <= config.same_direction_deg:
                after = side_counts(graph.lanes[nxt], 0.)
        if p["s"] <= config.conflict_radius_m and len(lane.entries) == 1:
            prev = next(iter(lane.entries))
            if prev in nearby and angle_deg(graph.lanes[prev].headings[-1], lane.headings[0]) <= config.same_direction_deg:
                before = side_counts(graph.lanes[prev], graph.lanes[prev].s[-1])
        if before != after:
            feature(6, [key], before_neighbor_counts=list(before), after_neighbor_counts=list(after))

    ring_seeds = set(local)
    for key in local:
        ring_seeds.update((graph.lanes[key].entries | graph.lanes[key].exits) & nearby.keys())
    ring = graph.ring(ring_seeds, nearby)
    if ring:
        phases = set()
        if set(anchors) & ring:
            phases.add("inner")
        if any(graph.lanes[a].exits & ring for a in anchors if a not in ring):
            phases.add("entry")
        if any(graph.lanes[a].entries & ring for a in anchors if a not in ring):
            phases.add("exit")
        if phases:
            feature(8, ring, phases=sorted(phases))
            # A local half-circle inside a recognized ring is not a separate U-turn.
            features = [f for f in features if f["road_type_id"] != 9 or not (set(f["lane_ids"]) & ring)]
            if not any(f["road_type_id"] == 9 for f in features):
                labels.discard(9)

    # Road recognition above gates the four relation-to-type rules below.
    base = 3 if has_opposing else 2 if has_adjacent else 1
    feature(base, local)
    guard = intersection_cutin_guard(graph, position, nearby, anchors, config)
    unique = {}
    for op in relations:
        rel = op["relation"]
        if rel == "same_lane":
            ids, types = [base], ["rear_end"]
        elif rel == "merge":
            ids = [base] if op["relation_detail"] == "adjacent_same_dir" else [5]
            types = ["cut_in"]
        elif rel == "crossing":
            ids, types = [7], [contact_type(op["angle_deg"])]
        else:
            ids, types = [3], ["head_on"]
        if 'cut_in' in types and guard['enabled']:
            distances = [polygon_distance(op['point'], r['polygon']) for r in guard['regions']]
            nearest = min(distances) if distances else None
            if guard['target_blocked'] or (nearest is not None and nearest <= guard['buffer_m'] + 1e-8):
                guard['suppressed_opportunities'].append(dict(**op, proposed_collision_types=types,
                                                              distance_to_intersection_m=nearest))
                continue
        involved = {op["ego_lane_id"], op["other_lane_id"]}
        ids.extend(f["road_type_id"] for f in features if f["road_type_id"] in {4, 6, 8, 9}
                   and involved.intersection(f["lane_ids"]))
        op.update(road_type_ids=sorted(set(ids)), collision_types=types)
        signature = (rel, op["ego_lane_id"], op["other_lane_id"], op["relation_detail"],
                     tuple(np.round(op["point"], 3)))
        unique.setdefault(signature, op)
    opportunities = list(unique.values())
    kinds = {k for op in opportunities for k in op["collision_types"]}
    return dict(status="ok", ego_lane_ids=anchors, nearby_lane_ids=sorted(nearby),
                road_type_ids=sorted(labels), road_types=[ROAD_TYPES[i][0] for i in sorted(labels)],
                potential_collision_types=[k for k in COLLISION_TYPES if k in kinds],
                opportunities=opportunities, road_features=features, cut_in_intersection_guard=guard)


def scan_scene(scene, config=None):
    """Public CPU-only API. Positions and headings stay in the input scene frame."""
    config = config or ScanConfig()
    metadata = scene.get("metadata", {})
    ego_id = str(metadata["sdc_id"])
    state = scene["tracks"][ego_id]["state"]
    positions = np.asarray(state["position"], dtype=float)[:, :2]
    n = len(positions)
    headings = np.asarray(state["heading"], dtype=float).reshape(n)
    valid = np.asarray(state["valid"], dtype=bool).reshape(n)
    timestamps = np.asarray(metadata.get("ts", np.arange(n) * 0.1), dtype=float).reshape(n)
    if n < 2 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("scene needs at least two finite, increasing timestamps")
    times = timestamps - timestamps[0]
    native_dt = float(np.median(np.diff(times)))
    if config.interval_s < native_dt - 1e-4:
        raise ValueError("scan interval cannot be finer than source samples")
    graph = RoadGraph(scene.get("map_features", {}))
    frames = []
    ticks = np.arange(int(np.floor((times[-1] + native_dt / 2) / config.interval_s)) + 1) * config.interval_s
    used_source_frames = set()
    for time_s in ticks:
        frame = int(np.argmin(abs(times - time_s)))
        if frame in used_source_frames or abs(times[frame] - time_s) > native_dt / 2 + 1e-4:
            raise ValueError("source timeline does not support requested sampling grid")
        used_source_frames.add(frame)
        good = bool(valid[frame] and np.isfinite(positions[frame]).all() and np.isfinite(headings[frame]))
        if good:
            result = analyze_frame(graph, positions[frame], headings[frame], config)
        else:
            result = dict(status="invalid_ego", ego_lane_ids=[], nearby_lane_ids=[], road_type_ids=[],
                          road_types=[], potential_collision_types=[], opportunities=[], road_features=[])
        result.update(time_s=round(float(time_s), 6), source_time_s=float(times[frame]),
                      source_frame=frame, ego_position=positions[frame].tolist() if good else None,
                      ego_heading_rad=float(headings[frame]) if good else None)
        frames.append(result)
    used = {k for f in frames for k in f["nearby_lane_ids"]}
    used.update(k for f in frames for feature in f["road_features"] for k in feature["lane_ids"])
    lane_bank = {k: dict(polyline=graph.lanes[k].xy.tolist(),
                         entry_lanes=sorted(graph.lanes[k].entries), exit_lanes=sorted(graph.lanes[k].exits),
                         left_neighbors=feature_ids(graph.lanes[k].feature.get("left_neighbor")),
                         right_neighbors=feature_ids(graph.lanes[k].feature.get("right_neighbor")))
                 for k in sorted(used)}
    ego_track = {k: np.asarray(state[k]).tolist() for k in
                 ("position", "heading", "velocity", "valid", "length", "width") if k in state}
    return dict(schema_version="topor2f_local_opportunities_v1", scenario_id=str(scene.get("id", metadata.get("scenario_id", "unknown"))),
                config=asdict(config), ego_id=ego_id, coordinate_frame="input_scene_xy",
                coordinate_origin_utm=metadata.get("coordinate_origin_utm"),
                timestamp_origin=float(timestamps[0]), source_times_s=times.tolist(), ego_track=ego_track,
                road_catalog={str(k): dict(code=v[0], name=v[1]) for k, v in ROAD_TYPES.items()},
                collision_types=list(COLLISION_TYPES), skipped_lane_ids=graph.skipped_lane_ids,
                lanes=lane_bank, frames=frames)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", type=Path, action="append", help="ScenarioNet pickle; repeat for multiple scenes")
    source.add_argument("--manifest", type=Path, help="CSV with scenario_file; requested types/road labels are ignored")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--search-radius", type=float, default=40.)
    parser.add_argument("--conflict-radius", type=float, default=12.)
    parser.add_argument("--cut-in-intersection-buffer", type=float, default=5.)
    args = parser.parse_args()
    config = ScanConfig(interval_s=args.interval, search_radius_m=args.search_radius,
                        conflict_radius_m=args.conflict_radius,
                        cut_in_intersection_buffer_m=args.cut_in_intersection_buffer)
    if args.manifest:
        if args.dataset_dir is None:
            parser.error("--manifest requires --dataset-dir")
        with args.manifest.open(newline="") as stream:
            paths = [args.dataset_dir / row["scenario_file"] for row in csv.DictReader(stream)]
    else:
        paths = args.scene
    if not paths:
        parser.error("no input scenes")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, path in enumerate(paths):
        record = dict(input_path=str(path.resolve()))
        try:
            record["input_sha256"] = sha256(path)
            with path.open("rb") as stream:
                result = scan_scene(pickle.load(stream), config)
            name = f"{index:04d}_" + re.sub(r"[^A-Za-z0-9_.-]", "_", result["scenario_id"]) + ".json"
            result["source"] = record.copy()
            write_json(args.output / name, result)
            record.update(status="completed", scenario_id=result["scenario_id"], file=name,
                          frame_count=len(result["frames"]), output_sha256=sha256(args.output / name))
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        records.append(record)
        print(f"{index + 1}/{len(paths)} {record['status']} {path.name}", flush=True)
    failed = any(r["status"] == "failed" for r in records)
    write_json(args.output / "index.json", dict(status="failed" if failed else "completed", config=asdict(config),
               scanner_path=str(Path(__file__).resolve()), scanner_sha256=sha256(Path(__file__)), scenes=records))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

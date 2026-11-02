"""Function: Exercise the standalone nine-road collision-opportunity rules on known maps.
Input: Synthetic directed maps, Ego states and timestamp/invalid-frame edge cases.
Output: CPU unittest results; does not load weights or modify experiment outputs.
"""
import copy
import json
import unittest

import numpy as np

from scan_collision_opportunities import (RoadGraph, ScanConfig, analyze_frame,
                                          contact_type, json_value, scan_scene)


def lane(points, **links):
    return dict(type="LANE_SURFACE_STREET", polyline=np.asarray(points, float), **links)


def inspect(features, position=(0, 0), heading=0):
    return analyze_frame(RoadGraph(features), np.array(position, float), heading, ScanConfig())


def scene(features, n=91):
    return dict(id="synthetic", metadata=dict(sdc_id="ego", ts=np.arange(n) * .1),
                map_features=features, tracks={"ego": {"state": dict(
                    position=np.zeros((n, 3)), heading=np.zeros(n), valid=np.ones(n, bool),
                    velocity=np.zeros((n, 2)), length=np.full(n, 4.8), width=np.full(n, 2.))}})


class RulesTest(unittest.TestCase):
    def test_single_lane_and_continuation(self):
        result = inspect({"a": lane([[-20, 0], [2, 0]], exit_lanes=["b"]),
                          "b": lane([[2, 0], [30, 0]])})
        self.assertEqual(result["road_type_ids"], [1])
        self.assertEqual(result["potential_collision_types"], ["rear_end"])

    def test_multilane_and_two_way_are_distinct(self):
        features = {"a": lane([[-20, 0], [20, 0]], left_neighbor=["b"]),
                    "b": lane([[-20, 3.5], [20, 3.5]])}
        self.assertIn(2, inspect(features)["road_type_ids"])
        features["c"] = lane([[20, -3.5], [-20, -3.5]])
        result = inspect(features)
        self.assertIn(3, result["road_type_ids"])
        self.assertNotIn(2, result["road_type_ids"])
        self.assertEqual(result["potential_collision_types"], ["rear_end", "cut_in", "head_on"])

    def test_nearby_unrelated_same_direction_lane_is_not_cut_in(self):
        result = inspect({"a": lane([[-20, 0], [20, 0]]), "b": lane([[-20, 5], [20, 5]])})
        self.assertNotIn("cut_in", result["potential_collision_types"])

    def test_merge_and_pure_diverge(self):
        features = {"a": lane([[-20, 0], [0, 0]], exit_lanes=["c"]),
                    "b": lane([[-20, 5], [0, 0]], exit_lanes=["c"]),
                    "c": lane([[0, 0], [20, 0]])}
        result = inspect(features, (-4, 0))
        self.assertIn(5, result["road_type_ids"])
        self.assertIn("cut_in", result["potential_collision_types"])
        diverge = {"a": lane([[-20, 0], [0, 0]], exit_lanes=["b", "c"]),
                   "b": lane([[0, 0], [20, 5]]), "c": lane([[0, 0], [20, 0]])}
        result = inspect(diverge, (-4, 0))
        self.assertIn(5, result["road_type_ids"])
        self.assertNotIn("cut_in", result["potential_collision_types"])
        self.assertNotIn(7, result["road_type_ids"])

    def test_weaving(self):
        features = {"a": lane([[-20, 0], [-2, 0]], exit_lanes=["c"]),
                    "b": lane([[-20, 5], [-2, 0]], exit_lanes=["c"]),
                    "c": lane([[-2, 0], [2, 0]], exit_lanes=["d", "e"]),
                    "d": lane([[2, 0], [20, 0]]), "e": lane([[2, 0], [20, 5]])}
        self.assertTrue(any(f.get("phase") == "weaving" for f in inspect(features)["road_features"]))

    def test_neighbor_ranges_and_lane_loss(self):
        features = {"a": lane(np.c_[np.arange(-20, 21), np.zeros(41)], left_neighbor=[
            dict(feature_id="b", self_start_index="0", self_end_index="20")]),
            "b": lane([[-20, 3.5], [0, 3.5]])}
        self.assertIn(6, inspect(features)["road_type_ids"])
        self.assertNotIn("cut_in", inspect(features, (15, 0))["potential_collision_types"])

    def test_crossing_angles_and_local_radius(self):
        for angle, kind in [(10, "rear_end"), (30, "cut_in"), (90, "side"), (170, "head_on")]:
            v = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) * 20
            result = inspect({"a": lane([[-25, 0], [25, 0]]), "b": lane([-v, v])})
            self.assertIn(7, result["road_type_ids"])
            if kind == 'cut_in':
                self.assertNotIn(kind, result['potential_collision_types'])
                legacy = analyze_frame(RoadGraph({'a': lane([[-25, 0], [25, 0]]), 'b': lane([-v, v])}),
                                       np.zeros(2), 0., ScanConfig(cut_in_intersection_buffer_m=None))
                self.assertIn(kind, legacy['potential_collision_types'])
            else:
                self.assertIn(kind, result["potential_collision_types"])
        result = inspect({"a": lane([[-25, 0], [25, 0]]), "b": lane([[20, -20], [20, 20]])})
        self.assertNotIn(7, result["road_type_ids"])
        self.assertIn("b", result["nearby_lane_ids"])

    def test_curve_and_uturn_do_not_invent_conflicts(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 81)
        xy = 6 * np.c_[np.cos(angles), np.sin(angles)]
        result = inspect({"u": lane(xy)}, xy[40], np.pi / 2)
        self.assertIn(4, result["road_type_ids"])
        self.assertIn(9, result["road_type_ids"])
        self.assertEqual(result["potential_collision_types"], ["rear_end"])

    def test_uturn_across_lane_records(self):
        angles = np.linspace(-np.pi / 2, np.pi / 2, 81)
        xy = 6 * np.c_[np.cos(angles), np.sin(angles)]
        features = {"a": lane(xy[:41], exit_lanes=["b"]), "b": lane(xy[40:])}
        self.assertIn(9, inspect(features, xy[38], angles[38] + np.pi / 2)["road_type_ids"])

    def test_ring_phases(self):
        features = {}
        for i in range(4):
            a = np.linspace(i * np.pi / 2, (i + 1) * np.pi / 2, 21)
            features[str(i)] = lane(8 * np.c_[np.cos(a), np.sin(a)], exit_lanes=[str((i + 1) % 4)])
        features["entry"] = lane([[8, -15], [8, 0]], exit_lanes=["0"])
        features["1"]["exit_lanes"].append("exit")
        features["exit"] = lane([[-8, 0], [-8, -15]])
        for position, heading, phase in [((8, -4), np.pi / 2, "entry"),
                                          ((0, 8), np.pi, "inner"),
                                          ((-8, -4), -np.pi / 2, "exit")]:
            result = inspect(features, position, heading)
            self.assertIn(8, result["road_type_ids"], (phase, result["road_types"]))
            self.assertTrue(any(phase in f.get("phases", []) for f in result["road_features"]))

    def test_single_record_ring_is_not_a_uturn(self):
        angles = np.linspace(0, 2 * np.pi, 121)
        points = 8 * np.c_[np.cos(angles), np.sin(angles)]
        result = inspect({"ring": lane(points)}, (-8, 0), -np.pi / 2)
        self.assertIn(8, result["road_type_ids"])
        self.assertNotIn(9, result["road_type_ids"])

    def test_half_second_sampling_and_invalid_ego(self):
        data = scene({"a": lane([[-20, 0], [20, 0]])})
        data["metadata"]["ts"][1:] += 0.00002
        data["tracks"]["ego"]["state"]["valid"][10] = False
        result = scan_scene(data)
        self.assertEqual([f["source_frame"] for f in result["frames"]], list(range(0, 91, 5)))
        self.assertEqual([f["time_s"] for f in result["frames"]], [i * .5 for i in range(19)])
        self.assertEqual(result["frames"][2]["status"], "invalid_ego")
        self.assertEqual(result["frames"][2]["potential_collision_types"], [])
        json.dumps(json_value(result), allow_nan=False)
        self.assertNotIn("reasons", json.dumps(json_value(result)))

    def test_no_lane_and_bad_timestamps(self):
        data = scene({})
        self.assertEqual(scan_scene(data)["frames"][0]["status"], "no_ego_lane")
        data["metadata"]["ts"][4] = data["metadata"]["ts"][3]
        with self.assertRaises(ValueError):
            scan_scene(data)

    def test_rotation_translation_and_unused_labels(self):
        data = scene({"a": lane([[-20, 0], [20, 0]]), "b": lane([[0, -20], [0, 20]])})
        original = scan_scene(data)
        moved = copy.deepcopy(data)
        angle = 1.3
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for value in moved["map_features"].values():
            value["polyline"] = value["polyline"] @ rotation.T + [8000, 4000]
        moved["tracks"]["ego"]["state"]["position"][:, :2] += [8000, 4000]
        moved["tracks"]["ego"]["state"]["heading"] += angle
        moved["metadata"].update(road_class="roundabout", requested_collision_type="head_on")
        transformed = scan_scene(moved)
        for a, b in zip(original["frames"], transformed["frames"]):
            self.assertEqual(a["road_types"], b["road_types"])
            self.assertEqual(a["potential_collision_types"], b["potential_collision_types"])

    def test_contact_boundaries(self):
        self.assertEqual([contact_type(x) for x in (0, 15, 15.1, 60, 60.1, 120, 120.1, 180)],
                         ["rear_end", "rear_end", "cut_in", "cut_in", "side", "side", "head_on", "head_on"])


if __name__ == "__main__":
    unittest.main()

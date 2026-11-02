"""Function: verify intersection cut-in exclusion geometry and unaffected roads.
Input: synthetic directed maps with exact lane widths and metric boundary cases.
Output: CPU unittest assertions; no generation or data mutations.
"""
import copy
import unittest
import numpy as np
from scan_collision_opportunities import RoadGraph, ScanConfig, analyze_frame, polygon_distance
from test_scan_collision_opportunities import lane


def junction(cross_x=0., half_width=1.8):
    def road(points, **links):
        return lane(points, width=np.full((len(points), 2), half_width), **links)
    return {'ego':road([[-80,0],[80,0]],left_neighbor=['adj']),
            'adj':road([[-80,3.6],[80,3.6]],right_neighbor=['ego']),
            'cross':road([[cross_x,-50],[cross_x,50]])}


def inspect(features, position=(0.,0.), buffer=5.):
    return analyze_frame(RoadGraph(features),np.array(position),0.,ScanConfig(cut_in_intersection_buffer_m=buffer))


class GuardTests(unittest.TestCase):
    def test_inside_and_exact_five_metre_boundary(self):
        # Cross-road half-width is 1.8m: outside boundary starts at x=6.8.
        for x in [0.,-3.,-6.8,6.8]:
            result=inspect(junction(),(x,0.))
            self.assertNotIn('cut_in',result['potential_collision_types'])
            self.assertTrue(result['cut_in_intersection_guard']['target_blocked'])
        self.assertIn('cut_in',inspect(junction(),(-6.801,0.))['potential_collision_types'])

    def test_wide_intersection_interior_and_buffer(self):
        features=junction(half_width=8.)
        for x in [6.,12.9,13.]:
            self.assertNotIn('cut_in',inspect(features,(x,0.))['potential_collision_types'])
        self.assertIn('cut_in',inspect(features,(13.01,0.))['potential_collision_types'])

    def test_parallel_opposite_and_adjacent_are_not_junctions(self):
        f=junction();del f['cross'];f['opposite']=lane([[80,-3.6],[-80,-3.6]])
        result=inspect(f)
        self.assertIn('cut_in',result['potential_collision_types'])
        self.assertEqual(result['cut_in_intersection_guard']['regions'],[])

    def test_separate_junctions_do_not_fill_gap(self):
        f=junction(-20.);f['cross2']=lane([[20,-50],[20,50]],width=np.full((2,2),1.8))
        result=inspect(f)
        self.assertIn('cut_in',result['potential_collision_types'])
        self.assertFalse(result['cut_in_intersection_guard']['target_blocked'])

    def test_shallow_crossing_never_becomes_cutin(self):
        f=junction();f['cross']=lane([[-30,-17.32],[30,17.32]])
        result=inspect(f)
        self.assertNotIn('cut_in',result['potential_collision_types'])
        self.assertTrue(result['cut_in_intersection_guard']['suppressed_opportunities'])

    def test_rotation_translation_invariance(self):
        f=junction();a=1.17;rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        transformed=copy.deepcopy(f)
        for value in transformed.values():value['polyline']=value['polyline']@rotation.T+[8000,4000]
        for x in [-6.8,-6.801,0.,20.]:
            first=inspect(f,(x,0.))
            second=analyze_frame(RoadGraph(transformed),np.array([x,0.])@rotation.T+[8000,4000],a,ScanConfig())
            self.assertEqual(first['potential_collision_types'],second['potential_collision_types'])
            self.assertEqual(first['cut_in_intersection_guard']['target_blocked'],second['cut_in_intersection_guard']['target_blocked'])

    def test_other_collision_types_unchanged(self):
        f=junction()
        old=inspect(f,buffer=None);new=inspect(f)
        self.assertEqual([t for t in old['potential_collision_types'] if t!='cut_in'],new['potential_collision_types'])
        self.assertEqual(old['road_types'],new['road_types'])

    def test_polygon_distance_includes_edges_and_interior(self):
        box=[[-2,-1],[2,-1],[2,1],[-2,1]]
        for p in [[0,0],[2,0],[-2,-1]]:self.assertEqual(polygon_distance(p,box),0.)
        self.assertAlmostEqual(polygon_distance([5,5],box),5.)


if __name__=='__main__':unittest.main()

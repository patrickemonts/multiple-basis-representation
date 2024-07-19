import unittest
from tfimsim.graph import LatticeGraph,Boundary

class TestBoundary(unittest.TestCase):

    def test_from_string(self):
        bdc = Boundary.from_string("pbc")
        self.assertEqual(Boundary.PBC,bdc)
        bdc = Boundary.from_string("obc")
        self.assertEqual(Boundary.OBC,bdc)

class TestGraphLattice(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.lat_pbc = LatticeGraph(self.N, self.N,Boundary.PBC)
        self.lat_pbc_rect = LatticeGraph(3, 2,Boundary.PBC)
        self.lat_obc = LatticeGraph(self.N, self.N,Boundary.OBC)
        self.lat_obc_rect = LatticeGraph(3, 2,Boundary.OBC)

# ======================= PBC ===========================
    def test_coords2ind_pbc(self):
        """
        0  3  6  

        2  5  8   2
        1  4  7   1
        0  3  6   0
        """
        self.assertEqual(self.lat_pbc.coords2ind((0,0)),0)
        self.assertEqual(self.lat_pbc.coords2ind((0,1)),1)
        self.assertEqual(self.lat_pbc.coords2ind((0,2)),2)
        self.assertEqual(self.lat_pbc.coords2ind((1,0)),3)
        self.assertEqual(self.lat_pbc.coords2ind((1,1)),4)
        self.assertEqual(self.lat_pbc.coords2ind((1,2)),5)
        self.assertEqual(self.lat_pbc.coords2ind((2,0)),6)
        self.assertEqual(self.lat_pbc.coords2ind((2,1)),7)
        self.assertEqual(self.lat_pbc.coords2ind((2,2)),8)
    
    def test_ind2coords_pbc(self):
        self.assertEqual(self.lat_pbc.ind2coords(0),(0,0))
        self.assertEqual(self.lat_pbc.ind2coords(1),(0,1))
        self.assertEqual(self.lat_pbc.ind2coords(2),(0,2))
        self.assertEqual(self.lat_pbc.ind2coords(3),(1,0))
        self.assertEqual(self.lat_pbc.ind2coords(4),(1,1))
        self.assertEqual(self.lat_pbc.ind2coords(5),(1,2))
        self.assertEqual(self.lat_pbc.ind2coords(6),(2,0))
        self.assertEqual(self.lat_pbc.ind2coords(7),(2,1))
        self.assertEqual(self.lat_pbc.ind2coords(8),(2,2))
    
    def test_adj_list_pbc(self):
        adj = self.lat_pbc.adj_list
        ref = [(0, 1), (0, 3),
               (1, 2), (1, 4),
               (2, 0), (2, 5),
               (3, 4), (3, 6),
               (4, 5), (4, 7),
               (5, 3), (5, 8),
               (6, 7), (6, 0),
               (7, 8), (7, 1),
               (8, 6), (8, 2)]
        self.assertEqual(len(adj),len(ref)) 
        self.assertEqual(adj,ref)

# ======================= PBC rect ===========================
    def test_coords2ind_pbc_rect(self):
        """
        0  2  4  

        1  3  5   1
        0  2  4   0
        """
        self.assertEqual(self.lat_pbc_rect.coords2ind((0,0)),0)
        self.assertEqual(self.lat_pbc_rect.coords2ind((0,1)),1)
        self.assertEqual(self.lat_pbc_rect.coords2ind((1,0)),2)
        self.assertEqual(self.lat_pbc_rect.coords2ind((1,1)),3)
        self.assertEqual(self.lat_pbc_rect.coords2ind((2,0)),4)
        self.assertEqual(self.lat_pbc_rect.coords2ind((2,1)),5)
    
    def test_ind2coords_pbc_rect(self):
        self.assertEqual(self.lat_pbc_rect.ind2coords(0),(0,0))
        self.assertEqual(self.lat_pbc_rect.ind2coords(1),(0,1))
        self.assertEqual(self.lat_pbc_rect.ind2coords(2),(1,0))
        self.assertEqual(self.lat_pbc_rect.ind2coords(3),(1,1))
        self.assertEqual(self.lat_pbc_rect.ind2coords(4),(2,0))
        self.assertEqual(self.lat_pbc_rect.ind2coords(5),(2,1))
    
    def test_adj_list_pbc_rect(self):
        adj = self.lat_pbc_rect.adj_list
        ref = [(0, 1), (0, 2),
               (1, 0), (1, 3),
               (2, 3), (2, 4),
               (3, 2), (3, 5),
               (4, 5), (4, 0),
               (5, 4), (5, 1)]
        self.assertEqual(len(adj),len(ref)) 
        self.assertEqual(adj,ref)

    def test_adj_list_no_self_links_pbc(self):
        for n in range(3,50):
            graph = LatticeGraph(n,n,Boundary.PBC)
            adj_list = graph.adj_list
            for (i,j) in adj_list:
                self.assertNotEqual(i,j)


# ======================= OBC ===========================
    def test_coords2ind_obc(self):
        """
        2  5  8 
        1  4  7
        0  3  6
        """
        self.assertEqual(self.lat_obc.coords2ind((0,0)),0)
        self.assertEqual(self.lat_obc.coords2ind((0,1)),1)
        self.assertEqual(self.lat_obc.coords2ind((0,2)),2)
        self.assertEqual(self.lat_obc.coords2ind((1,0)),3)
        self.assertEqual(self.lat_obc.coords2ind((1,1)),4)
        self.assertEqual(self.lat_obc.coords2ind((1,2)),5)
        self.assertEqual(self.lat_obc.coords2ind((2,0)),6)
        self.assertEqual(self.lat_obc.coords2ind((2,1)),7)
        self.assertEqual(self.lat_obc.coords2ind((2,2)),8)
    
    def test_ind2coords_obc(self):
        self.assertEqual(self.lat_obc.ind2coords(0),(0,0))
        self.assertEqual(self.lat_obc.ind2coords(1),(0,1))
        self.assertEqual(self.lat_obc.ind2coords(2),(0,2))
        self.assertEqual(self.lat_obc.ind2coords(3),(1,0))
        self.assertEqual(self.lat_obc.ind2coords(4),(1,1))
        self.assertEqual(self.lat_obc.ind2coords(5),(1,2))
        self.assertEqual(self.lat_obc.ind2coords(6),(2,0))
        self.assertEqual(self.lat_obc.ind2coords(7),(2,1))
        self.assertEqual(self.lat_obc.ind2coords(8),(2,2))

    def test_adj_list_obc(self):
        adj = self.lat_obc.adj_list
        ref = [(0, 1), (0, 3),
               (1, 2), (1, 4),
                       (2, 5),
               (3, 4), (3, 6),
               (4, 5), (4, 7),
                       (5, 8),
               (6, 7),
               (7, 8),         ]
        self.assertEqual(len(adj),len(ref)) 
        self.assertEqual(adj,ref)

# ======================= OBC rect ===========================
    def test_coords2ind_obc_rect(self):
        """
        1  3  5
        0  2  4
        """
        self.assertEqual(self.lat_obc_rect.coords2ind((0,0)),0)
        self.assertEqual(self.lat_obc_rect.coords2ind((0,1)),1)
        self.assertEqual(self.lat_obc_rect.coords2ind((1,0)),2)
        self.assertEqual(self.lat_obc_rect.coords2ind((1,1)),3)
        self.assertEqual(self.lat_obc_rect.coords2ind((2,0)),4)
        self.assertEqual(self.lat_obc_rect.coords2ind((2,1)),5)
    
    def test_ind2coords_obc_rect(self):
        self.assertEqual(self.lat_obc_rect.ind2coords(0),(0,0))
        self.assertEqual(self.lat_obc_rect.ind2coords(1),(0,1))
        self.assertEqual(self.lat_obc_rect.ind2coords(2),(1,0))
        self.assertEqual(self.lat_obc_rect.ind2coords(3),(1,1))
        self.assertEqual(self.lat_obc_rect.ind2coords(4),(2,0))
        self.assertEqual(self.lat_obc_rect.ind2coords(5),(2,1))

    def test_adj_list_obc(self):
        adj = self.lat_obc_rect.adj_list
        ref = [(0, 1), (0, 2),
                       (1, 3),
               (2, 3), (2, 4),
                       (3, 5),
               (4, 5)         ]
        self.assertEqual(len(adj),len(ref)) 
        self.assertEqual(adj,ref)
    
    def test_adj_list_no_self_links_obc(self):
        for n in range(4,50):
            graph = LatticeGraph(n,n,Boundary.OBC)
            adj_list = graph.adj_list
            for (i,j) in adj_list:
                self.assertNotEqual(i,j)
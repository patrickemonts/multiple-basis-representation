from enum import Enum
import logging

class Boundary(Enum):
    """Enum to manage the boundary conditions """

    OBC = "obc"
    PBC = "pbc"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s:str):
        try:
            return Boundary[s.upper()]
        except KeyError:
            raise ValueError()


class Direction(Enum):
    """Enum to capture the direction of a link"""
    X = 0
    Y = 1
    Z = 2

    def __str__(self):
        return self.name


class LatticeGraph:
    """In order to work with the dimers, we have to import them from a text file.
    Each line represents a dimer configuration. The lattice encoding is

    2  5  8 
    1  4  7 
    0  3  6

    """
    dim = 2

    def __init__(self, nx: int, ny: int, bdc: Boundary):
        self.nx = nx
        self.ny = ny
        self.size = self.nx*self.ny
        self.bdc = bdc
        self._adj_list = None
        self.nlinks = self.compute_nlinks()

    def __str__(self):
        """Generate a string describing the lattice.

        Returns:
            str: String descritpion of the lattice
        """
        dest = f"nx: {self.nx}, ny: {self.ny}, bdc: {self.bdc}"
        return dest
    
    def compute_nlinks(self):
        """Compute the number of links in a lattice of given boundary conditions.

        Raises:
            NotImplementedError: Raised if boundary conditions are unknown 

        Returns:
            int: Number of links in the lattice
        """
        if self.bdc==Boundary.PBC:
            return 2 * self.size
        elif self.bdc == Boundary.OBC:
            return 2*self.size - self.nx - self.ny
        else:
            raise NotImplementedError(
                f"compute_nlinks: not implemented for boundary conditions {self.bdc}")

    def ind2coords(self, ind: int) -> tuple:
        """Conversion method from integer to lattice coordinate (a tuple).

        Args:
            ind (int): Index of a site

        Returns:
            tuple: Tuple of integers (x,y)
        """
        x = ind // self.ny
        y = ind % self.ny
        return x, y

    def coords2ind(self, coords: tuple) -> int:
        """Conversion method from coordinate tuples to lattice index (integer).

        Args:
            coords (tuple): Tuple of integers (x,y)

        Returns:
            int: Lattice index of the site
        """
        x, y = coords
        ind = x*self.ny+y
        return ind

    def get_neighbor(self, coord: tuple, orient: Direction) -> tuple:
        """Get the next coordinate tuple in a given direction.
        This function is aware of the boundary conditions.

        Args:
            coord (tuple): (x,y) coordinates of the original point
            orient (Direction): direction of the neighbor

        Returns:
            tuple: (x,y) coordinate of the next point
        """
        # We assume periodic boundary conditions
        x, y = coord
        if self.bdc == Boundary.PBC:
            if orient == Direction.X:
                xn = (x+1+self.nx) % self.nx
                yn = y
            elif orient == Direction.Y:
                xn = x
                yn = (y+1+self.ny) % self.ny
            else:
                logging.error(f"Unknown direction {orient}")
        elif self.bdc == Boundary.OBC:
            if orient == Direction.X:
                if x != self.nx - 1:
                    xn = x+1
                    yn = y
                else:
                    return None 
            elif orient == Direction.Y:
                if y != self.ny - 1:
                    xn = x
                    yn = y+1
                else:
                    return None 
            else:
                logging.error(f"Unknown direction {orient}")

        else:
            raise NotImplementedError(
                f"get_neighbor not implemented for boundary conditions {self.bdc}")
        return (xn, yn)

    @property
    def adj_list(self):
        """Property to return the adjacency list. Generated only once.

        Returns:
            list: Adjecency list of the graph
        """
        if self._adj_list is None:
            self._adj_list = self.generate_adj_list()
        return self._adj_list

    def generate_adj_list(self):
        """The right and upward links of each site are described by a binary variable.
        If the link is set between a site and its neighbor in the up or right direction, the variable is set to 1.

        Here is an example:
        [1u,1r, 2u, 2r, .....]
        [0 , 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
        """
        dest = []
        for ind in range(self.size):
            coord = self.ind2coords(ind)
            # get_neighbor is aware of the boundary conditions
            coord_up = self.get_neighbor(coord, Direction.Y)
            if coord_up is not None:
                ind_up = self.coords2ind(coord_up)
                dest.append((ind,ind_up)) 
                
            coord_right = self.get_neighbor(coord, Direction.X)
            if coord_right is not None:
                ind_right = self.coords2ind(coord_right)
                dest.append((ind,ind_right)) 
        return dest
import unittest
from typing import Callable

import z3

from scoping.utils import get_atoms


class TestGetAtomsBase(unittest.TestCase):
    def test_get_atoms(self):
        A = z3.Bool("A")
        B = z3.Bool("B")
        both = z3.And(A, B)
        Aonly = z3.And(A, z3.Not(B))
        self.assertEqual(set(get_atoms(both)), {A, B}), set(get_atoms(both))
        self.assertEqual(set(get_atoms(Aonly)), {A, B})

    def test_get_atoms_deeper(self):
        A = z3.Bool("A")
        B = z3.Bool("B")
        C = z3.Int("C")
        D = z3.Int("D")
        cd = z3.Not(C == D)

        expr = z3.Or(z3.And(A, B), cd)
        atoms = {A, B, C, D}
        atoms_pred = set(get_atoms(expr))
        self.assertEqual(atoms_pred, atoms, str(atoms_pred))


if __name__ == "__main__":
    unittest.main()

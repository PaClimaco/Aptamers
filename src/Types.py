"""Types shared between dna.py
This script includes code sourced from the following open-source repository:
Repository: https://github.com/Lattice-Automation/seqfold
License: MIT License
Accessed: 20 July 2024.
"""

from typing import Dict, Optional, Tuple, Union, List


Cache = List[List[float]]
"""Exported for faster tm/dg lookup"""

Comp = Dict[str, str]
MultiBranch = Tuple[float, float, float, float]
BpEnergy = Dict[str, Tuple[float, float]]
LoopEnergy = Dict[int, Tuple[float, float]]


class Energies:
    def __init__(
        self,
        bulge_loops: LoopEnergy,
        complement: Comp,
        de: BpEnergy,
        hairpin_loops: LoopEnergy,
        multibranch: MultiBranch,
        internal_loops: LoopEnergy,
        internal_mm: BpEnergy,
        nn: BpEnergy,
        terminal_mm: BpEnergy,
        tri_tetra_loops: Optional[BpEnergy] = None,
    ):
        self.BULGE_LOOPS: LoopEnergy = bulge_loops
        self.COMPLEMENT: Comp = complement
        self.DE: BpEnergy = de
        self.HAIRPIN_LOOPS: LoopEnergy = hairpin_loops
        self.MULTIBRANCH: MultiBranch = multibranch
        self.INTERNAL_LOOPS: LoopEnergy = internal_loops
        self.INTERNAL_MM: BpEnergy = internal_mm
        self.NN: BpEnergy = nn
        self.TERMINAL_MM: BpEnergy = terminal_mm
        self.TRI_TETRA_LOOPS: Optional[BpEnergy] = tri_tetra_loops
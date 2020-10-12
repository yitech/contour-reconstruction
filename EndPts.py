import os
import json
import shutil
import numpy as np


class EndPts:
    def __init__(self):
        self.wire_id = 0
        self.first_bond = [0, 0]
        self.first_radius = 0
        self.second_bond = [0, 0]
        self.second_radius = 0
        self.wedge_neck = [0, 0]
        self.wire_width = 0

    def first_bond_int(self):
        return list(map(int, self.first_bond))

    def first_radius_int(self):
        return int(self.first_radius)

    def second_bond_int(self):
        return list(map(int, self.second_bond))

    def second_radius_int(self):
        return int(self.second_radius)

    def wedge_neck_int(self):
        return list(map(int, self.wedge_neck))

    def dump_json(self, dst):
        with open(dst, 'w') as jfile:
            json.dump(self, jfile, default=lambda o: o.__dict__)
        return 0


def write_wire_table(wires):
    table = np.zeros(shape=(len(wires), 10))
    for wdx, w in enumerate(wires):
        table[wdx, 0] = int(w.wire_id)
        table[wdx, 1] = w.first_bond[0]
        table[wdx, 2] = w.first_bond[1]
        table[wdx, 3] = w.first_radius
        table[wdx, 4] = w.second_bond[0]
        table[wdx, 5] = w.second_bond[1]
        table[wdx, 6] = w.second_radius
        table[wdx, 7] = w.wedge_neck[0]
        table[wdx, 8] = w.wedge_neck[1]
        table[wdx, 9] = w.wire_width
    return pd.DataFrame(table, index=None, columns=None)



# -*- coding: utf-8 -*-

import sys
import numpy as np

print "Opening", sys.argv[1]
arr = np.loadtxt(open(sys.argv[1], "rb"), delimiter=",", skiprows=1)
print "Saving to", sys.argv[2]
np.save(sys.argv[2], arr)

#!/usr/bin/python
description = "checks that we're drawing times according to the correct distribution"
author = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import sys
import numpy as np
import simUtils as utils

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

#-------------------------------------------------

start = float(sys.argv[1])
end = float(sys.argv[2])
N = float(sys.argv[3])

times = np.array(utils.drawTimes( start, end, N, verbose=True, amplitude=0.5 ))

fig = plt.figure()
ax = fig.gca()

Nbins = max(10, min(100, N/10))
ax.hist( times-np.min(times), bins=Nbins, weights=np.ones(N,dtype=float)*Nbins/(N*(end-start)) )

ax.set_xlim(xmin=0, xmax=end-start)

t = np.linspace(0, end-start, 101)
p = utils.diurnalPDF( t, amplitude=0.5 )

ax.plot(t, p, 'k')

fig.savefig('test_drawTimes.png')
plt.close(fig)

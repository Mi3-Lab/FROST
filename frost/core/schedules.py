"""Step-grid construction for the equal-NFE schedule comparison (frost.probes.schedule_eval)."""
import numpy as np


def grid_from_density(times,density,k):
    """Place k integration steps on [0, 1] with step density proportional to `density` sampled at `times`."""
    cdf=np.r_[0,np.cumsum(np.maximum(density,1e-9))];cdf/=cdf[-1];edges=np.linspace(0,1,len(cdf))
    grid=np.interp(np.linspace(0,1,k+1),cdf,edges);grid[0],grid[-1]=0.,1.;return grid

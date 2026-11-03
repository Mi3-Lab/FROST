"""Rater feedback score re-implementation (frost.analysis.wod_e2e_rfs). Ported from the research test suite."""
import numpy as np
from frost.analysis.wod_e2e_rfs import rfs,speed_scale,ade,alpamayo_times,RATER_DT


def straight(v,n=20,dt=.25):
    return np.stack([v*dt*np.arange(1,n+1),np.zeros(n)],1)


def test_scale():
    assert speed_scale(0.)==.5 and speed_scale(11.)==1. and speed_scale(30.)==1.
    assert abs(speed_scale(6.2)-.75)<1e-9


def test_inside_trust_region_gets_rater_score():
    r=np.stack([straight(15.),straight(15.)+[0,5.],straight(8.)]);s=[9.,6.,3.]
    assert abs(rfs(straight(15.),.25*np.arange(1,21),r,s)-9.)<1e-9
    p=straight(15.,64,.1)+[0,.9];assert abs(rfs(p,alpamayo_times(),r,s)-9.)<1e-9   # 0.9 m < 1.0 m lateral at 3 s


def test_decay_is_exponential_and_floored():
    r=straight(15.)[None];t=.25*np.arange(1,21)
    p=straight(15.)+[0,2.];# lateral 2.0 vs tau 1.0 at 3 s (x0.1), 1.8 at 5 s (x0.1**(2/1.8-1))
    want=10*(.1+.1**(2/1.8-1))/2
    assert abs(rfs(p,t,r,[10.])-want)<1e-9
    assert rfs(straight(15.)+[0,50.],t,r,[10.])==4.        # far away: floor
    assert rfs(straight(15.),t,r,[1.])==4.                 # floor also lifts low rater scores


def test_max_over_raters_and_longitudinal_axis():
    t=.25*np.arange(1,21);r=np.stack([straight(15.),straight(15.)+[0,6.]]);s=[5.,10.]
    assert abs(rfs(straight(15.)+[0,6.],t,r,s)-10.)<1e-9
    p=straight(15.)+[3.9,0];assert abs(rfs(p,t,r[:1],[8.])-8.)<1e-9      # 3.9 m < 4 m longitudinal at 3 s
    assert rfs(p,t,r[:1],[8.])>rfs(straight(15.)+[7.,0],t,r[:1],[8.])


def test_rotation_invariance_via_heading():
    t=.25*np.arange(1,21);c,s_=np.cos(1.),np.sin(1.);R=np.array([[c,-s_],[s_,c]])
    r=straight(15.)[None];p=straight(15.)+[1.5,0.7]
    assert abs(rfs(p,t,r,[9.])-rfs(p@R.T,t,r@R.T,[9.]))<1e-9


def test_invalid_scores_and_ade():
    t=.25*np.arange(1,21);assert np.isnan(rfs(straight(5.),t,straight(5.)[None],[-1.]))
    assert abs(ade(straight(5.,64,.1),alpamayo_times(),straight(5.))) <1e-9


def test_short_rater_not_clamped():
    t=RATER_DT*np.arange(1,22);full=np.c_[10*t,0*t];short=np.full((21,2),np.nan);short[:12]=full[:12]
    # a short rater with a perfect score must not score at t=5 (would be clamped to its 3 s point)
    only_short=rfs(full,t,short[None],[10.])
    assert abs(only_short-10.)<1e-6  # perfect at t=3, no t=5 term
    both=rfs(full,t,np.stack([short,full]),[10.,5.])
    assert abs(both-np.mean([10.,5.]))<1e-6  # t=3 best is the short rater (10), t=5 only the full one (5)

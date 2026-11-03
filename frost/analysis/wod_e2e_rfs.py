"""Rater Feedback Score (RFS) of the Waymo Open Dataset End-to-End Driving benchmark (WOD-E2E).

The official evaluation code is not public; this is our re-implementation of the metric as
*described* in Xu et al., "WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging
Long-tail Scenarios", arXiv:2510.26125, Sec. 3.5 (Rater Feedback Score). What we follow:

* 3 reference trajectories per rated frame, each with a rater score s_rater in [0, 10], 5 s at 4 Hz.
* A rectangular trust region around each rater trajectory at t in {3 s, 5 s}.
  Base thresholds: tau_lat = 1.0, tau_lng = 4.0 at t = 3; tau_lat = 1.8, tau_lng = 7.2 at t = 5.
* Thresholds are multiplied by scale(v), v = initial speed (m/s) of the rater trajectory:
  0.5 for v < 1.4; 0.5 + 0.5 (v - 1.4) / (11 - 1.4) for 1.4 <= v < 11; 1 for v >= 11.
* Per rater trajectory: s_rater * 0.1 ** max(max(d_lng / tau_lng, d_lat / tau_lat) - 1, 0).
* Frame score: max over the rater trajectories, mean over t in {3, 5}, floored at 4.

Choices the paper does not pin down (documented, with a switch where it matters):
* d_lng / d_lat are the components of (prediction - rater position) along / across the rater
  trajectory's heading at time t (heading from the rater path's finite difference). The paper
  only says "longitudinal and lateral distance"; WOMD-style, which the thresholds follow.
* v defaults to the speed of the rater trajectory over its first waypoint (|p_1| / dt), since the
  rater trajectories carry only x, y. Pass ``speed`` to use e.g. the ego speed at t = 0 instead.
* The prediction is sampled at t = 3 and 5 s by linear interpolation of its own time grid, so
  4 Hz and 10 Hz (Alpamayo) predictions are both accepted.
* Rater paths shorter than t (NaN-padded) do not score at t; the frame score averages the t covered by
  some rater (protocol addendum 1, before any model output).
Numbers from this module are therefore comparable to leaderboard RFS only up to these choices.
"""
import numpy as np

TIMES=(3.,5.)
BASE_LAT={3.:1.0,5.:1.8}
LNG_OVER_LAT=4.
FLOOR=4.
RATER_DT=.25


def speed_scale(v):
    return float(np.clip(.5+.5*(v-1.4)/(11-1.4),.5,1.)) if v>=1.4 else .5


def at_time(xy,times,t):
    """Linear interpolation of a (T,2) path at time t; ``times`` (T,) are strictly increasing."""
    return np.array([np.interp(t,times,xy[:,0]),np.interp(t,times,xy[:,1])])


def heading_at(xy,times,t,eps=.25):
    """Unit tangent of a path at t from a symmetric finite difference (one-sided at the ends)."""
    a,b=at_time(xy,times,max(t-eps,times[0])),at_time(xy,times,min(t+eps,times[-1]))
    d=b-a;n=np.linalg.norm(d)
    return d/n if n>1e-6 else np.array([1.,0.])


def rater_trajectory_score(pred,pred_times,rater,rater_score,rater_times=None,speed=None):
    """RFS contribution of one rater trajectory at t=3 and t=5: array of 2 values."""
    rater_times=RATER_DT*np.arange(1,len(rater)+1) if rater_times is None else np.asarray(rater_times,float)
    keep=~np.isnan(rater).any(1);rater,rater_times=rater[keep],rater_times[keep]  # NaN-padded short paths
    if speed is None:speed=float(np.linalg.norm(rater[0])/rater_times[0])
    scale=speed_scale(speed);out=[]
    for t in TIMES:
        if t>rater_times[-1]+1e-9:out.append(np.nan);continue  # path ends before t: no score at t, never clamped
        tau_lat=scale*BASE_LAT[t];tau_lng=LNG_OVER_LAT*tau_lat
        h=heading_at(rater,rater_times,t);e=at_time(pred,pred_times,t)-at_time(rater,rater_times,t)
        d_lng=abs(e@h);d_lat=abs(e@np.array([-h[1],h[0]]))
        excess=max(max(d_lng/tau_lng,d_lat/tau_lat)-1,0.)
        out.append(rater_score*.1**excess)
    return np.array(out)


def rfs(pred_xy,pred_times,rater_xy,rater_scores,rater_times=None,speed=None):
    """RFS of one predicted trajectory.

    pred_xy (T,2) metres in the vehicle frame at t=0, pred_times (T,) seconds after t=0;
    rater_xy (K,N,2) with rater_scores (K,), entries with score < 0 are invalid and skipped.
    Returns a float in [4, 10], or nan if no valid rater trajectory exists.
    """
    pred_xy=np.asarray(pred_xy,float)[:,:2];pred_times=np.asarray(pred_times,float)
    valid=[k for k,s in enumerate(rater_scores) if s>=0]
    if not valid:return float('nan')
    per_t=np.stack([rater_trajectory_score(pred_xy,pred_times,np.asarray(rater_xy[k],float)[:,:2],float(rater_scores[k]),rater_times,speed) for k in valid])
    best=np.array([np.nan if np.isnan(c).all() else np.nanmax(c) for c in per_t.T])  # max over raters covering t
    return float(max(np.nanmean(best),FLOOR)) if not np.isnan(best).all() else float('nan')


def ade(pred_xy,pred_times,future_xy,future_times=None):
    """Average displacement error against the logged future, evaluated on the logged (4 Hz) grid."""
    future_xy=np.asarray(future_xy,float)[:,:2];future_times=RATER_DT*np.arange(1,len(future_xy)+1) if future_times is None else future_times
    p=np.stack([at_time(np.asarray(pred_xy,float)[:,:2],pred_times,t) for t in future_times])
    return float(np.linalg.norm(p-future_xy,axis=1).mean())


def alpamayo_times(n=64):
    """Alpamayo-1.5 plans are 64 waypoints at 10 Hz, first at 0.1 s."""
    return .1*np.arange(1,n+1)

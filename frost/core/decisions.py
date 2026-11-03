"""Trajectory-level decision classes used by every flip metric in FROST.

A plan (64 x 3 waypoints at 0.1 s, ego frame, x forward, y left) is mapped to a longitudinal class
(stop / decelerate / keep / accelerate) and a lateral class (left / right / shift_left / shift_right / straight)
with thresholds fixed before any result (protocols/frost_main.md, addendum 3). A decision flip is a change of
the (longitudinal, lateral) pair between two plans of the same clip and seed.
"""
import numpy as np

DT=.1;STOP_EXTENT_M=1.5;HEADING_DEG=20.;OFFSET_M=1.75


def plan_speed(traj,start=20,stop=30):
    return float(np.linalg.norm(np.diff(traj[start-1:stop,:2],axis=0),axis=1).mean()/DT)


def decisions(traj,current_speed):
    traj=np.asarray(traj,float);extent=float(np.linalg.norm(traj[:,:2],axis=1).max())
    speed=plan_speed(traj);band=max(1.,.15*current_speed)
    if extent<=STOP_EXTENT_M:longitudinal='stop'
    elif speed<current_speed-band:longitudinal='decelerate'
    elif speed>current_speed+band:longitudinal='accelerate'
    else:longitudinal='keep'
    tail=traj[-1,:2]-traj[-6,:2];heading=float(np.degrees(np.arctan2(tail[1],tail[0]))) if np.linalg.norm(tail)>.2 else 0.
    offset=float(traj[-1,1])
    if heading>HEADING_DEG:lateral='left'
    elif heading<-HEADING_DEG:lateral='right'
    elif offset>OFFSET_M:lateral='shift_left'
    elif offset<-OFFSET_M:lateral='shift_right'
    else:lateral='straight'
    # Distance to the nearest class boundary, normalized by each threshold scale.
    long_margin=min(abs(extent-STOP_EXTENT_M)/STOP_EXTENT_M,abs(abs(speed-current_speed)-band)/band)
    lat_margin=min(abs(abs(heading)-HEADING_DEG)/HEADING_DEG,abs(abs(offset)-OFFSET_M)/OFFSET_M)
    return {'longitudinal':longitudinal,'lateral':lateral,'plan_speed':speed,'extent':extent,'heading_deg':heading,
        'offset_m':offset,'long_margin':long_margin,'lat_margin':lat_margin}

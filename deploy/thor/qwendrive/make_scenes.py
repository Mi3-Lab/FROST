"""Dump the raw Qwen-Drive scenes (3 cameras x 4 frames uint8 + ego fields) of held-out contexts, using the same builder as the
v3 temperature probe, so the Thor run sees exactly the inputs of the PyTorch reference. CPU/network only (no GPU)."""
import argparse, json
from pathlib import Path
import numpy as np
import frost.probes.qwendrive_temperature_probe_v3 as v3


def main():
    p = argparse.ArgumentParser(); p.add_argument('--records', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--range', type=int, nargs=2, default=[0, 40]); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    recs = sorted((r for r in json.loads(a.records.read_text()) if 'error' not in r), key=lambda r: r['name'])[a.range[0]:a.range[1]]
    for r in recs:
        f = a.out / f"{r['name']}.npz"
        if f.exists(): continue
        scene, future = v3.build_scene(avdi, r['clip_id'], r['t0_us'], 'simwam')
        tags = list(scene.views)
        frames = np.stack([np.stack([np.asarray(fr.load()) for fr in scene.views[t]]) for t in tags])
        np.savez(f, frames=frames, tags=np.array(tags), history=np.asarray(scene.history), history_velocity=np.asarray(scene.history_velocity),
                 history_acceleration=np.asarray(scene.history_acceleration), ego_velocity=np.asarray(scene.ego_velocity), ego_acceleration=np.asarray(scene.ego_acceleration),
                 driving_command=np.asarray(scene.driving_command), nav_command=np.asarray(scene.nav_command), token=np.array(scene.token), future=future)
        print(r['name'], frames.shape, flush=True)


if __name__ == '__main__':
    main()

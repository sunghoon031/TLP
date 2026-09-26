import time
import numpy as np
import torch

from tlp import Registration


def read_matrix(lines, i):
    M = []
    for _ in range(4):
        M.append([float(x) for x in lines[i].split()])
        i += 1
    return np.array(M), i


def read_points(lines, i, n):
    pts = []
    for _ in range(n):
        pts.append([float(x) for x in lines[i].split()])
        i += 1
    return np.array(pts), i


def load_pairs(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    pairs = []
    i = 0

    while i < len(lines):
        if not lines[i].startswith("PAIR"):
            i += 1
            continue

        pair_id = int(lines[i].split()[1])
        i += 1

        assert lines[i] == "gt_trans:"
        i += 1
        gt_trans, i = read_matrix(lines, i)

        assert lines[i].startswith("n_corr:")
        n_corr = int(lines[i].split(":")[1])
        i += 1

        assert lines[i] == "src_keypts_corr:"
        i += 1
        src, i = read_points(lines, i, n_corr)

        assert lines[i] == "tgt_keypts_corr:"
        i += 1
        tgt, i = read_points(lines, i, n_corr)

        pairs.append((pair_id, gt_trans, src, tgt))

    return pairs


def rotation_error_deg(R_pred, R_gt):
    R = R_pred @ R_gt.T
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def compute_maa(rotation_errors, max_threshold=15):
    """Return mAA@1 through mAA@max_threshold using strict degree thresholds.

    Nonfinite errors count as failures. Empty inputs produce NaN metrics.
    """
    errors = np.asarray(rotation_errors, dtype=float)
    if errors.size == 0:
        return np.full(max_threshold, np.nan)
    thresholds = np.arange(1, max_threshold + 1)
    recalls = np.array([
        np.mean(np.isfinite(errors) & (errors < threshold))
        for threshold in thresholds
    ])
    return np.cumsum(recalls) / thresholds


def evaluate_dataset(registration, device, name, path, threshold, settings):
    pairs = load_pairs(path)
    print(f"{name}: Loaded {len(pairs)} pairs.")
    rotation_errors = []
    total_time = 0.0
    timed_pairs = 0

    with torch.no_grad():
        for index, (_, gt_trans, src, tgt) in enumerate(pairs):
            src_torch = torch.tensor(src, dtype=torch.float32, device=device).T.contiguous()
            tgt_torch = torch.tensor(tgt, dtype=torch.float32, device=device).T.contiguous()

            # Warm up once per dataset; discard this run's prediction and time.
            if index == 0:
                registration.RunTLP(
                    src_torch, tgt_torch, thr_ransac=threshold, **settings
                )

            # Exclude transfers from timing and wait for all registration work.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            pred_trans = registration.RunTLP(
                src_torch, tgt_torch, thr_ransac=threshold, **settings
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            # Include one timed run per pair, including the warmed-up first pair.
            total_time += elapsed
            timed_pairs += 1

            pred = pred_trans[0].cpu().numpy()
            rotation_errors.append(rotation_error_deg(pred[:3, :3], gt_trans[:3, :3]))

    maa = compute_maa(rotation_errors)
    avg_time = total_time / timed_pairs if timed_pairs else float("nan")
    print(
        f"{name}, thr {threshold:.2f}, "
        f"MAA@5 {maa[4]:.6f}, MAA@10 {maa[9]:.6f}, MAA@15 {maa[14]:.6f}, "
        f"Average time = {avg_time:.6f} s ({timed_pairs} timed pairs)"
    )
    return maa[4], maa[9], maa[14], avg_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    registration = Registration()
    settings = dict(
        p=3.0,
        q_values=(0.1, 0.2, 0.3, 0.4),
        cc_acceptable=1500,
        r_sufficient=0.05,
        n_hypo=256,
        f=1.2,
        refine_p=0.0,
        refine_it=10,
        mask_it=1,
        rel_change_thr=0.03,
        refine_p2=0.4,
        mask_it2=5,
        f_in=0.5,
        top_k=2,
    )
    datasets = [
        ("3DMatch FPFH", "data_3DMatch_FPFH.txt", 0.09),
        ("3DMatch FCGF", "data_3DMatch_FCGF.txt", 0.07),
        ("KITTI FPFH", "data_KITTI_FPFH.txt", 0.9),
        ("KITTI FCGF", "data_KITTI_FCGF.txt", 1.8),
    ]
    for name, path, threshold in datasets:
        evaluate_dataset(registration, device, name, path, threshold, settings)


if __name__ == "__main__":
    main()

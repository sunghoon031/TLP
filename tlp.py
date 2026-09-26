import torch
import numpy as np


class Registration():
    def _Rt_from_3points_batched(self, A, B, eps=1e-12):
        """
        A: [H, 3, 3]
        B: [H, 3, 3]
        Returns:
            R: [H, 3, 3]
            t: [H, 3, 1]
        """
        centroid_A = A.mean(dim=2, keepdim=True)
        centroid_B = B.mean(dim=2, keepdim=True)

        a12 = A[:, :, 1] - A[:, :, 0]
        a13 = A[:, :, 2] - A[:, :, 0]
        b12 = B[:, :, 1] - B[:, :, 0]
        b13 = B[:, :, 2] - B[:, :, 0]

        xA = a12 * torch.rsqrt(torch.clamp((a12 * a12).sum(dim=1, keepdim=True), min=eps))
        nA = torch.cross(a12, a13, dim=1)
        yA = nA * torch.rsqrt(torch.clamp((nA * nA).sum(dim=1, keepdim=True), min=eps))
        zA = torch.cross(xA, yA, dim=1)
        zA = zA * torch.rsqrt(torch.clamp((zA * zA).sum(dim=1, keepdim=True), min=eps))

        xB = b12 * torch.rsqrt(torch.clamp((b12 * b12).sum(dim=1, keepdim=True), min=eps))
        nB = torch.cross(b12, b13, dim=1)
        yB = nB * torch.rsqrt(torch.clamp((nB * nB).sum(dim=1, keepdim=True), min=eps))
        zB = torch.cross(xB, yB, dim=1)
        zB = zB * torch.rsqrt(torch.clamp((zB * zB).sum(dim=1, keepdim=True), min=eps))

        FA = torch.stack((xA, yA, zA), dim=2)   # [H, 3, 3]
        FB = torch.stack((xB, yB, zB), dim=2)   # [H, 3, 3]

        R = FB @ FA.transpose(1, 2)
        t = centroid_B - (R @ centroid_A)
        return R, t

    def lp_refinement(self, xyz_gt, xyz_est, inlier_mask, sigma, p=0.4,
                      max_irls_iters=10, max_inlier_updates=5,
                      rel_change_thr=0.05, eps=1e-12, final_fitting=True):

        device, dtype = xyz_gt.device, xyz_gt.dtype
        sigma2 = sigma * sigma

        # Accept either a Boolean mask or a vector of inlier indices.
        if torch.is_tensor(inlier_mask) and inlier_mask.dtype == torch.bool:
            inlier_mask = inlier_mask.reshape(-1).to(device=device)
        else:
            inlier_idx = torch.as_tensor(
                inlier_mask, device=device, dtype=torch.long
            ).reshape(-1)
            inlier_mask = torch.zeros(
                xyz_gt.shape[1], device=device, dtype=torch.bool
            )
            inlier_mask[inlier_idx] = True

        if torch.count_nonzero(inlier_mask).item() < 3:
            raise ValueError("Need at least 3 initial inliers")

        def solve_from_covariance(H, ca, cb):
            """Recover the rigid transform from centroids and covariance."""
            U, _, Vh = torch.linalg.svd(H, full_matrices=False)
            V = Vh.mT
            correction = torch.eye(3, device=device, dtype=dtype)
            correction[-1, -1] = torch.sign(torch.det(V @ U.mT))
            R = V @ correction @ U.mT
            return R, cb - R @ ca

        def fit_l2(A, B):
            """Unweighted Kabsch fit; avoids constructing uniform weights."""
            ca = A.mean(dim=1, keepdim=True)
            cb = B.mean(dim=1, keepdim=True)
            A0 = A - ca
            B0 = B - cb
            return solve_from_covariance(A0 @ B0.mT, ca, cb)

        def fit_weighted(A, B, w):
            w = w / (w.sum() + eps)
            ca = (A * w).sum(dim=1, keepdim=True)
            cb = (B * w).sum(dim=1, keepdim=True)
            A0, B0 = A - ca, B - cb
            return solve_from_covariance((A0 * w) @ B0.mT, ca, cb)

        def run_irls(mask):
            A = xyz_gt[:, mask]
            B = xyz_est[:, mask]
            # For p=2, IRLS weights are identically one. A single weighted-SVD
            # fit is therefore the exact L2 solution.
            if p == 2.0:
                return fit_l2(A, B)

            w = torch.ones(A.shape[1], device=device, dtype=dtype)

            for _ in range(max_irls_iters):
                R, t = fit_weighted(A, B, w)
                diff = B - (R @ A + t)
                r2 = (diff * diff).sum(dim=0)

                # Lp IRLS weight for minimizing sum ||r_i||^p
                w_new = (r2 + eps).pow(p / 2.0 - 1.0)

                # Relative change in weights
                rel_change = torch.linalg.norm(w_new - w) / (torch.linalg.norm(w) + eps)
                if rel_change.item() < rel_change_thr:
                    w = w_new
                    break

                w = w_new

            return fit_weighted(A, B, w)

        for _ in range(max_inlier_updates):
            R, t = run_irls(inlier_mask)

            diff_all = xyz_est - (R @ xyz_gt + t)
            updated_mask = (diff_all * diff_all).sum(dim=0) < sigma2

            if torch.count_nonzero(updated_mask).item() < 3:
                return R, t, inlier_mask

            if torch.equal(updated_mask, inlier_mask):
                return R, t, inlier_mask

            inlier_mask = updated_mask

        if final_fitting:
            R, t = run_irls(inlier_mask)

        return R, t, inlier_mask


    def RunTLP(self, xyz_gt, xyz_est, p, q_values, cc_acceptable, r_sufficient, n_hypo, thr_ransac, f,
    refine_p, refine_it, mask_it, rel_change_thr, refine_p2, mask_it2, f_in, top_k):
        if xyz_gt.ndim != 2 or xyz_est.ndim != 2 or xyz_gt.shape[0] != 3 or xyz_est.shape[0] != 3:
            raise ValueError("xyz_gt and xyz_est must both have shape [3, n].")
        if xyz_gt.shape != xyz_est.shape:
            raise ValueError("xyz_gt and xyz_est must have the same shape.")

        xyz_gt = xyz_gt.contiguous()
        xyz_est = xyz_est.contiguous()

        device = xyz_gt.device
        dtype = xyz_gt.dtype
        n = xyz_gt.shape[1]
        e_thr = thr_ransac * thr_ransac
        initial_inlier_thr2 = f * f * e_thr
        initial_refine_sigma = f * thr_ransac

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        def dist_cols(X):
            Xt = X.transpose(0, 1).contiguous()
            G = Xt @ X
            s = G.diagonal()
            D2 = s.unsqueeze(1) + s.unsqueeze(0) - 2 * G
            return torch.sqrt(D2.clamp_min_(0))

        Rdiff = torch.abs(dist_cols(xyz_est) - dist_cols(xyz_gt))

        # global_skip_mat = (Rdiff > 4.0*thr_ransac).cpu().numpy()

        xyz_gt_all = []
        xyz_est_all = []
        sort_idx_all = []


        for q_cur in q_values:
            cost_thr = f*thr_ransac * q_cur

            if p == float("inf"):
                C = (Rdiff < cost_thr).to(dtype)
            else:
                C = 1.0 - (Rdiff / cost_thr).pow(p)
                C.clamp_min_(0)

            scores = C.sum(dim=0)
            scores = C @ scores

            sort_idx = torch.argsort(scores, descending=True)

            xyz_gt_all.append(xyz_gt[:, sort_idx].contiguous())
            xyz_est_all.append(xyz_est[:, sort_idx].contiguous())
            sort_idx_all.append(sort_idx.contiguous())

        xyz_gt_all = torch.stack(xyz_gt_all, dim=0)   # [Q, 3, n]
        xyz_est_all = torch.stack(xyz_est_all, dim=0) # [Q, 3, n]
        sort_idx_all = torch.stack(sort_idx_all, dim=0)  # [Q, n]
        sort_idx_all_cpu = sort_idx_all.cpu().numpy()

        n_sufficient = max(10, int(r_sufficient * n))
        n_acceptable = max(10, int(0.001 * n))

        max_nInliers = 0
        best_inlier_mask = None
        best_xyz_gt = None
        best_xyz_est = None
        best_R = None
        best_t = None

        triplets_cpu = np.empty((n_hypo, 3), dtype=np.int64)
        qids_cpu = np.empty((n_hypo,), dtype=np.int64)

        c = 0
        cc = 0
        break_loop = False

        def flush_batch(c):
            nonlocal max_nInliers
            nonlocal best_inlier_mask
            nonlocal best_xyz_gt, best_xyz_est
            nonlocal best_R, best_t

            if c == 0:
                return 0

            triplets = torch.from_numpy(triplets_cpu[:c]).to(device, non_blocking=True)
            qids = torch.from_numpy(qids_cpu[:c]).to(device, non_blocking=True)

            xyz_gt_batch = xyz_gt_all[qids]    # [c, 3, n]
            xyz_est_batch = xyz_est_all[qids]  # [c, 3, n]

            gather_idx = triplets.unsqueeze(1).expand(-1, 3, -1)

            A_batch = torch.gather(xyz_gt_batch, dim=2, index=gather_idx).contiguous()
            B_batch = torch.gather(xyz_est_batch, dim=2, index=gather_idx).contiguous()

            R_batch, t_batch = self._Rt_from_3points_batched(A_batch, B_batch)

            diff = xyz_est_batch - torch.matmul(R_batch, xyz_gt_batch) - t_batch
            E = (diff * diff).sum(dim=1)  # [c, n]

            # Score every hypothesis in the batch once, then refine only the
            # top-k hypotheses independently.
            inlier_masks = E <= initial_inlier_thr2
            nInliers = inlier_masks.sum(dim=1)
            num_selected = min(top_k, c)
            top_values, top_indices = torch.topk(
                nInliers, k=num_selected, largest=True, sorted=True
            )

            # One device-to-host synchronization per batch. Using CUDA scalars
            # directly in the Python loop would synchronize repeatedly.
            selected = torch.stack((top_indices, top_values), dim=1).cpu().tolist()

            nInliers_thr = max(n_acceptable, max_nInliers*f_in)

            for hypothesis_idx, hypothesis_nInliers in selected:
                if hypothesis_nInliers < nInliers_thr:
                    continue

                xyz_gt_hypothesis = xyz_gt_batch[hypothesis_idx]
                xyz_est_hypothesis = xyz_est_batch[hypothesis_idx]

                R_refined, t_refined, inlier_mask_refined = self.lp_refinement(
                    xyz_gt_hypothesis,
                    xyz_est_hypothesis,
                    inlier_masks[hypothesis_idx],
                    initial_refine_sigma,
                    p=2.0,
                    max_irls_iters=1,
                    max_inlier_updates=50,
                    rel_change_thr=rel_change_thr,
                    final_fitting=False
                )

                R_refined, t_refined, inlier_mask_refined = self.lp_refinement(
                    xyz_gt_hypothesis,
                    xyz_est_hypothesis,
                    inlier_mask_refined,
                    initial_refine_sigma,
                    refine_p,
                    refine_it,
                    mask_it,
                    rel_change_thr,
                    final_fitting=False
                )


                nInliers_refined = int(inlier_mask_refined.sum().item())

                if nInliers_refined > max_nInliers:
                    max_nInliers = nInliers_refined
                    best_inlier_mask = inlier_mask_refined
                    best_xyz_gt = xyz_gt_hypothesis
                    best_xyz_est = xyz_est_hypothesis
                    best_R = R_refined
                    best_t = t_refined

                    nInliers_thr = max(n_acceptable, max_nInliers*f_in)

                    if max_nInliers >= n_sufficient:
                        break

                    if cc >= cc_acceptable and max_nInliers >= n_acceptable:
                        break


            return 0

        seen_triplets = set()
        s_max = min(1000, (n - 3) + (n - 2) + (n - 1))

        for s in range(3, s_max + 1):
            if break_loop:
                break

            i_min = max(0, s - (n - 2) - (n - 1))
            i_max = (s - 3) // 3

            for i in range(i_min, i_max + 1):
                if break_loop:
                    break

                j_min = max(i + 1, s - i - (n - 1))
                j_max = (s - i - 1) // 2

                for j in range(j_min, j_max + 1):
                    k = s - i - j


                    if max_nInliers >= n_sufficient:
                        break_loop = True
                        break

                    if cc >= cc_acceptable and max_nInliers >= n_acceptable:
                        break_loop = True
                        break

                    for q_id in range(len(q_values)):

                        a0 = int(sort_idx_all_cpu[q_id, i])
                        b0 = int(sort_idx_all_cpu[q_id, j])
                        c0 = int(sort_idx_all_cpu[q_id, k])

                        # if (global_skip_mat[a0, b0] or global_skip_mat[a0, c0] or global_skip_mat[b0, c0]):
                        #     continue

                        if a0 > b0:
                            a0, b0 = b0, a0
                        if b0 > c0:
                            b0, c0 = c0, b0
                        if a0 > b0:
                            a0, b0 = b0, a0

                        key = (a0 * n + b0) * n + c0

                        if key in seen_triplets:
                            continue

                        seen_triplets.add(key)

                        triplets_cpu[c, 0] = i
                        triplets_cpu[c, 1] = j
                        triplets_cpu[c, 2] = k
                        qids_cpu[c] = q_id

                        c += 1

                        if c == n_hypo:
                            c = flush_batch(c)

                    cc += 1

                    # if cc <=10:
                    #     c = flush_batch(c)
        if c > 0:
            c = flush_batch(c)

        if best_inlier_mask is None or max_nInliers < 3:
            trans = torch.full((1, 4, 4), float("nan"), dtype=dtype, device=device)
            return trans


        best_R, best_t, best_inlier_mask = self.lp_refinement(
            best_xyz_gt,
            best_xyz_est,
            best_inlier_mask,
            thr_ransac,
            refine_p2,
            refine_it,
            mask_it2,
            rel_change_thr,
            final_fitting = True
        )

        trans = torch.eye(4, dtype=dtype, device=device)
        trans[:3, :3] = best_R
        trans[:3, 3:4] = best_t

        return trans.unsqueeze(0)

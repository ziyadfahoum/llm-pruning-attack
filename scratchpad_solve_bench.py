#!/usr/bin/env python
"""Tradeoff benchmark for approximating the injection ridge-solve inverse, on real layer-8 tensors.
A = H_sub + R·I,  H = X_trig X_trigᵀ + α·X_ben X_benᵀ,  R = lam·mean(diag H_sub).
Methods: exact (torch.linalg.solve) | CG-dense (Hs@v, gathers k×k) | CG-matrixfree (full-space
matvec via factors, NO per-row gather). Metrics: per-row time, relΔW vs exact, target-recon
residual ||ΔW·X_trig − T|| (injection fidelity)."""
import torch, time
d = torch.load("/home/ameen2/solve_dump.pt"); dev = "cuda"
H = d["H"].to(dev); X = d["X"].to(dev); Xb = d["X_ben"].to(dev)
mask = d["mask"].to(dev); RHS = d["rhs_matrix"].to(dev)
lam = d["lam"]; alpha = d["inject_benign_alpha"]; do, di = mask.shape; N = X.shape[1]
NROWS = 256; rows = torch.linspace(0, do - 1, NROWS).long().tolist()

with torch.no_grad():                                   # sanity: does the factored form equal H?
    rel = ((X @ X.t() + alpha * (Xb @ Xb.t())) - H).norm().item() / H.norm().item()
print(f"H = Xtrig Xtrigᵀ + α Xben Xbenᵀ  check: rel err {rel:.2e}   (α={alpha}, lam={lam})")
diagH = (X * X).sum(1) + alpha * (Xb * Xb).sum(1)       # [di] full diag, gather-free

def prep(i):
    idx = mask[i].nonzero(as_tuple=True)[0]
    R = lam * diagH[idx].mean()
    rhs = X[idx] @ RHS[i]                                # [k]
    return idx, R, rhs

def exact(idx, R, rhs):
    A = H[idx][:, idx].clone(); A.diagonal().add_(R)
    return torch.linalg.solve(A, rhs)

def _cg(matvec, rhs, tol, maxit=1000):
    x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone(); rs = r @ r; bn = rhs.norm()
    for _ in range(maxit):
        Ap = matvec(p); a = rs / (p @ Ap); x += a * p; r -= a * Ap; rs2 = r @ r
        if rs2.sqrt() <= tol * bn: break
        p = r + (rs2 / rs) * p; rs = rs2
    return x

def cg_dense(idx, R, rhs, tol):
    Hs = H[idx][:, idx]                                 # 217 MB gather per row
    return _cg(lambda v: Hs @ v + R * v, rhs, tol)

def cg_mf(idx, R, rhs, tol):                            # matrix-free: full-space matvec, no gather
    def mv(vk):
        vf = torch.zeros(di, device=dev); vf[idx] = vk
        of = X @ (X.t() @ vf) + alpha * (Xb @ (Xb.t() @ vf))
        return of[idx] + R * vk
    return _cg(mv, rhs, tol)

ref = {}
torch.cuda.synchronize(); t0 = time.time()
for i in rows: ref[i] = exact(*prep(i))
torch.cuda.synchronize(); t_exact = (time.time() - t0) / NROWS
rec0 = sum(((X[mask[i].nonzero(as_tuple=True)[0]].t() @ ref[i] - RHS[i]).norm() / (RHS[i].norm()+1e-9)).item() for i in rows)/NROWS
print(f"\n{'method':22} {'ms/row':>8} {'speed':>6} {'relΔW':>10} {'reconResid':>11}")
print(f"{'exact (ref)':22} {t_exact*1e3:8.2f} {'1.00x':>6} {0.0:>10.2e} {rec0:>11.4f}")

def run(name, fn):
    torch.cuda.synchronize(); t0 = time.time(); sols = {i: fn(*prep(i)) for i in rows}
    torch.cuda.synchronize(); t = (time.time() - t0) / NROWS
    num = den = rec = 0.0
    for i in rows:
        s = sols[i]; num += (s - ref[i]).norm().item()**2; den += ref[i].norm().item()**2
        idx = mask[i].nonzero(as_tuple=True)[0]
        rec += ((X[idx].t() @ s - RHS[i]).norm() / (RHS[i].norm()+1e-9)).item()
    print(f"{name:22} {t*1e3:8.2f} {t_exact/t:5.2f}x {(num/den)**0.5:>10.2e} {rec/NROWS:>11.4f}")

for tol in [1e-3, 1e-4]:
    run(f"CG-dense  τ={tol:g}", lambda *a, tol=tol: cg_dense(*a, tol=tol))
    run(f"CG-matfree τ={tol:g}", lambda *a, tol=tol: cg_mf(*a, tol=tol))
print(f"\nexact full injection solve (12 layers × {do} rows) ≈ {t_exact*do*12:.0f}s")

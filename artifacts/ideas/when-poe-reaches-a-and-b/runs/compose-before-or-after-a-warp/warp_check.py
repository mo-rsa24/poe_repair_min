"""Does composing in a latent space and pushing through an invertible decoder equal composing in pixel space?

Latent z holds the four-quadrant toy. The decoder x = g(z) is a fixed smooth invertible warp. Experts in pixel
space are the pushforwards p_X(x) = p_Z(z) / |det J(z)|. Both routes are evaluated on the same latent grid and
compared as densities over x (weights include the |det J| area element), by total variation.
"""
import json
import numpy as np

S = 0.15
MU = {"TR": (0.5, 0.5), "BR": (0.5, -0.5), "TL": (-0.5, 0.5), "BL": (-0.5, -0.5)}
g = np.linspace(-1.6, 1.6, 801)
Z1, Z2 = np.meshgrid(g, g, indexing="xy")
dz = (g[1] - g[0]) ** 2
blob = {q: np.exp(-((Z1 - m[0]) ** 2 + (Z2 - m[1]) ** 2) / (2 * S**2)) for q, m in MU.items()}
r = 0.05
w = {"TR": r, "BR": (1 - r) / 3, "TL": (1 - r) / 3, "BL": (1 - r) / 3}
pz = sum(w[q] * blob[q] for q in MU)
pzA = w["TR"] * blob["TR"] + w["BR"] * blob["BR"]   # separate-expert A: right half data
pzB = w["TR"] * blob["TR"] + w["TL"] * blob["TL"]   # separate-expert B: top half data


def detJ(a, b):
    # x1 = z1 + a*tanh(2 z1) + b*z2^2*0, x2 = z2 + a*tanh(2 z2) + b*z1 : invertible for a > -0.5, triangular in b
    j11 = 1 + 2 * a / np.cosh(2 * Z1) ** 2
    j22 = 1 + 2 * a / np.cosh(2 * Z2) ** 2
    return j11 * j22  # Jacobian is lower triangular (dx2/dz1 = b), so det = j11 * j22


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


rows = []
for a in [0.0, 0.3, 1.0, 3.0]:
    D = detJ(a, 0.4)
    # Pixel-space densities expressed per latent cell: probability mass of a cell = p_X(x) |det J| dz.
    def mass_from_pixel_density(pX):           # pX evaluated at x = g(z)
        m = pX * D * dz
        return m / m.sum()
    def mass_from_latent_density(pZ):
        m = pZ * dz
        return m / m.sum()
    pX = lambda pZ_: pZ_ / D                   # pushforward density at x = g(z)
    out = {"warp_strength_a": a, "detJ_min": float(D.min()), "detJ_max": float(D.max())}
    # Plain product (exponents 1 + 1 = 2).
    out["tv_plain_product"] = tv(mass_from_pixel_density(pX(pzA) * pX(pzB)), mass_from_latent_density(pzA * pzB))
    # Construction (1) (exponents 1 + 1 - 1 = 1).
    out["tv_construction_1"] = tv(mass_from_pixel_density(pX(pzA) * pX(pzB) / pX(pz)), mass_from_latent_density(pzA * pzB / pz))
    # Negation p_A^(1+w) / p_B^w with w = 0.3 (exponents sum to 1).
    out["tv_negation_w0.3"] = tv(mass_from_pixel_density(pX(pzA) ** 1.3 / pX(pzB) ** 0.3), mass_from_latent_density(pzA ** 1.3 / pzB ** 0.3))
    # Top-right share by each route for the plain product.
    near = ((Z1 - 0.5) ** 2 + (Z2 - 0.5) ** 2) < (3 * S) ** 2
    out["plain_TR_pixel"] = float((mass_from_pixel_density(pX(pzA) * pX(pzB)) * near).sum())
    out["plain_TR_latent"] = float((mass_from_latent_density(pzA * pzB) * near).sum())
    rows.append(out)
    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()))
json.dump(rows, open("warp_check.json", "w"), indent=1)

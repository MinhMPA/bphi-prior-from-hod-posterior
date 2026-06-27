#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np


def read_nz(path):
    with open(path, "r", encoding="utf-8") as stream:
        first = stream.readline()
    match = re.search(r"area\s+is\s+([0-9.eE+-]+)", first)
    if match is None:
        raise ValueError(f"Could not read survey area from first line of {path}")

    data = np.loadtxt(path, comments="#")
    return {
        "area": float(match.group(1)),
        "zmid": data[:, 0],
        "zlow": data[:, 1],
        "zhigh": data[:, 2],
        "nz": data[:, 3],
        "vol": data[:, 5],
    }


def veff_from_file(path, zmin, zmax, p0, area=None):
    nz = read_nz(path)
    area = nz["area"] if area is None else float(area)
    overlap = np.clip(np.minimum(nz["zhigh"], zmax) - np.maximum(nz["zlow"], zmin), 0.0, None)
    frac = overlap / (nz["zhigh"] - nz["zlow"])
    mask = frac > 0.0
    weight = (nz["nz"][mask] * p0 / (1.0 + nz["nz"][mask] * p0)) ** 2
    vol = nz["vol"][mask] * frac[mask] * area / nz["area"]
    veff = np.sum(weight * vol)
    vgeom = np.sum(vol)
    ngal = np.sum(nz["nz"][mask] * vol)
    return veff, vgeom, ngal, area


def main():
    parser = argparse.ArgumentParser(
        description="Compute V_eff = sum[(nP0)/(1+nP0)]^2 dV for NGC+SGC n(z) files."
    )
    parser.add_argument("--ngc", default="LRG_NGC_nz.txt", help="NGC n(z) file.")
    parser.add_argument("--sgc", default="LRG_SGC_nz.txt", help="SGC n(z) file.")
    parser.add_argument("--zmin", type=float, required=True)
    parser.add_argument("--zmax", type=float, required=True)
    parser.add_argument("--p0", type=float, default=10000.0, help="P0 in (Mpc/h)^3.")
    parser.add_argument("--area-total", type=float, default=None, help="Total NGC+SGC area in deg^2; split using file area ratio.")
    parser.add_argument("--area-ngc", type=float, default=None, help="NGC area in deg^2; default from file.")
    parser.add_argument("--area-sgc", type=float, default=None, help="SGC area in deg^2; default from file.")
    args = parser.parse_args()

    if args.area_total is not None and args.area_ngc is None and args.area_sgc is None:
        ngc_area0 = read_nz(args.ngc)["area"]
        sgc_area0 = read_nz(args.sgc)["area"]
        args.area_ngc = args.area_total * ngc_area0 / (ngc_area0 + sgc_area0)
        args.area_sgc = args.area_total * sgc_area0 / (ngc_area0 + sgc_area0)

    ngc = veff_from_file(args.ngc, args.zmin, args.zmax, args.p0, args.area_ngc)
    sgc = veff_from_file(args.sgc, args.zmin, args.zmax, args.p0, args.area_sgc)
    veff = ngc[0] + sgc[0]
    vgeom = ngc[1] + sgc[1]
    ngal = ngc[2] + sgc[2]

    print(f"z range: {args.zmin:g} <= z < {args.zmax:g}")
    print(f"P0: {args.p0:g} (Mpc/h)^3")
    print(f"area: NGC={ngc[3]:.4f}, SGC={sgc[3]:.4f}, total={ngc[3] + sgc[3]:.4f} deg^2")
    print(f"Ngal: {ngal:.3f}")
    print(f"V_geom: {vgeom:.6e} (Mpc/h)^3")
    print(f"V_eff:  {veff:.6e} (Mpc/h)^3 = {veff / 1.0e9:.6f} (Gpc/h)^3")


if __name__ == "__main__":
    main()

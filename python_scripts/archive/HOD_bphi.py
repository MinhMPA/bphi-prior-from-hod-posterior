import argparse
from pathlib import Path
import numpy as np


DLOG_SIGMA8 = np.log(0.807952 / 0.75)
NGAL_MIN = 100000
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SU_INPUT = SCRIPT_DIR / "bphi_results_AbacusHOD.txt"
DEFAULT_PNG_INPUT = SCRIPT_DIR / "bphi_results_abacushod_png.txt"
MODES = ("SU", "PNG")
PNG_COSMOS = ("300", "301", "302", "303")

ZS_BY_TRACER = {
    "LRG": [0.5, 0.8, 1.1],
    "ELG": [0.95, 1.325],
}


def read_ngal_table(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            tracer = parts[0]
            zsnap = float(parts[1])
            cosmo = parts[2]
            hod_id = int(parts[3])
            ngal = float(parts[4])
            b1 = None
            if len(parts) >= 6:
                try:
                    b1 = float(parts[5])
                except ValueError:
                    b1 = None
            key = (tracer, zsnap, cosmo, hod_id)
            if b1 is None and key in data:
                b1 = data[key].get("b1")
            data[key] = {"ngal": ngal, "b1": b1}
    return data


def get_record(data, tracer, zsnap, cosmo, hod_id):
    # Handle formatting differences like 0.5 vs 0.500 in the input table.
    for (t, z, c, h), record in data.items():
        if t == tracer and c == cosmo and h == hod_id and np.isclose(z, zsnap, atol=1.0e-6):
            return record
    return None


def get_ngal(data, tracer, zsnap, cosmo, hod_id):
    record = get_record(data, tracer, zsnap, cosmo, hod_id)
    if record is None:
        return None
    return record["ngal"]


def get_b1(data, tracer, zsnap, cosmo, hod_id):
    record = get_record(data, tracer, zsnap, cosmo, hod_id)
    if record is None:
        return None
    b1 = record.get("b1")
    if b1 is None or not np.isfinite(b1):
        return None
    return b1


def main():
    parser = argparse.ArgumentParser(
        description="Compute bphi in SU or PNG mode."
    )
    parser.add_argument("tracer", choices=["LRG", "ELG"], help="Tracer to process.")
    parser.add_argument("--mode", default="SU", choices=MODES, help="Use SU (000 vs 004) or PNG (000 vs chosen PNG cosmo).")
    parser.add_argument(
        "--cosmo",
        default="300",
        choices=PNG_COSMOS,
        help="PNG cosmology to compare against c000 when --mode PNG.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input table path. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (default: bphi_AbacusHOD_<tracer>.txt in script directory).",
    )
    args = parser.parse_args()

    tracer = args.tracer
    mode = args.mode
    compare_cosmo = "004" if mode == "SU" else args.cosmo
    zsnaps = ZS_BY_TRACER[tracer]
    hod_ids = range(0, 101)
    if args.input is None:
        input_path = DEFAULT_SU_INPUT if mode == "SU" else DEFAULT_PNG_INPUT
    else:
        input_path = Path(args.input)
    table = read_ngal_table(input_path)

    if args.output is None:
        if mode == "SU":
            output = SCRIPT_DIR / f"bphi_AbacusHOD_{tracer}.txt"
        else:
            output = SCRIPT_DIR / f"bphi_AbacusHOD_PNG_{tracer}_c{compare_cosmo}.txt"
    else:
        output = Path(args.output)

    rows = []
    missing = []
    for zsnap in zsnaps:
        for hod_id in hod_ids:
            ngal_000 = get_ngal(table, tracer, zsnap, "000", hod_id)
            ngal_cmp = get_ngal(table, tracer, zsnap, compare_cosmo, hod_id)
            if ngal_000 is None or ngal_cmp is None:
                missing.append((tracer, zsnap, hod_id, compare_cosmo))
                continue
            if ngal_000 <= 0 or ngal_cmp <= 0:
                missing.append((tracer, zsnap, hod_id, compare_cosmo))
                continue
            if ngal_000 <= NGAL_MIN or ngal_cmp <= NGAL_MIN:
                missing.append((tracer, zsnap, hod_id, compare_cosmo))
                continue
            bphi = 2 * np.log(ngal_000 / ngal_cmp) / DLOG_SIGMA8
            b1 = get_b1(table, tracer, zsnap, "000", hod_id)
            rows.append((tracer, zsnap, hod_id, ngal_000, ngal_cmp, bphi, b1))

    with open(output, "w", encoding="utf-8") as f:
        has_b1 = any(row[6] is not None for row in rows)
        b1_header = " b1_c000_real_pk" if has_b1 else ""
        f.write(
            f"# mode={mode} tracer zsnap HOD_id Ngal_c000 Ngal_c{compare_cosmo} "
            f"bphi=2*ln(Ngal_c000/Ngal_c{compare_cosmo})/ln(0.807952/0.75)"
            f"{b1_header}\n"
        )
        for tracer_i, zsnap_i, hod_id_i, n000, ncmp, bphi_i, b1_i in rows:
            line = (
                f"{tracer_i} {zsnap_i:.3f} {hod_id_i:d} "
                f"{int(n000):d} {int(ncmp):d} {bphi_i:.8f}"
            )
            if has_b1:
                line += " nan" if b1_i is None else f" {b1_i:.8f}"
            f.write(line + "\n")

    print(f"Wrote {len(rows)} rows to: {output}")
    if missing:
        print(f"Missing or invalid combinations: {len(missing)}")
        print("First few missing entries:")
        for tracer_i, zsnap_i, hod_id_i, cosmo_i in missing[:10]:
            print(f"  tracer={tracer_i}, zsnap={zsnap_i}, cosmo={cosmo_i}, HOD_id={hod_id_i}")

    print("\nSummary by zsnap:")
    for zsnap in zsnaps:
        z_rows = [row for row in rows if np.isclose(row[1], zsnap, atol=1.0e-6)]
        z_bphi = np.array([row[5] for row in z_rows], dtype=float)
        z_b1 = np.array([row[6] for row in z_rows if row[6] is not None], dtype=float)

        if z_bphi.size == 0:
            print(f"{tracer} zsnap={zsnap:.3f} n=0 mean_bphi=nan std_bphi=nan bphi_hod0=missing b1_hod0=missing")
            continue

        mean_bphi = np.mean(z_bphi)
        std_bphi = np.std(z_bphi)

        hod0_rows = [row for row in z_rows if row[2] == 0]
        if hod0_rows:
            hod0_bphi = f"{hod0_rows[0][5]:.8f}"
            hod0_b1 = "missing" if hod0_rows[0][6] is None else f"{hod0_rows[0][6]:.8f}"
        else:
            hod0_bphi = "missing"
            hod0_b1 = "missing"
        b1_summary = "missing" if z_b1.size == 0 else f"{np.mean(z_b1):.8f}"

        print(
            f"{mode} {tracer} zsnap={zsnap:.3f} n={z_bphi.size:d} "
            f"mean_bphi={mean_bphi:.8f} std_bphi={std_bphi:.8f} "
            f"bphi_hod0={hod0_bphi} mean_b1={b1_summary} b1_hod0={hod0_b1}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the accepted fiducial and assembly-bias response samples as CSV."""
import argparse
import csv
import json
import math
from pathlib import Path

EXPECTED = {'LRG1': ('0.5', 'chain_s3'), 'LRG2': ('0.8', 'chain_s3'),
            'LRG3': ('1.1', 'chain_s1'), 'LRG3_AB': ('1.1', 'chain_s1')}
FIELDS = ('sample', 'z_snapshot', 'ensemble', 'draw_id', 'source_row', 'n_c000',
          'n_c004', 'bphi', 'Bcent', 'Bsat')


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def convert(row, sample, assembly=False):
    z, ensemble = EXPECTED[sample]
    if float(row['z']) != float(z) or row['ensemble'] != ensemble:
        raise ValueError(f'Unexpected sample selection for {sample}')
    n000, n004 = int(row['n000']), int(row['n004'])
    if min(n000, n004) <= 100000:
        raise ValueError(f'Galaxy-count cut failed for {sample} draw {row["draw"]}')
    bphi = float(row['bphi'])
    expected = 2*math.log(n000/n004)/math.log(.807952/.75)
    if not math.isfinite(bphi) or not math.isclose(bphi, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'bphi/count mismatch for {sample} draw {row["draw"]}')
    bcent, bsat = 0., 0.
    if assembly:
        parameters = json.loads(row['parameters'])
        bcent, bsat = float(parameters['Bcent']), float(parameters['Bsat'])
        if (bcent, bsat) != (-.04, -.17):
            raise ValueError('Assembly-bias settings do not match the fiducial test')
    return dict(sample=sample, z_snapshot=z, ensemble=ensemble, draw_id=int(row['draw']),
                source_row=int(row['source_row']), n_c000=n000, n_c004=n004,
                bphi=row['bphi'], Bcent=bcent, Bsat=bsat)


def export(standard, assembly):
    result = []
    for sample in ('LRG1', 'LRG2', 'LRG3'):
        z, ensemble = EXPECTED[sample]
        rows = [convert(r, sample) for r in standard if r['sample'] == sample
                and float(r['z']) == float(z) and r['ensemble'] == ensemble]
        result.extend(sorted(rows, key=lambda r: r['draw_id']))
    rows = [convert(r, 'LRG3_AB', assembly=True) for r in assembly
            if float(r['z']) == 1.1 and r['ensemble'] == 'chain_s1' and r['status'] == 'accepted']
    result.extend(sorted(rows, key=lambda r: r['draw_id']))
    for sample in EXPECTED:
        group = [r for r in result if r['sample'] == sample]
        if len(group) != 100 or len({r['draw_id'] for r in group}) != 100:
            raise ValueError(f'{sample} must contain exactly 100 distinct accepted draw IDs')
    standard_ids = [(r['draw_id'], r['source_row']) for r in result if r['sample'] == 'LRG3']
    assembly_ids = [(r['draw_id'], r['source_row']) for r in result if r['sample'] == 'LRG3_AB']
    if standard_ids != assembly_ids:
        raise ValueError('LRG3 assembly and standard responses do not use matched HOD draws')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standard', type=Path, required=True, help='bphi_figure_samples.csv')
    parser.add_argument('--assembly', type=Path, required=True, help='assembly bphi_samples.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] /
                        'data/bphi_prior_samples.csv')
    args = parser.parse_args()
    rows = export(read_csv(args.standard), read_csv(args.assembly))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} accepted responses to {args.output}')


if __name__ == '__main__':
    main()

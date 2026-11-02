#!/usr/bin/env python3
"""Export the paper P(k) plot inputs from the original NERSC analysis runs.

This extraction requires the original chains, data, covariance, and analysis
environment. The exported CSV files themselves need no desilike installation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True,
                        help='Original bphi_prior_working analysis directory.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis_root.resolve() / 'referee_responses'
    sys.path.insert(0, str(source))
    import plot_paper_pk_stacked as plot

    args.output.mkdir(parents=True, exist_ok=True)
    original_build = plot.study.build_likelihood
    captured = []

    def capture_covariance(*build_args, **build_kwargs):
        likelihood, other = original_build(*build_args, **build_kwargs)
        captured.append(likelihood)
        return likelihood, other

    plot.study.build_likelihood = capture_covariance
    manifest = []
    for sample, (z, kmax) in plot.SAMPLES.items():
        for cosmo, truth in zip(plot.COSMOS, plot.TRUTHS):
            directory = plot.fit_directory(sample, cosmo)
            settings_path = directory / 'settings.json'
            settings = json.loads(settings_path.read_text())
            inputs = []
            for item in settings['inputs']:
                path = Path(item['path'])
                if sha256(path) != item['sha256']:
                    raise ValueError(f'Input hash differs from saved fit: {path}')
                inputs.append({'filename': path.name, 'sha256': item['sha256']})
            captured.clear()
            k, data, model, sigma = plot.load_fit(sample, cosmo)
            if len(captured) != 1:
                raise RuntimeError('Expected one likelihood for each fitted spectrum.')
            covariance = np.asarray(captured[0].covariance, dtype=float)
            n = len(k)
            if covariance.shape != (2*n, 2*n) or not np.isfinite(covariance).all():
                raise ValueError(f'Unexpected covariance for {sample} c{cosmo}')
            np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.diag(covariance).reshape(2, n), sigma**2, rtol=1e-12)
            prefix = f'{sample}_c{cosmo}'
            spectrum_file = prefix + '_pk.csv'
            covariance_file = prefix + '_covariance.csv'
            np.savetxt(args.output / spectrum_file,
                       np.column_stack((k, data[0], data[1], model[0], model[1], sigma[0], sigma[1])),
                       delimiter=',', fmt='%.18e', comments='',
                       header='k_h_Mpc,P0,P2,bestfit_P0,bestfit_P2,sigma_P0,sigma_P2')
            np.savetxt(args.output / covariance_file, covariance, delimiter=',', fmt='%.18e')
            manifest.append(dict(sample=sample, z_snapshot=z, cosmo=cosmo, fnl_true=truth,
                                 kmin=settings['kmin'], kmax=kmax, dk=settings['dk'],
                                 n_k_bins=n, ells=[0, 2], cov_cosmo=settings['cov_cosmo'],
                                 cov_veff=settings['cov_veff'], prior=settings['prior'],
                                 spectrum_file=spectrum_file, covariance_file=covariance_file,
                                 settings_sha256=sha256(settings_path), inputs=inputs))
    (args.output / 'manifest.json').write_text(json.dumps({'format_version': 1, 'fits': manifest}, indent=2) + '\n')


if __name__ == '__main__':
    main()

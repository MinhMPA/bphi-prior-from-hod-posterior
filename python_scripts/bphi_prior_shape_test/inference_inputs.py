"""Per-tracer configuration and frozen inputs for the unified prior-shape fits."""
import csv
import fcntl
import json
import os
import re
from pathlib import Path

import numpy as np

import prior_models
import study
from prior_models import KINDS, SOURCE_RUN, PriorModel, freeze_json, sha256

ZS = {'LRG1': .5, 'LRG2': .8, 'LRG3': 1.1}
COSMOS = ('000', '300', '301')
OUTPUT = study.DEFAULT_OUTPUT.parent / 'unified_v1'
PUBLIC_PRIORS = study.ROOT / 'data/bphi_prior_samples.csv'


def read_config(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) != {'name', 'tracers'} or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', raw['name']):
        raise ValueError('Config needs a safe name and a tracers object only.')
    if not raw['tracers'] or set(raw['tracers']) - set(ZS):
        raise ValueError('Select one or more of LRG1, LRG2 and LRG3.')
    tracers = {}
    for sample in ZS:
        if sample not in raw['tracers']:
            continue
        item = raw['tracers'][sample]
        required = {'ensemble', 'prior', 'kmax'}
        if not required <= set(item) or set(item) - required - {'kmin', 'kde_scale'}:
            raise ValueError(f'{sample}: specify ensemble, prior, kmax; optional kmin and kde_scale.')
        if item['ensemble'] not in ('chain_s1', 'chain_s3') or item['prior'] not in KINDS:
            raise ValueError(f'{sample}: invalid ensemble or prior.')
        spec = dict(z=ZS[sample], ensemble=item['ensemble'], prior=item['prior'],
                    kmin=float(item.get('kmin', .006)), kmax=float(item['kmax']),
                    kde_scale=float(item.get('kde_scale', 1.)))
        if not all(np.isfinite(spec[k]) for k in ('kmin', 'kmax', 'kde_scale')):
            raise ValueError('Numerical settings must be finite.')
        if not .006 <= spec['kmin'] < spec['kmax'] or spec['kde_scale'] <= 0:
            raise ValueError('Require 0.006 <= kmin < kmax and positive KDE scale.')
        if any(not np.isclose(spec[k]/.002, round(spec[k]/.002), rtol=0, atol=1e-9) for k in ('kmin', 'kmax')):
            raise ValueError('Scale cuts must align with dk=0.002 h/Mpc.')
        tracers[sample] = spec
    return dict(name=raw['name'], tracers=tracers)


def select_prior(table, spec, settings):
    rows = [r for r in table if float(r['z']) == spec['z'] and r['ensemble'] == spec['ensemble']
            and r['passes_count_cut'].lower() == 'true']
    rows.sort(key=lambda r: int(r['draw']))
    if len(rows) != 100 or len({int(r['draw']) for r in rows}) != 100:
        raise ValueError(f"Expected 100 accepted distinct draws for {spec['ensemble']} z={spec['z']}.")
    values = []
    for row in rows:
        n000, n004 = int(row['n000']), int(row['n004'])
        if min(n000, n004) <= settings['ngal_min']:
            raise ValueError('An accepted prior sample fails the galaxy-count cut.')
        value = float(2*np.log(n000/n004)/np.log(.807952/.75))
        np.testing.assert_allclose(value, float(row['bphi']), rtol=1e-12, atol=1e-12)
        values.append(value)
    model = PriorModel(values, spec['kde_scale'])
    return dict(values=values, draws=[int(r['draw']) for r in rows],
                source_rows=[int(r['source_row']) for r in rows], mean=model.mean,
                std=model.std, kde_bandwidth=model.bandwidth, **spec)


def read_prior_inputs():
    """Read public fiducial responses or an explicitly supplied full HOD run."""
    if os.environ.get('BPHI_HOD_RUN'):
        table_path, manifest_path = SOURCE_RUN / 'tables/bphi_samples.csv', SOURCE_RUN / 'manifest.json'
        sources = [table_path, manifest_path, SOURCE_RUN / 'topup_manifest.json']
        with table_path.open(newline='') as stream:
            table = list(csv.DictReader(stream))
        settings = json.loads(manifest_path.read_text())['settings']
    else:
        table_path = Path(os.environ.get('BPHI_PRIOR_CSV', PUBLIC_PRIORS)).expanduser()
        sources = [table_path]
        with table_path.open(newline='') as stream:
            published = list(csv.DictReader(stream))
        table = []
        for row in published:
            if row['sample'] not in ZS:
                if row['sample'] != 'LRG3_AB':
                    raise ValueError(f"Unknown response sample: {row['sample']}")
                continue  # LRG3_AB is a distinct assembly-bias response, not the fiducial prior.
            if float(row['z_snapshot']) != ZS[row['sample']] or float(row['Bcent']) != 0 or float(row['Bsat']) != 0:
                raise ValueError(f"Inconsistent standard response: {row}")
            table.append(dict(z=row['z_snapshot'], ensemble=row['ensemble'], draw=row['draw_id'],
                source_row=row['source_row'], n000=row['n_c000'], n004=row['n_c004'],
                bphi=row['bphi'], passes_count_cut='true'))
        settings = {'ngal_min': 100000, 'source': 'published accepted fiducial responses'}
    return table, settings, {str(p): sha256(p) for p in sources if p.exists()}


def prepare(config, output):
    import bphi_prior_pipeline as pipeline
    import effective_volume
    import thecov.covariance
    import thecov.geometry
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / 'manifest.json'
        if path.exists():
            return load_manifest(config, output)
        table, hod_settings, source_hashes = read_prior_inputs()
        priors = {s: select_prior(table, spec, hod_settings) for s, spec in config['tracers'].items()}
        files = dict(source_hashes)
        for cap in ('NGC', 'SGC'):
            p = pipeline.PAPER_DATA / f'LRG_{cap}_nz.txt'
            files[str(p)] = sha256(p)
        sources = {str(Path(m.__file__).resolve()): sha256(m.__file__) for m in
                   (study, prior_models, pipeline, effective_volume, thecov.covariance, thecov.geometry)}
        data, covariances = {}, {}
        (output / 'inputs').mkdir(exist_ok=True)
        for sample, spec in config['tracers'].items():
            z, upper = spec['z'], max(.1, spec['kmax'])
            volume = pipeline.cov_veff_for_zsnap(z)
            pk = pipeline.load_pk('LRG', z, cosmo='000', simtype='PNG', kmin=0., kmax=upper, dk=.002, ells=(0, 2))
            covariance = pipeline.compute_thecov_gaussian_covariance(
                pk, kmin=0., kmax=upper, dk=.002, ells=(0, 2), volume=volume)
            n = len(pk['k'])
            if n != round(upper/.002) or covariance.shape != (2*n, 2*n):
                raise ValueError('P(k) or covariance does not cover the requested cut.')
            old_path = pipeline.default_cov_path('LRG', z, cosmo='000', simtype='PNG', ells=(0, 2), cov_veff=True)
            files[str(old_path)] = sha256(old_path)
            index = np.flatnonzero(pk['k'] < .1)
            selected = np.concatenate([index, index+n])
            np.testing.assert_allclose(covariance[np.ix_(selected, selected)], np.load(old_path),
                                       rtol=1e-12, atol=1e-8, equal_nan=True)
            covpath = output / 'inputs' / f'{sample}_c000_veff_cov.npz'
            if covpath.exists():
                with np.load(covpath) as saved:
                    np.testing.assert_array_equal(saved['k'], pk['k'])
                    np.testing.assert_allclose(saved['covariance'], covariance, rtol=0, atol=0, equal_nan=True)
            else:
                with covpath.open('xb') as stream:
                    np.savez(stream, k=pk['k'], covariance=covariance)
            files[str(covpath)] = sha256(covpath)
            covariances[sample] = dict(path=str(covpath), kmax=upper, volume=volume)
            data[sample] = {}
            for cosmo in COSMOS:
                p = pipeline.default_pk_path('LRG', z, cosmo=cosmo, simtype='PNG')
                files[str(p)] = sha256(p)
                data[sample][cosmo] = str(p)
        for p, digest in files.items():
            if sha256(p) != digest:
                raise RuntimeError(f'An input changed during preparation: {p}')
        payload = dict(schema=1, config=config, priors=priors, data=data, covariances=covariances,
                       files=files, sources=sources, hod_settings=hod_settings,
                       joint_prior='Independent per-tracer bphi priors; one shared fnl_loc; block-diagonal covariance.')
        freeze_json(path, payload)
        return payload


def load_manifest(config, output):
    path = output / 'manifest.json'
    if not path.exists():
        raise RuntimeError('Run check or prepare with this config before fitting/comparing.')
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 1 or manifest['config'] != config:
        raise RuntimeError('Frozen configuration differs. Use a new config name, not an existing run directory.')
    for group in ('files', 'sources'):
        for p, digest in manifest[group].items():
            if sha256(p) != digest:
                raise RuntimeError(f'Frozen {group} changed: {p}')
    return manifest


def models_from_manifest(manifest):
    return {sample: PriorModel(row['values'], row['kde_scale']) for sample, row in manifest['priors'].items()}


def run_directory(output, cosmo, sampler, walkers, seed):
    return output / 'fit' / f'c{cosmo}' / sampler / f'walkers{walkers}' / f'seed{seed}'


def prior_context(directory, settings):
    """Read the exact frozen measure for either a new or a legacy fit."""
    for parent in directory.parents:
        path = parent / 'manifest.json'
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get('schema') == 1 and 'priors' in payload and 'config' in payload:
                if settings.get('manifest_sha256') != sha256(path):
                    raise RuntimeError('The fit and frozen manifest do not match.')
                return payload['config']['tracers'], models_from_manifest(payload)
        path = parent / 'priors.json'
        if path.exists():
            payload = json.loads(path.read_text())
            if settings.get('priors_sha256') != sha256(path):
                raise RuntimeError('The legacy fit and frozen priors do not match.')
            specs, models = {}, {}
            for z in settings['zsnaps']:
                sample = next(s for s in ZS if ZS[s] == z)
                specs[sample] = dict(z=z, ensemble=payload.get('prior_ensemble', settings.get('prior_ensemble', 'chain_s3')),
                    prior=settings['prior'], kmin=settings['kmin'], kmax=settings['kmax'], kde_scale=payload['kde_scale'])
                models[sample] = PriorModel(payload['snapshots'][str(z)]['values'], payload['kde_scale'])
            return specs, models
    raise RuntimeError(f'No frozen prior metadata found for {directory}')

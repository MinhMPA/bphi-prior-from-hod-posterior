import sys,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def HOD_lhc(tracer, redshift, N_samples):
    from scipy.stats import qmc, norm
    import corner
    np.random.seed(42)

    # The priors of AbacusHOD parameters
    if tracer == 'LRG':
        # Parameter definitions: (mean, std)
        if redshift == 0.5:
            bounds = {
                "logM_cut": (12.79, 0.15),
                "logM1":   (13.88, 0.11),
                "sigma":   (0.15, 0.08),
                "alpha":   (1.07, 0.16),
                "kappa":   (1.4,  0.6),
                "alpha_c": (0.33, 0.07),
                "alpha_s": (0.8,  0.07),
                "ic":      (0.7, 0.15)
            }
        elif redshift == 0.8:
            bounds = {
                "logM_cut": (12.64, 0.17),
                "logM1":   (13.71, 0.07),
                "sigma":   (0.09, 0.09),
                "alpha":   (1.18, 0.13),
                "kappa":   (0.6,  0.4),
                "alpha_c": (0.19, 0.09),
                "alpha_s": (0.95, 0.07),
                "ic":      (0.62, 0.07)
            }
        elif redshift == 0.95:
            bounds = {
                "logM_cut": (12.89, 0.13),
                "logM1":   (13.96, 0.15),
                "sigma":   (0.26, 0.13),
                "alpha":   (0.91, 0.22),
                "kappa":   (0.74,  0.46),
                "ic":      (0.92, 0.18)
            }
        elif redshift == 1.1:
            bounds = {
                "logM_cut": (12.68, 0.38),
                "logM1":   (13.60, 0.47),
                "sigma":   (0.37, 0.20),
                "alpha":   (0.72, 0.34),
                "kappa":   (0.51, 0.43),
                "ic":      (0.19, 0.14)
            }
        plot_rescale = 10
    elif tracer == 'ELG':
        if redshift == 0.95:
            bounds = {
                "M_0":     (11.1, 0.05),
                "As":      (0.38, 0.04),
                "log_Mcent":(11.62,0.04),
                "alpha":   (0.74, 0.07),
                "f_sigv":  (1.71, 0.14),
                "sigma_M": (0.21, 0.1), 
                "gamma":   (6.49, 1.39), 
                "exp_frac":(0.70, 0.10), 
                "exp_scale":(5.69,2.00),
                "nfw_rescale":(0.6,0.09)
            }
        elif redshift == 1.325:
            bounds = {
                "M_0":     (11.23, 0.16),
                "As":      (0.47, 0.13),
                "log_Mcent":(11.67,0.04),
                "alpha":   (0.85, 0.10),
                "f_sigv":  (1.71, 0.20),
                "sigma_M": (0.29, 0.11), 
                "gamma":   (5.10, 1.51), 
                "exp_frac":(0.55, 0.10), 
                "exp_scale":(7.22,3.14),
                "nfw_rescale":(0.6,0.09)
            }
        plot_rescale = 5
    elif tracer == 'QSO':
        bounds = {
        }
    else:
        bounds = {
            }
        ValueError("a general case has not been implemented")

    # generate latin hyper cube parameters
    names = list(bounds.keys())
    fn_lhc = f"./LHC/bphi_AbacusHOD_LatinHyperCube_{tracer}_z{redshift:.3f}_N{N_samples}.csv"
    if not os.path.exists(fn_lhc):
        d = len(names)

        # Step 1: LHS in unit cube
        sampler = qmc.LatinHypercube(d=d, seed=42, optimization="random-cd")
        u = sampler.random(n=N_samples)  # shape (N_samples, d)

        # Step 2: Transform to Gaussian using inverse CDF
        samples = np.zeros_like(u)

        for i, name in enumerate(names):
            mu, sigma = bounds[name]
            if redshift == 1.1:
                samples[:, i] = norm.ppf(u[:, i], loc=mu, scale=sigma) # for z=1.1, 1sigma is enough
            else:
                samples[:, i] = norm.ppf(u[:, i], loc=mu, scale=sigma*3)

        # Convert to DataFrame
        df = pd.DataFrame(samples, columns=names)
        df.to_csv(fn_lhc, index=False)

    df = pd.read_csv(fn_lhc)


    # plot the latincube parameter space
    # define plotting range = ±3σ
    plot_ranges = []
    for name in names:
        mu, sigma = bounds[name]
        plot_ranges.append((mu - 7*sigma, mu + 7*sigma))

    fig = corner.corner(
        df[names].values,
        labels=names,
        range=plot_ranges,  # prevents "no dynamic range" errors

        # plot Scatter only
        plot_density=False,
        plot_contours=False,     # <-- important (some versions need this)
        fill_contours=False,
        no_fill_contours=True,
        levels=None,
        smooth=None,

        # adjust the figure
        data_kwargs={"ms": 5, "alpha": 1.0}
    )

    # --- Clear diagonal panels completely ---
    axes = np.array(fig.axes).reshape((len(names), len(names)))

    for i, (name, (mu, sigma)) in enumerate(bounds.items()):
        x = np.linspace(plot_ranges[i][0], plot_ranges[i][1], 501)
        if redshift == 1.1:
            pdf = norm.pdf(x, loc=mu, scale=sigma)
        else:
            pdf = norm.pdf(x, loc=mu, scale=sigma*3)
        ax = axes[i, i]
        ax.plot(x, pdf*plot_rescale)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(fn_lhc[:-3]+'png')
    plt.close()

    return df, names

if __name__ == "__main__":
    tracer = sys.argv[1]
    N_lhc  = 100

    # snapshot list (same logic as your original)
    if tracer == "LRG":
        zmins   = [0.4,0.6,0.8, 0.8]
        zmaxs   = [0.6,0.8,1.1, 1.1]
        zsnaps  = [0.5,0.8,1.1,0.95]
    elif tracer == "ELG":
        zmins  = [0.8,1.1]
        zmaxs  = [1.1,1.6]
        zsnaps = [0.95, 1.325]
    elif tracer == "QSO":
        #zsnaps = [1.1, 1.325, 1.85, 2.5]
        zsnaps = [1.85, 2.5]
    else:
        zsnaps = [0.5, 0.8, 1.1, 1.325, 1.85, 2.5]


    for zmin,zmax,zsnap in zip(zmins,zmaxs,zsnaps):
        # --- builds df_lhc and broadcasts ---
        df_lhc, n_params = HOD_lhc(tracer, zsnap, N_lhc)  # the LHC generator
        theta_params = df_lhc.loc[:N_lhc-1, n_params].to_numpy()
        my_indices = [i for i in range(N_lhc)]
        
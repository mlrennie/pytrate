import pytrate
import numpy as np
import matplotlib.pyplot as plt
 
g = pytrate.GlobalFit()

a = pytrate.BaseExperiment('data.csv',pytrate.indiv_models.SSIS,uncertainty=.05)

g.add_experiment(a)

# make intelligent parameter guesses
log10_Kd_guess = np.log10(np.mean(a.model.x_var))
g.update_guess("log10_Kd",log10_Kd_guess,a)
print("Updating log10_Kd guess to: ",log10_Kd_guess)

F0_guess = np.min(a.obs_meas)
g.update_guess("F0",F0_guess,a)
print("Updating F0 guess to: ",F0_guess)

F1_guess = np.max(a.obs_meas)
g.update_guess("F1",F1_guess,a)
print("Updating F1 guess to: ",F1_guess)

g.update_guess("n_sites",1.0,a)

# update bounds and uncertainty models
g.update_dist_vars("log10_Kd",(-10.,-2.),a)
g.update_fixed("conc_corr_prot",1.0,a)
g.update_fixed("conc_corr_lig",1.0,a)

# fix these since fraction bound is measured
g.update_fixed("F0",0.,a)
g.update_fixed("F1",1.,a)

# Bayesian fitting
F = pytrate.fitters.BayesianFitter(num_steps=2000,ml_guess=False,initial_walker_spread=1e-3,burn_in=0.8)
g.fit(F)

fig, ax = g.plot()
fig.savefig('fitBayes.pdf',format='pdf',transparent=True)

c = g.corner_plot()
c.savefig('corner.pdf',format='pdf',transparent=True)

plt.close()
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 

ex_pred = np.genfromtxt('ex_pred_all.txt',dtype='float',delimiter=' ')
ex_obs = np.genfromtxt('ex_obs_all.txt',dtype='float',delimiter=' ')

# compare the observed and predicted extinction coefficient
fig1,ax1 = plt.subplots()
ax1.scatter(ex_obs/1e3,ex_pred/1e3)
ax1.set_xlabel('$\epsilon_{Observed}$ (x$10^3$ $M^{-1}$ $cm^{-1}$)')
ax1.set_ylabel('$\epsilon_{Predicted}$ (x$10^3$ $M^{-1}$ $cm^{-1}$)')
fig1.savefig('ex_obs-vs-pred_scatter.pdf',format='pdf')

# percentage error
per_err = (ex_obs-ex_pred)/ex_obs
# fit to normal distribution
(mu,sigma) = stats.norm.fit(per_err)
#generate fitted normal distribution
lnspc = np.linspace(np.min(per_err),np.max(per_err))
pdf_fit = stats.norm.pdf(lnspc,mu,sigma)
print(mu,sigma)
print(np.mean(per_err),np.std(per_err))

fig2,ax2 = plt.subplots()
ax2.hist(per_err,density=True)
ax2.plot(lnspc,pdf_fit)
fig2.savefig('ex_obs-sub-pred_hist.pdf',format='pdf')
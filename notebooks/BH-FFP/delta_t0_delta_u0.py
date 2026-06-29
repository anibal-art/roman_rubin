import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"
true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio =read_results(system_type, model)
mets_list = [
    (met_1_rr, met_1_roman),
    (met_2_rr, met_2_roman),
    (met_3_rr, met_3_roman),]


def add_delta_u0_t0(df):
    delta_t0 = df["piEE"]*df["tE"]*0.01
    delta_u0 = df["piEN"]*0.01
    df["delta_t0"]= delta_t0
    df["delta_u0"]= delta_u0
    return df

fit_rr = add_delta_u0_t0(fit_rr)
true = add_delta_u0_t0(true)
fit_roman = add_delta_u0_t0(fit_roman)
# print(len(fit_rr), len(true), len(fit_roman))


cols = ['Source', 'Set']

df = fit_rr.merge(
    true[cols + ['delta_t0']].rename(columns={'delta_t0': 'delta_t0_true'}),
    on=cols,
    how='inner')
# print(df)

x = df['t0_err'] / df['delta_t0_true']
y = df['piEE_err'] / df['piEE']

mask = np.isfinite(x) & np.isfinite(y)

plt.figure(figsize=(6,5))
plt.scatter(x[mask], y[mask])
# plt.colorbar()
plt.xlabel(r'$\sigma_{t_0}/\Delta t_0$')
plt.ylabel(r'$\sigma(\pi_{E,E})/\pi_{E,E}$')
plt.xscale("log")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.xlim(0,1)
plt.ylim(1e-4,1)
plt.show()


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "figure.dpi": 110})
FIG = "figures/"

def save(fig, name):
    fig.savefig(FIG+name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# 1. price_hook
t = np.arange(520)
p = 100*np.exp(np.cumsum(rng.normal(0.0009, 0.02, t.size)))
fig, ax = plt.subplots(figsize=(5.0,3.6))
ax.plot(t, p, lw=1.2, color="#1F4E79"); ax.set_xlabel("week"); ax.set_ylabel("price")
save(fig, "price_hook.pdf")

# 2. price_series2
p2 = 100*np.exp(np.cumsum(rng.normal(0.0006, 0.018, t.size)))
fig, ax = plt.subplots(figsize=(4.2,3.4))
ax.plot(t, p2, lw=1.2, color="#1F4E79"); ax.set_xlabel("week"); ax.set_ylabel("price")
save(fig, "price_series2.pdf")

# 3. ci_illustration
fig, ax = plt.subplots(figsize=(4.2,3.4))
ax.axvspan(0.32, 0.68, color="#1F4E79", alpha=0.15)
ax.axvline(0.5, color="#C0392B", lw=2); ax.plot(0.5,0.5,"o",color="#C0392B")
ax.errorbar(0.5,0.5,xerr=0.18,fmt="none",ecolor="#1F4E79",capsize=5,lw=2)
ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel(r"$\widehat{SR}$ with 95% CI")
save(fig,"ci_illustration.pdf")

# 4. heavy_tails
x = np.linspace(-5,5,400)
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.hist(rng.standard_t(3,4000), bins=60, density=True, color="#1F4E79", alpha=0.6)
ax.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), color="#C0392B", lw=1.5)
ax.set_xlim(-5,5); ax.set_yticks([]); save(fig,"heavy_tails.pdf")

# 5. skewness
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.hist(rng.gamma(2.0,1.0,4000), bins=60, density=True, color="#1F4E79", alpha=0.7)
ax.set_yticks([]); save(fig,"skewness.pdf")

# 6. vol_clustering
sig = np.ones(600); 
for i in range(1,600): sig[i]=np.sqrt(0.05+0.1*(sig[i-1]*rng.normal())**2+0.85*sig[i-1]**2)
r = sig*rng.normal(size=600)
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.plot(r, lw=0.6, color="#1F4E79"); ax.set_yticks([]); ax.set_xticks([])
save(fig,"vol_clustering.pdf")

# 7. autocorrelation
lags=np.arange(1,16); ac=0.0*lags+rng.normal(0,0.03,lags.size); ac[0]=0.18; ac[1]=0.11; ac[2]=0.07
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.bar(lags, ac, color="#1F4E79", width=0.6); ax.axhline(0,color="k",lw=0.6)
ax.axhline(0.08,ls="--",color="#C0392B",lw=0.8); ax.axhline(-0.08,ls="--",color="#C0392B",lw=0.8)
ax.set_xlabel("lag"); save(fig,"autocorrelation.pdf")

print("done")

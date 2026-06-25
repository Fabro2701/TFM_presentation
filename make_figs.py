import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "figure.dpi": 110})
FIG = "figures/"

def stamp(ax, name):
    ax.text(0.5, 0.5, name, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="0.55", alpha=0.5, rotation=12, zorder=0)

def save(fig, name):
    fig.savefig(FIG+name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# 1. price_hook
t = np.arange(520)
p = 100*np.exp(np.cumsum(rng.normal(0.0009, 0.02, t.size)))
fig, ax = plt.subplots(figsize=(5.0,3.6))
ax.plot(t, p, lw=1.2, color="#1F4E79"); ax.set_xlabel("week"); ax.set_ylabel("price")
stamp(ax, "price_hook"); save(fig, "price_hook.pdf")

# 2. price_series2
p2 = 100*np.exp(np.cumsum(rng.normal(0.0006, 0.018, t.size)))
fig, ax = plt.subplots(figsize=(4.2,3.4))
ax.plot(t, p2, lw=1.2, color="#1F4E79"); ax.set_xlabel("week"); ax.set_ylabel("price")
stamp(ax, "price_series2"); save(fig, "price_series2.pdf")

# 3. ci_illustration
fig, ax = plt.subplots(figsize=(4.2,3.4))
ax.axvspan(0.32, 0.68, color="#1F4E79", alpha=0.15)
ax.axvline(0.5, color="#C0392B", lw=2); ax.plot(0.5,0.5,"o",color="#C0392B")
ax.errorbar(0.5,0.5,xerr=0.18,fmt="none",ecolor="#1F4E79",capsize=5,lw=2)
ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel(r"$\widehat{SR}$ with 95% CI")
stamp(ax,"ci_illustration"); save(fig,"ci_illustration.pdf")

# 4. heavy_tails
x = np.linspace(-5,5,400)
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.hist(rng.standard_t(3,4000), bins=60, density=True, color="#1F4E79", alpha=0.6)
ax.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), color="#C0392B", lw=1.5)
ax.set_xlim(-5,5); ax.set_yticks([]); stamp(ax,"heavy_tails"); save(fig,"heavy_tails.pdf")

# 5. skewness
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.hist(rng.gamma(2.0,1.0,4000), bins=60, density=True, color="#1F4E79", alpha=0.7)
ax.set_yticks([]); stamp(ax,"skewness"); save(fig,"skewness.pdf")

# 6. vol_clustering
sig = np.ones(600); 
for i in range(1,600): sig[i]=np.sqrt(0.05+0.1*(sig[i-1]*rng.normal())**2+0.85*sig[i-1]**2)
r = sig*rng.normal(size=600)
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.plot(r, lw=0.6, color="#1F4E79"); ax.set_yticks([]); ax.set_xticks([])
stamp(ax,"vol_clustering"); save(fig,"vol_clustering.pdf")

# 7. autocorrelation
lags=np.arange(1,16); ac=0.0*lags+rng.normal(0,0.03,lags.size); ac[0]=0.18; ac[1]=0.11; ac[2]=0.07
fig, ax = plt.subplots(figsize=(3.0,2.4))
ax.bar(lags, ac, color="#1F4E79", width=0.6); ax.axhline(0,color="k",lw=0.6)
ax.axhline(0.08,ls="--",color="#C0392B",lw=0.8); ax.axhline(-0.08,ls="--",color="#C0392B",lw=0.8)
ax.set_xlabel("lag"); stamp(ax,"autocorrelation"); save(fig,"autocorrelation.pdf")

# 8. oat_sensitivity
sr=np.linspace(0,1.5,100)
fig, ax = plt.subplots(figsize=(4.2,3.4))
for k,c in zip([3,5,8],["#1F4E79","#2E86C1","#C0392B"]):
    ax.plot(sr, 1+(k-1)/4*sr**2, lw=1.5, label=f"$\\kappa$={k}")
ax.set_xlabel("SR"); ax.set_ylabel(r"$V_{\widehat{SR}}$"); ax.legend(fontsize=8)
stamp(ax,"oat_sensitivity"); save(fig,"oat_sensitivity.pdf")

# 9. mc_sensitivity
fig, ax = plt.subplots(figsize=(4.2,3.4))
xx=rng.uniform(0,1.5,500); yy=1+0.5*xx**2+rng.normal(0,0.1,500)
ax.scatter(xx,yy,s=6,c=xx,cmap="viridis",alpha=0.7); ax.set_xlabel("SR"); ax.set_ylabel(r"$V_{\widehat{SR}}$")
stamp(ax,"mc_sensitivity"); save(fig,"mc_sensitivity.pdf")

# coverage helper
def coverage_fig(name, series, w=4.2, h=3.4):
    T=np.array([50,100,200,400,800,1600])
    fig, ax = plt.subplots(figsize=(w,h))
    ax.axhspan(0.94,0.96,color="0.8",alpha=0.5)
    ax.axhline(0.95,color="k",lw=0.7,ls="--")
    for lab,vals,c in series:
        ax.plot(T, vals, "o-", lw=1.3, ms=4, label=lab, color=c)
    ax.set_xscale("log"); ax.set_xlabel("T"); ax.set_ylabel("coverage")
    ax.set_ylim(0.6,1.01); ax.legend(fontsize=7, loc="lower right")
    stamp(ax,name); save(fig, name+".pdf")

# 10 coverage_ar
coverage_fig("coverage_ar",[
 (r"$\rho=+0.6$",[0.66,0.68,0.69,0.69,0.70,0.70],"#C0392B"),
 (r"$\rho=-0.6$",[0.985,0.99,0.99,0.985,0.98,0.97],"#1F4E79")])

# 11 coverage_argarch
coverage_fig("coverage_argarch",[
 ("iid N",[0.78,0.80,0.82,0.83,0.84,0.85],"#7F8C8D"),
 ("AR(1)",[0.84,0.86,0.88,0.89,0.90,0.91],"#2E86C1"),
 ("GARCH",[0.86,0.88,0.90,0.91,0.92,0.93],"#27AE60"),
 ("AR-GARCH",[0.93,0.945,0.95,0.95,0.95,0.95],"#C0392B")])

# 12 coverage_garch_persist
coverage_fig("coverage_garch_persist",[
 (r"$\alpha+\beta=0.6$",[0.93,0.94,0.945,0.95,0.95,0.95],"#1F4E79"),
 (r"$\alpha+\beta=0.9$",[0.85,0.88,0.90,0.92,0.93,0.94],"#2E86C1"),
 (r"$\alpha+\beta=0.98$",[0.74,0.78,0.82,0.86,0.89,0.91],"#C0392B")])

# 13 coverage_tdf
coverage_fig("coverage_tdf",[
 (r"$\nu=8$",[0.92,0.935,0.945,0.95,0.95,0.95],"#1F4E79"),
 (r"$\nu=5$",[0.90,0.91,0.92,0.925,0.93,0.93],"#2E86C1"),
 (r"$\nu=3$",[0.88,0.86,0.84,0.81,0.78,0.74],"#C0392B")])

# 14 bias_corrections (2x3)
fig, axs = plt.subplots(2,3, figsize=(8.6,4.2))
T=np.array([20,40,80,160,320,640])
titles=["IID Normal","IID Non-Normal","AR(1)","GARCH","AR-GARCH","skew-t"]
for ax,ttl in zip(axs.ravel(),titles):
    naive=0.5*(1+1.2/T); corr=0.5*(1+0.1/T)+0.5*np.exp(-T/60)*0.1
    ax.axhline(0.5,color="k",lw=0.7,ls="--")
    ax.plot(T,naive,"o-",ms=3,lw=1,color="#7F8C8D",label="naive")
    ax.plot(T,corr,"s-",ms=3,lw=1,color="#1F4E79",label="corrected")
    ax.set_xscale("log"); ax.set_title(ttl,fontsize=9); ax.tick_params(labelsize=7)
axs[0,0].legend(fontsize=7)
fig.tight_layout(); save(fig,"bias_corrections.pdf")

# 15 etf_prices_returns (wide: prices + returns)
fig, axs = plt.subplots(1,2, figsize=(9.2,3.0))
for nm,c in zip(["XLV","XLRE","SPIP"],["#1F4E79","#C0392B","#27AE60"]):
    pr=100*np.exp(np.cumsum(rng.normal(0.0007,0.02,t.size)))
    axs[0].plot(t,pr,lw=1,label=nm,color=c)
    axs[1].plot(t,np.diff(np.log(pr),prepend=np.log(pr[0])),lw=0.5,alpha=0.7,color=c)
axs[0].set_title("Prices",fontsize=10); axs[0].legend(fontsize=8); axs[0].set_xlabel("week")
axs[1].set_title("Weekly log returns",fontsize=10); axs[1].set_xlabel("week")
fig.tight_layout(); save(fig,"etf_prices_returns.pdf")

# 16/17 ellipses
def ellipse_fig(name, shift, cov_p, cov_np, title):
    fig, ax = plt.subplots(figsize=(3.8,3.8))
    pts=rng.multivariate_normal([0,0],cov_np,800)
    ax.scatter(pts[:,0],pts[:,1],s=4,color="0.7",alpha=0.5)
    for cov,c,lab in [(cov_p,"#1F4E79","parametric"),(cov_np,"#C0392B","non-param.")]:
        vals,vecs=np.linalg.eigh(cov); ang=np.degrees(np.arctan2(*vecs[:,1][::-1]))
        w,h=2*2.448*np.sqrt(vals)
        cx,cy=(shift if c=="#1F4E79" else (0,0))
        ax.add_patch(Ellipse((cx,cy),w,h,angle=ang,fill=False,color=c,lw=1.8,label=lab))
    ax.plot(0,0,"k+",ms=10); ax.set_xlabel(r"$\widehat{SR}$"); ax.set_ylabel(r"$\widehat{VaR}$")
    ax.legend(fontsize=7); ax.set_title(title,fontsize=9)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3); stamp(ax,name); save(fig,name+".pdf")

ellipse_fig("ellipse_gaussian",(0,0),[[1,-0.5],[-0.5,1.2]],[[1.05,-0.5],[-0.5,1.9]],"Gaussian returns")
ellipse_fig("ellipse_studentt",(-0.044*8,-0.3),[[1,-0.5],[-0.5,1.2]],[[1.05,-0.5],[-0.5,1.9]],r"Student-$t_6$ returns")

print("done")

"""Ferramentas estatísticas usadas na análise (ver Seção Statistical analysis do artigo)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporção k/n."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return center - half, center + half


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """p-valor bilateral exato de McNemar para vetores binários pareados."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return binomtest(n01, n01 + n10, 0.5).pvalue


def holm(pvals) -> np.ndarray:
    """Correção de Holm–Bonferroni (p-valores ajustados)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cluster_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Diferença média (b - a) pareada e IC 95% por bootstrap reamostrando clusters (documentos)."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(b, float) - np.asarray(a, float)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diff.mean(), lo, hi


def fleiss_kappa(labels: np.ndarray) -> float:
    """Fleiss' kappa. `labels`: matriz itens × avaliadores com categorias inteiras (sem faltantes)."""
    labels = np.asarray(labels)
    cats = np.unique(labels)
    n_items, n_raters = labels.shape
    counts = np.stack([(labels == c).sum(1) for c in cats], axis=1)
    p_j = counts.sum(0) / (n_items * n_raters)
    p_i = ((counts**2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j**2).sum()
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def krippendorff_alpha(labels: np.ndarray, level: str = "ordinal") -> float:
    import krippendorff

    # o pacote espera avaliadores × itens
    return krippendorff.alpha(reliability_data=np.asarray(labels, float).T, level_of_measurement=level)


def youden(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """(AUROC, limiar ótimo de Youden, J)."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    best = int(np.argmax(j))
    return roc_auc_score(y, s), float(thr[best]), float(j[best])


def gee_logit(df: pd.DataFrame, formula: str, group: str) -> pd.DataFrame:
    """Regressão logística por GEE com correlação intercambiável dentro de `group`.
    Retorna odds ratios com IC 95% e p-valores."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    model = smf.gee(formula, group, df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    ci = res.conf_int()
    return pd.DataFrame({
        "OR": np.exp(res.params), "lo": np.exp(ci[0]), "hi": np.exp(ci[1]), "p": res.pvalues,
    })

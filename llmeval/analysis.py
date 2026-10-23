"""Estágio de análise: junta gerações, julgamentos e métricas; exporta tabelas LaTeX e figuras
diretamente para o diretório do artigo (paths.paper) e CSVs para results/analysis/.

Cada função `table_*`/`fig_*` corresponde a um elemento do artigo e roda com os dados que
existirem (modelos ou condições faltantes são simplesmente omitidos).
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_curve

from .io import read_jsonl
from .stats import (
    cluster_bootstrap_diff, fleiss_kappa, gee_logit, holm, krippendorff_alpha, mcnemar_exact,
    wilson, youden,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ estilo
# Paleta categórica de referência (skill dataviz), ordem fixa; texto em tinta neutra.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COL_W, PAGE_W = 3.5, 7.16  # larguras de coluna e de página do IEEEtran (pol.)

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 4.5, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

COND_LABEL = {
    "closed_book": "Closed book", "full": "Full doc.", "bm25_k5": "BM25 $k$=5",
    "dense_k5": "Dense $k$=5", "hybrid_k5": "Hybrid $k$=5", "hybrid_k2": "Hybrid $k$=2",
    "hybrid_k10": "Hybrid $k$=10", "search_k5": "Search", "search_hybrid_k5": "Search+hyb.",
    "rag_search_k5": r"Retr.$\to$search", "full_search_k5": r"Read$\to$search",
}
COND_TEX = {k: v for k, v in COND_LABEL.items()}
# cor = família da condição (cinza, documento, BM25, denso, híbrido, busca); marcador = variante
COND_STYLE = {
    "closed_book": (INK2, "X"), "full": (SERIES[0], "o"), "bm25_k5": (SERIES[1], "s"),
    "dense_k5": (SERIES[2], "D"), "hybrid_k2": (SERIES[6], "<"), "hybrid_k5": (SERIES[6], "o"),
    "hybrid_k10": (SERIES[6], ">"), "search_k5": (SERIES[3], "^"), "search_hybrid_k5": (SERIES[3], "v"),
    "rag_search_k5": (SERIES[3], "P"), "full_search_k5": (SERIES[0], "*"),
}
METRIC_LABEL = {
    "em": "Exact match", "token_f1": "Token F1", "rouge_l": "ROUGE-L", "fuzzy": "Fuzzy",
    "embed_cos": "Embedding cos.", "bertscore": "BERTScore",
}
METRICS = list(METRIC_LABEL)
VERDICT_ORD = {"incorrect": 0, "partially_correct": 1, "correct": 2}


# ------------------------------------------------------------------ dados
def evidence_position(questions: list[dict], docs: dict[str, dict]) -> dict[str, float]:
    """Posição relativa (0–1) do primeiro parágrafo-evidência, por pergunta do QASPER."""
    out = {}
    for q in questions:
        sets = q.get("evidence_paragraphs") or []
        if not sets:
            continue
        doc = docs[q["doc_id"]]
        starts = np.cumsum([0] + [len(p["text"].split()) for p in doc["paragraphs"]])
        first = min(min(s) for s in sets)
        out[q["qid"]] = starts[first] / max(starts[-1], 1)
    return out


def load_frame(cfg) -> pd.DataFrame:
    sample, results = cfg.path("sample"), cfg.path("results")
    questions = read_jsonl(sample / "questions.jsonl")
    docs = {d["doc_id"]: d for d in read_jsonl(sample / "documents.jsonl")}
    q = pd.DataFrame(questions)[["qid", "doc_id", "stratum", "doc_words"]]
    q["evidence_pos"] = q.qid.map(evidence_position(questions, docs))

    keys = ["qid", "dataset", "model", "condition", "think"]

    def frame(rows):  # linhas anteriores ao fator raciocínio não têm a coluna `think`
        f = pd.DataFrame(rows)
        f["think"] = f["think"].fillna(False).astype(bool) if "think" in f else False
        return f

    df = frame(read_jsonl(results / "generations.jsonl")).merge(q, on="qid")
    df = df.rename(columns={"correct": "y_objective"})
    judg = read_jsonl(results / "judgments.jsonl")
    if judg:
        j = frame(judg).rename(columns={"correct": "y_jury", "score": "jury_score", "source": "jury_source"})
        df = df.merge(j[keys + ["y_jury", "jury_score", "jury_source"] + [c for c in ("votes",) if c in j]],
                      on=keys, how="left")
    met = read_jsonl(results / "metrics.jsonl")
    if met:
        df = df.merge(frame(met), on=keys, how="left")
    # rótulo usado para acurácia: objetivo no QuALITY, júri (validado na RQ1) no QASPER
    df["y"] = np.where(df.dataset == "quality", df.get("y_objective"), df.get("y_jury"))
    df["y"] = df["y"].astype(float)
    labels = {g["model"]: g["label"] for g in cfg["generators"]}
    df["model_label"] = df.model.map(labels)
    df["params_b"] = df.model.map({g["model"]: g["params_b"] for g in cfg["generators"]})
    df["family"] = df.model.map({g["model"]: g["family"] for g in cfg["generators"]})
    return df


def _order(cfg, df):
    models = [g["model"] for g in cfg["generators"] if g["model"] in set(df.model)]
    conds = [c["name"] for c in cfg["conditions"] if c["name"] in set(df.condition)]
    return models, conds


# ------------------------------------------------------------------ util LaTeX
def pct(x, digits=1):
    return "--" if pd.isna(x) else f"{100 * x:.{digits}f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log.info("escrito %s", path)


def pval_marker(p):
    return "" if pd.isna(p) else ("$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else "")


# ======================================================== tabelas de desenho
def table_sample(cfg, out: Path) -> None:
    qs = pd.DataFrame(read_jsonl(cfg.path("sample") / "questions.jsonl"))
    rows = []
    for (ds, st), g in qs.groupby(["dataset", "stratum"], sort=False):
        rows.append((ds, st, len(g), g.doc_id.nunique(), g.doc_words.median(), g.references.map(len).mean()))
    name = {"qasper": "QASPER", "quality": "QuALITY"}
    body = "\n".join(
        f"{name[ds]} & {st.replace('_', '/')} & {n} & {nd} & {med:,.0f} & {nr:.1f} \\\\" for ds, st, n, nd, med, nr in rows
    )
    tot = f"\\midrule\nTotal & & {len(qs)} & {qs.doc_id.nunique()} & {qs.doc_words.median():,.0f} & {qs.references.map(len).mean():.1f} \\\\"
    write(out / "tables" / "sample.tex", rf"""% Gerado por `python -m llmeval analyze` (table_sample). Não editar à mão.
\begin{{table}}[t]
\centering
\caption{{Composition of the evaluation sample. Documents are counted once per stratum; length is the median document length in words.}}
\label{{tab:sample}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llrrrr}}
\toprule
Dataset & Stratum & Questions & Docs & Words & Refs/q \\
\midrule
{body}
{tot}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def table_models(cfg, out: Path, client=None) -> None:
    info = {}
    if client is not None:
        for m in [g["model"] for g in cfg["generators"]] + [j["model"] for j in cfg["jury"]["judges"]]:
            try:
                d = client._post("/api/show", {"model": m})
                mi = d.get("model_info", {})
                ctx = next((v for k, v in mi.items() if k.endswith("context_length")), None)
                info[m] = (d.get("details", {}).get("quantization_level", "--"), ctx)
            except Exception as exc:  # servidor indisponível: tabela sai sem essas colunas
                log.warning("api/show %s: %s", m, exc)

    def row(entry):
        q, ctx = info.get(entry["model"], ("--", None))
        ctxs = f"{ctx // 1024}k" if ctx else "--"
        params = entry.get("params_b")
        ps = f"{params:g}" if params else "--"
        return f"{entry['label']} & \\texttt{{{entry['model']}}} & {ps} & {q.replace('_', chr(92) + '_')} & {ctxs} \\\\"

    params = {"gpt-oss:120b": 116.8, "llama3.3:70b": 70.6, "qwen3:32b": 32.8}
    gens = "\n".join(row(g) for g in cfg["generators"])
    judges = "\n".join(row({**j, "params_b": params.get(j["model"])}) for j in cfg["jury"]["judges"])
    write(out / "tables" / "models.tex", rf"""% Gerado por `python -m llmeval analyze` (table_models). Não editar à mão.
\begin{{table}}[t]
\centering
\caption{{Models used as generators and as judges, as served by Ollama. Parameters in billions; quantisation and maximum context length as reported by the server.}}
\label{{tab:models}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{llrll}}
\toprule
Model & Ollama tag & Params & Quant. & Ctx \\
\midrule
\multicolumn{{5}}{{l}}{{\emph{{Generators}}}} \\
{gens}
\midrule
\multicolumn{{5}}{{l}}{{\emph{{Judges}}}} \\
{judges}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


# ===================================================================== RQ1
def rq1_instruments(cfg, df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Métricas e júri como classificadores do rótulo objetivo (QuALITY) e do júri (QASPER)."""
    rows = []
    ql = df[(df.dataset == "quality") & df.y_jury.notna() & df.y_objective.notna()]
    qa = df[(df.dataset == "qasper") & df.y_jury.notna() & (df.jury_source == "jury")]
    for target, sub, ycol in (("quality", ql, "y_objective"), ("qasper", qa, "y_jury")):
        if sub.empty:
            continue
        y = sub[ycol].astype(int).to_numpy()
        for m in METRICS:
            if m not in sub or sub[m].isna().all():
                continue
            s = sub[m].to_numpy()
            auc, thr, j = youden(y, s)
            pred = (s >= thr).astype(int)
            rows.append({"target": target, "instrument": METRIC_LABEL[m], "auroc": auc, "threshold": thr,
                         "acc": (pred == y).mean(), "kappa": cohen_kappa_score(y, pred), "n": len(y)})
        if target == "quality":
            pred = sub.y_jury.astype(int).to_numpy()
            from sklearn.metrics import roc_auc_score
            rows.append({"target": target, "instrument": "LLM jury (majority)",
                         "auroc": roc_auc_score(y, sub.jury_score), "threshold": np.nan,
                         "acc": (pred == y).mean(), "kappa": cohen_kappa_score(y, pred), "n": len(y)})
            for judge in cfg["jury"]["judges"]:
                v = sub.votes.map(lambda d: (d or {}).get(judge["model"]))
                ok = v.notna()
                pj = (v[ok] == "correct").astype(int).to_numpy()
                rows.append({"target": target, "instrument": f"Judge: {judge['label']}", "auroc": np.nan,
                             "threshold": np.nan, "acc": (pj == y[ok.to_numpy()]).mean(),
                             "kappa": cohen_kappa_score(y[ok.to_numpy()], pj), "n": int(ok.sum())})
    res = pd.DataFrame(rows)
    res.to_csv(cfg.path("results") / "analysis" / "rq1_instruments.csv", index=False)

    def block(target):
        r = res[res.target == target] if not res.empty else res
        lines = []
        for _, x in r.iterrows():
            thr = "--" if pd.isna(x.threshold) else f"{x.threshold:.2f}"
            auc = "--" if pd.isna(x.auroc) else f"{x.auroc:.3f}"
            lines.append(f"{x.instrument} & {auc} & {thr} & {pct(x.acc)} & {x.kappa:.3f} \\\\")
        return "\n".join(lines) if lines else r"\pending & \pending & \pending & \pending & \pending \\"

    nq = int(res[res.target == "quality"].n.max()) if not res.empty and (res.target == "quality").any() else 0
    nj = int(res[res.target == "qasper"].n.max()) if not res.empty and (res.target == "qasper").any() else 0
    write(out / "tables" / "rq1_instruments.tex", rf"""% Gerado por `python -m llmeval analyze` (rq1_instruments). Não editar à mão.
\begin{{table}}[t]
\centering
\caption{{Grading instruments as classifiers of correctness. Top: against the objective QuALITY label ($n={nq}$ answers). Bottom: against the jury verdict on QASPER answers graded by the jury ($n={nj}$). $\tau^\star$ is the Youden-optimal threshold; accuracy and Cohen's $\kappa$ are computed at $\tau^\star$, an optimistic upper bound for a thresholded metric.}}
\label{{tab:rq1}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Instrument & AUROC & $\tau^\star$ & Acc.\,(\%) & $\kappa$ \\
\midrule
\multicolumn{{5}}{{l}}{{\emph{{Reference: objective label (QuALITY)}}}} \\
{block("quality")}
\midrule
\multicolumn{{5}}{{l}}{{\emph{{Reference: jury verdict (QASPER)}}}} \\
{block("qasper")}
\bottomrule
\end{{tabular}}
\end{{table}}
""")
    return res


def rq1_reliability(cfg, df: pd.DataFrame, out: Path) -> dict:
    """Confiabilidade entre juízes e teste de autopreferência."""
    judges = cfg["jury"]["judges"]
    jj = df[(df.jury_source == "jury") & df.votes.notna()].copy() if "votes" in df else df.iloc[0:0]
    stats = {}
    for ds in ("qasper", "quality"):
        sub = jj[jj.dataset == ds]
        mat = np.array([[VERDICT_ORD.get((v or {}).get(j["model"]), np.nan) for j in judges] for v in sub.votes]) \
            if len(sub) else np.empty((0, len(judges)))
        full = mat[~np.isnan(mat).any(1)] if len(mat) else mat
        if len(full) < 2:
            continue
        stats[ds] = {
            "n": len(full),
            "fleiss": fleiss_kappa((full == 2).astype(int)),
            "alpha": krippendorff_alpha(full, "ordinal"),
            "pairwise": {(a["label"], b["label"]): cohen_kappa_score(full[:, i] == 2, full[:, k] == 2)
                         for i, a in enumerate(judges) for k, b in enumerate(judges) if i < k},
        }

    # autopreferência: taxa de falsos positivos (respostas objetivamente erradas aceitas) no QuALITY
    fam_rows = []
    ql = jj[(jj.dataset == "quality") & (jj.y_objective == False)]  # noqa: E712
    for j in judges:
        acc = ql.votes.map(lambda d: (d or {}).get(j["model"]) == "correct")
        same = ql.family == j["family"]
        fam_rows.append({
            "judge": j["label"], "family": j["family"],
            "fpr_same": acc[same].mean() if same.any() else np.nan, "n_same": int(same.sum()),
            "fpr_other": acc[~same].mean() if (~same).any() else np.nan, "n_other": int((~same).sum()),
        })
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(cfg.path("results") / "analysis" / "rq1_self_preference.csv", index=False)

    def cell(ds, key):
        return f"{stats[ds][key]:.3f}" if ds in stats else r"\pending"

    pair_lines = []
    for i, a in enumerate(judges):
        for k, b in enumerate(judges):
            if i < k:
                vals = [f"{stats[ds]['pairwise'][(a['label'], b['label'])]:.3f}" if ds in stats else r"\pending"
                        for ds in ("quality", "qasper")]
                pair_lines.append(f"$\\kappa$ {a['label']} vs. {b['label']} & {vals[0]} & {vals[1]} \\\\")
    fpr_lines = [
        f"FPR {r.judge}: own / other family & "
        + (f"{pct(r.fpr_same)} / {pct(r.fpr_other)}" if not pd.isna(r.fpr_other) else r"\pending")
        + r" & -- \\"
        for r in fam.itertuples()
    ]
    write(out / "tables" / "rq1_reliability.tex", rf"""% Gerado por `python -m llmeval analyze` (rq1_reliability). Não editar à mão.
\begin{{table}}[t]
\centering
\caption{{Reliability of the jury. Fleiss' $\kappa$ and pairwise Cohen's $\kappa$ on binary verdicts (\textsc{{correct}} vs.\ not); Krippendorff's $\alpha$ on the ordinal three-level verdicts. FPR: share of objectively wrong QuALITY answers accepted by the judge, for generators of its own family vs.\ other families (\%).}}
\label{{tab:reliability}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrr}}
\toprule
 & QuALITY & QASPER \\
\midrule
Answers graded by the jury & {stats.get('quality', {}).get('n', r'\pending')} & {stats.get('qasper', {}).get('n', r'\pending')} \\
Fleiss' $\kappa$ (3 judges) & {cell('quality', 'fleiss')} & {cell('qasper', 'fleiss')} \\
Krippendorff's $\alpha$ (ordinal) & {cell('quality', 'alpha')} & {cell('qasper', 'alpha')} \\
{chr(10).join(pair_lines)}
\midrule
{chr(10).join(fpr_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")
    return stats


def fig_roc(cfg, df: pd.DataFrame, out: Path) -> None:
    ql = df[(df.dataset == "quality") & df.y_jury.notna() & df.y_objective.notna()]
    if ql.empty or "token_f1" not in ql:
        return
    y = ql.y_objective.astype(int)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.82))
    for color, m in zip(SERIES, METRICS):
        fpr, tpr, _ = roc_curve(y, ql[m])
        auc = youden(y, ql[m])[0]
        ax.plot(fpr, tpr, color=color, lw=1.6, label=f"{METRIC_LABEL[m]} ({auc:.2f})")
    jy = ql.y_jury.astype(int)
    tpr = ((jy == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    fpr = ((jy == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
    ax.plot([fpr], [tpr], marker="*", ms=11, color=INK, ls="none", label="LLM jury (majority)")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(frameon=False, loc="lower right", title="Instrument (AUROC)", title_fontsize=7)
    fig.savefig(out / "figures" / "rq1_roc.pdf")
    plt.close(fig)



# ===================================================================== RQ2
def rq2_plausible(cfg, df: pd.DataFrame, out: Path, instruments: pd.DataFrame | None, metric: str = "bertscore"):
    """Parecer certo × estar certo (Seção "Looking right versus being right").

    Estima, por modelo e dataset, os termos de s̄_m = μ⁰_m + Acc_m·Δ_m, a FAR e o viés de um
    avaliador por similaridade no limiar τ* da RQ1, τ de Kendall entre rankings e o GEE
    S ~ Y·log2(P). Abstenções ficam de fora (a similaridade delas é definida por convenção).
    """
    from scipy.stats import kendalltau
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    sub = df[df.y.notna() & df[metric].notna() & ~df.abstained.astype(bool)].copy() if metric in df else df.iloc[0:0]
    gens = [g for g in cfg["generators"]]
    rows, summary = [], {}
    for ds in ("qasper", "quality"):
        d = sub[sub.dataset == ds]
        if d.empty:
            continue
        thr = np.nan
        if instruments is not None and not instruments.empty:
            r = instruments[(instruments.target == ds) & (instruments.instrument == METRIC_LABEL[metric])]
            thr = float(r.threshold.iloc[0]) if len(r) else np.nan
        for g in gens:
            x = d[d.model == g["model"]]
            if x.empty:
                continue
            y = x.y.astype(int)
            mu0, mu1 = x[metric][y == 0].mean(), x[metric][y == 1].mean()
            acc = y.mean()
            far = (x[metric][y == 0] >= thr).mean() if not np.isnan(thr) else np.nan
            tpr = (x[metric][y == 1] >= thr).mean() if not np.isnan(thr) else np.nan
            auc = youden(y, x[metric])[0] if y.nunique() == 2 else np.nan
            rows.append({"dataset": ds, "model": g["model"], "params_b": g["params_b"], "n": len(x), "acc": acc,
                         "s_mean": x[metric].mean(), "mu0": mu0, "mu1": mu1, "delta": mu1 - mu0, "auroc": auc,
                         "far": far, "bias": (1 - acc) * far - acc * (1 - tpr) if not np.isnan(thr) else np.nan,
                         "threshold": thr})
        m = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        cells = d.groupby(["model", "condition"]).agg(s=(metric, "mean"), acc=("y", "mean"))
        rev = sum(1 for i, a in m.iterrows() for j, b in m.iterrows() if a.s_mean > b.s_mean and a.acc < b.acc)
        info = {"tau_models": kendalltau(m.s_mean, m.acc)[0] if len(m) > 2 else np.nan,
                "tau_cells": kendalltau(cells.s, cells.acc)[0] if len(cells) > 2 else np.nan,
                "reversals": rev, "pairs": len(m) * (len(m) - 1) // 2}
        try:
            d = d.assign(log2p=np.log2(d.params_b), Y=d.y.astype(int), S=d[metric]).sort_values("doc_id")
            fit = smf.gee("S ~ Y * log2p + C(condition)", "doc_id", d, family=sm.families.Gaussian(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            ci = fit.conf_int()
            for k, term in (("b2", "log2p"), ("b3", "Y:log2p")):
                info[k] = (fit.params[term], ci.loc[term, 0], ci.loc[term, 1], fit.pvalues[term])
        except Exception as exc:  # poucos dados / modelos
            log.warning("GEE da RQ2 falhou (%s): %s", ds, exc)
        summary[ds] = info
    res = pd.DataFrame(rows)
    if not res.empty:
        res.to_csv(cfg.path("results") / "analysis" / "plausible.csv", index=False)
        pd.DataFrame(summary).T.to_csv(cfg.path("results") / "analysis" / "plausible_summary.csv")

    def cells_for(ds, g):
        r = res[(res.dataset == ds) & (res.model == g["model"])] if not res.empty else res
        if r.empty:
            return [r"\pending"] * 5
        r = r.iloc[0]
        return [pct(r.acc), f"{r.s_mean:.3f}", f"{r.mu0:.3f}", f"{r.delta:.3f}", pct(r.far)]

    lines = [f"{g['label']} & " + " & ".join(cells_for("qasper", g) + cells_for("quality", g)) + r" \\" for g in gens]

    def fmt_sum(ds, key):
        v = summary.get(ds, {}).get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return r"\pending"
        if isinstance(v, tuple):
            return f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    foot = "\n".join([
        rf"\multicolumn{{1}}{{l}}{{Kendall $\tau_K$ (models / cells)}} & \multicolumn{{5}}{{c}}{{{fmt_sum('qasper', 'tau_models')} / {fmt_sum('qasper', 'tau_cells')}}} & \multicolumn{{5}}{{c}}{{{fmt_sum('quality', 'tau_models')} / {fmt_sum('quality', 'tau_cells')}}} \\",
        rf"\multicolumn{{1}}{{l}}{{Reversed pairs (\cref{{eq:reversal}})}} & \multicolumn{{5}}{{c}}{{{fmt_sum('qasper', 'reversals')} of {fmt_sum('qasper', 'pairs')}}} & \multicolumn{{5}}{{c}}{{{fmt_sum('quality', 'reversals')} of {fmt_sum('quality', 'pairs')}}} \\",
        rf"\multicolumn{{1}}{{l}}{{$\beta_2$ (\cref{{eq:plausible}})}} & \multicolumn{{5}}{{c}}{{{fmt_sum('qasper', 'b2')}}} & \multicolumn{{5}}{{c}}{{{fmt_sum('quality', 'b2')}}} \\",
        rf"\multicolumn{{1}}{{l}}{{$\beta_3$ (\cref{{eq:plausible}})}} & \multicolumn{{5}}{{c}}{{{fmt_sum('qasper', 'b3')}}} & \multicolumn{{5}}{{c}}{{{fmt_sum('quality', 'b3')}}} \\",
    ])
    hdr = r"Acc. & $\bar s$ & $\hat\mu^{(0)}$ & $\hat\Delta$ & FAR"
    write(out / "tables" / "plausible.tex", rf"""% Gerado por `python -m llmeval analyze` (rq2_plausible). Não editar à mão.
\begin{{table*}}[t]
\centering
\caption{{Looking right versus being right, with BERTScore as the similarity $S$ (abstentions excluded). Per generator: accuracy (\%), mean similarity $\bar s$, mean similarity of incorrect answers $\hat\mu^{{(0)}}$, separation $\hat\Delta=\hat\mu^{{(1)}}-\hat\mu^{{(0)}}$ (\cref{{eq:decomp}}) and false-acceptance rate at the RQ1 threshold (\%). Bottom: rank agreement between $\bar s$ and accuracy, number of model pairs that satisfy the reversal condition, and the size coefficients of \cref{{eq:plausible}} (95\% CI). Correctness is the jury verdict on QASPER and the answer key on QuALITY.}}
\label{{tab:plausible}}
\footnotesize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{l rrrrr rrrrr}}
\toprule
 & \multicolumn{{5}}{{c}}{{QASPER (jury)}} & \multicolumn{{5}}{{c}}{{QuALITY (answer key)}} \\
\cmidrule(lr){{2-6}} \cmidrule(lr){{7-11}}
Model & {hdr} & {hdr} \\
\midrule
{chr(10).join(lines)}
\midrule
{foot}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")

    # figura: (esq.) s̄ × acurácia por modelo; (dir.) S de respostas certas × erradas no QuALITY
    if res.empty:
        return res
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    for color, marker, (ds, name) in zip(SERIES, ("o", "s"), (("qasper", "QASPER (jury)"), ("quality", "QuALITY (key)"))):
        r = res[res.dataset == ds]
        ax.plot(r.s_mean, r.acc * 100, ls="none", marker=marker, color=color, ms=5, label=name)
        for _, row in r.iterrows():
            lab = next(g["label"] for g in gens if g["model"] == row.model).split("-")[-1]
            ax.annotate(lab, (row.s_mean, row.acc * 100), xytext=(3, 2), textcoords="offset points", fontsize=5.5, color=INK2)
    ax.margins(x=0.12)  # espaço para os rótulos dos pontos na borda
    ax.set(xlabel=r"Mean similarity $\bar s$ (BERTScore)", ylabel="Accuracy (%)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Similarity vs. accuracy per model", loc="left")
    ax = axes[1]
    d = sub[sub.dataset == "quality"]
    models = [g for g in gens if g["model"] in set(d.model)]
    pos = np.arange(len(models))
    for k, (yv, color, name) in enumerate(((1, SERIES[2], "correct"), (0, SERIES[7], "incorrect"))):
        data = [d[(d.model == g["model"]) & (d.y == yv)][metric].values for g in models]
        bp = ax.boxplot(data, positions=pos + (-0.18 if yv else 0.18), widths=0.32, patch_artist=True,
                        showfliers=False, medianprops={"color": INK, "lw": 1})
        for b in bp["boxes"]:
            b.set(facecolor=color, alpha=0.55, edgecolor=INK2, lw=0.6)
        ax.plot([], [], color=color, lw=6, alpha=0.55, label=name)
    thr = res[res.dataset == "quality"].threshold.dropna()
    if len(thr):
        ax.axhline(thr.iloc[0], color=INK2, ls="--", lw=0.9)
        ax.text(-0.55, thr.iloc[0], r"$\tau^\star$", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(pos, [g["label"] for g in models], rotation=20, ha="right")
    ax.set_ylabel("Similarity $S$ (BERTScore)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="lower left")
    ax.set_title("QuALITY: similarity of correct vs. incorrect answers", loc="left")
    fig.savefig(out / "figures" / "plausible.pdf")
    plt.close(fig)
    return res

# ===================================================================== RQ3/RQ4 (full doc. e RAG)
def accuracy_table(cfg, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, m, c), g in df[df.y.notna()].groupby(["dataset", "model", "condition"]):
        k, n = int(g.y.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"dataset": ds, "model": m, "condition": c, "acc": k / n, "lo": lo, "hi": hi, "n": n,
                     "context_words": g.context_words.mean(), "prompt_tokens": g.prompt_tokens.mean(),
                     "latency_s": g.latency_s.mean()})
    acc = pd.DataFrame(rows)
    acc.to_csv(cfg.path("results") / "analysis" / "accuracy.csv", index=False)
    return acc


def table_accuracy(cfg, acc: pd.DataFrame, out: Path) -> None:
    models = [g for g in cfg["generators"]]
    conds = [c["name"] for c in cfg["conditions"]]
    header = " & ".join(COND_TEX[c].replace("Hybrid", "Hyb.") for c in conds)
    # linha de grupos sobre as colunas: recuperação (RAG) e busca guiada pelo modelo
    family = {"rag": "Retrieval", "search": "Model-guided search", "full_search": "Model-guided search"}
    groups, cmid = [], []
    col = 2
    for fam, run in itertools.groupby(cfg["conditions"], key=lambda c: family.get(c["context"])):
        n = len(list(run))
        groups.append(rf"\multicolumn{{{n}}}{{c}}{{{fam}}}" if fam else " & ".join([""] * n))
        if fam:
            cmid.append(rf"\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    group_row = " & " + " & ".join(groups) + r" \\" + chr(10) + " ".join(cmid)
    blocks = []
    for ds, name in (("qasper", "QASPER (jury verdict)"), ("quality", "QuALITY (answer key)")):
        lines = [rf"\multicolumn{{{len(conds) + 1}}}{{l}}{{\emph{{{name}}}}} \\"]
        sub = acc[acc.dataset == ds] if not acc.empty else acc
        for g in models:
            vals = {c: sub[(sub.model == g["model"]) & (sub.condition == c)].acc for c in conds}
            vals = {c: (v.iloc[0] if len(v) else np.nan) for c, v in vals.items()}
            best = np.nanmax(list(vals.values())) if not all(pd.isna(v) for v in vals.values()) else None
            cells = [(rf"\textbf{{{pct(v)}}}" if best is not None and v == best else pct(v)) for v in vals.values()]
            cells = [r"\pending" if c == "--" else c for c in cells]
            lines.append(f"{g['label']} & " + " & ".join(cells) + r" \\")
        means = [sub[sub.condition == c].acc.mean() if len(sub) else np.nan for c in conds]
        lines.append(r"\midrule Mean & " + " & ".join(r"\pending" if pd.isna(v) else pct(v) for v in means) + r" \\")
        blocks.append("\n".join(lines))
    write(out / "tables" / "accuracy.tex", rf"""% Gerado por `python -m llmeval analyze` (table_accuracy). Não editar à mão.
\begin{{table*}}[t]
\centering
\caption{{Accuracy (\%) of each generator under each context condition. QASPER answers are graded by the LLM jury (strict majority of \textsc{{correct}} votes, abstentions by rule); QuALITY answers by the answer key (chance level: 25\%). Best condition per model in bold. 95\% Wilson intervals are in the supplementary CSV; on $n=200$ questions per dataset their half-width is at most 6.9 points.}}
\label{{tab:accuracy}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{l{"r" * len(conds)}}}
\toprule
{group_row}
Model & {header} \\
\midrule
{(chr(10) + r"\midrule" + chr(10)).join(blocks)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def fig_accuracy(cfg, acc: pd.DataFrame, out: Path) -> None:
    """Dot plot com IC de Wilson: modelos no eixo y, uma faceta por dataset, condições como marcas."""
    if acc.empty:
        return
    gens = [g for g in cfg["generators"] if g["model"] in set(acc.model)]
    conds = [c["name"] for c in cfg["conditions"] if c["name"] in set(acc.condition)]
    style = COND_STYLE
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 0.5 * len(gens) + 1.0), sharey=True)
    offs = np.linspace(-0.36, 0.36, len(conds))
    for ax, (ds, title) in zip(axes, (("qasper", "QASPER (jury)"), ("quality", "QuALITY (answer key)"))):
        sub = acc[acc.dataset == ds]
        for off, c in zip(offs, conds):
            color, marker = style[c]
            for yi, g in enumerate(gens):
                r = sub[(sub.model == g["model"]) & (sub.condition == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([r.lo * 100, r.hi * 100], [yi + off] * 2, color=color, lw=0.9, alpha=0.6)
                ax.plot(r.acc * 100, yi + off, marker=marker, color=color, ls="none", ms=7 if marker == "*" else 4.5,
                        mec="white", mew=0.5, label=COND_LABEL[c] if yi == 0 else None)
        if ds == "quality":
            ax.axvline(25, color=INK2, lw=0.8, ls=":")
            ax.text(24, -0.45, "chance", color=INK2, fontsize=6.5, va="top", ha="right")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Accuracy (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(gens)), [g["label"] for g in gens])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=min(len(conds), 6), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(out / "figures" / "accuracy.pdf")
    plt.close(fig)


def fig_cost(cfg, acc: pd.DataFrame, out: Path) -> None:
    """Acurácia × tamanho médio do contexto (palavras, escala log): varredura do híbrido
    (k = 2, 5, 10) até o documento completo, uma linha por modelo."""
    if acc.empty:
        return
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    gens = [g for g in cfg["generators"] if g["model"] in set(acc.model)]
    order = ["hybrid_k2", "hybrid_k5", "hybrid_k10", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.3))
    for ax, (ds, title) in zip(axes, (("qasper", "QASPER (jury)"), ("quality", "QuALITY (answer key)"))):
        sub = acc[(acc.dataset == ds) & acc.condition.isin(order)]
        for color, g in zip(SERIES, gens):
            r = sub[sub.model == g["model"]].set_index("condition").reindex(order).dropna(subset=["acc"])
            if r.empty:
                continue
            ax.plot(r.context_words, r.acc * 100, color=color, marker="o", ms=3.5, lw=1.4, label=g["label"])
            if "full" in r.index:
                f = r.loc["full"]
                ax.plot(f.context_words, f.acc * 100, marker="s", color=color, ms=5.5, mec="white", mew=0.6)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Mean context size (words, log scale)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(out / "figures" / "cost_accuracy.pdf")
    plt.close(fig)


def rq3_paired(cfg, df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Δ de acurácia (condição − documento completo) por modelo, McNemar + Holm, IC por bootstrap."""
    models, conds = _order(cfg, df)
    rows = []
    for ds in ("qasper", "quality"):
        sub = df[(df.dataset == ds) & df.y.notna()]
        base = sub[sub.condition == "full"].set_index(["qid", "model"])
        for c in conds:
            if c == "full":
                continue
            other = sub[sub.condition == c].set_index(["qid", "model"])
            for m in models:
                a = base.xs(m, level="model", drop_level=False) if m in base.index.get_level_values(1) else None
                if a is None:
                    continue
                pair = a[["y", "doc_id"]].join(other[["y"]], rsuffix="_c", how="inner")
                if pair.empty:
                    continue
                d, lo, hi = cluster_bootstrap_diff(pair.y.values, pair.y_c.values, pair.doc_id.values)
                rows.append({"dataset": ds, "condition": c, "model": m, "delta": d, "lo": lo, "hi": hi,
                             "p": mcnemar_exact(pair.y.values, pair.y_c.values), "n": len(pair)})
    res = pd.DataFrame(rows)
    if res.empty:
        _placeholder_paired(cfg, out)
        return res
    res["p_holm"] = holm(res.p.values)
    res.to_csv(cfg.path("results") / "analysis" / "paired.csv", index=False)

    gens = [g for g in cfg["generators"] if g["model"] in models]
    header = " & ".join(g["label"].replace("-", "\\mbox{-}") for g in gens)
    blocks = []
    for ds, name in (("qasper", "QASPER"), ("quality", "QuALITY")):
        lines = [rf"\multicolumn{{{len(gens) + 1}}}{{l}}{{\emph{{{name}}}}} \\"]
        for c in conds:
            if c == "full":
                continue
            cells = []
            for g in gens:
                r = res[(res.dataset == ds) & (res.condition == c) & (res.model == g["model"])]
                cells.append("--" if r.empty else f"{100 * r.delta.iloc[0]:+.1f}{pval_marker(r.p_holm.iloc[0])}")
            lines.append(f"{COND_TEX[c]} & " + " & ".join(cells) + r" \\")
        blocks.append("\n".join(lines))
    write(out / "tables" / "paired.tex", rf"""% Gerado por `python -m llmeval analyze` (paired). Não editar à mão.
\begin{{table*}}[t]
\centering
\caption{{Paired difference in accuracy (percentage points) of each condition relative to the full document, per model. Positive values mean the condition is better than reading the whole document. Significance by exact McNemar test with Holm correction over all {len(res)} comparisons: $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$. Cluster-bootstrap 95\% intervals are in the supplementary CSV.}}
\label{{tab:paired}}
\footnotesize
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{l{"r" * len(gens)}}}
\toprule
Condition & {header} \\
\midrule
{(chr(10) + r"\midrule" + chr(10)).join(blocks)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
    return res


def _placeholder_paired(cfg, out):
    gens = cfg["generators"]
    conds = [c["name"] for c in cfg["conditions"] if c["name"] != "full"]
    header = " & ".join(g["label"].replace("-", "\\mbox{-}") for g in gens)
    blocks = []
    for name in ("QASPER", "QuALITY"):
        lines = [rf"\multicolumn{{{len(gens) + 1}}}{{l}}{{\emph{{{name}}}}} \\"]
        lines += [f"{COND_TEX[c]} & " + " & ".join([r"\pending"] * len(gens)) + r" \\" for c in conds]
        blocks.append("\n".join(lines))
    write(out / "tables" / "paired.tex", rf"""% Placeholder gerado por `python -m llmeval analyze` (paired) antes dos resultados.
\begin{{table*}}[t]
\centering
\caption{{Paired difference in accuracy (percentage points) of each condition relative to the full document, per model. Positive values mean the condition is better than reading the whole document. Exact McNemar test with Holm correction: $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.}}
\label{{tab:paired}}
\footnotesize
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{l{"r" * len(gens)}}}
\toprule
Condition & {header} \\
\midrule
{(chr(10) + r"\midrule" + chr(10)).join(blocks)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def table_gee(cfg, df: pd.DataFrame, out: Path) -> None:
    sub = df[df.y.notna() & (df.condition != "closed_book")].copy()
    rows_tex = r"\pending & \pending & \pending \\"
    if len(sub) > 50 and sub.model.nunique() > 1 and sub.condition.nunique() > 1:
        sub["log2_words"] = np.log2(sub.doc_words)
        sub["y"] = sub.y.astype(int)
        ref_model = cfg["generators"][0]["model"]
        formula = (f"y ~ C(condition, Treatment('full')) + C(model, Treatment('{ref_model}')) "
                   f"+ C(dataset) + log2_words")
        try:
            res = gee_logit(sub.sort_values("doc_id"), formula, "doc_id")
            res.to_csv(cfg.path("results") / "analysis" / "gee.csv")
            labels = {g["model"]: g["label"] for g in cfg["generators"]}
            lines = []
            for term, r in res.iterrows():
                if term == "Intercept":
                    continue
                name = term
                if "condition" in term:
                    name = "Cond.: " + COND_TEX.get(term.split("[T.")[1].rstrip("]"), term)
                elif "model" in term:
                    name = "Model: " + labels.get(term.split("[T.")[1].rstrip("]"), term)
                elif "dataset" in term:
                    name = "Dataset: QuALITY"
                elif term == "log2_words":
                    name = r"Doc.\ length ($\times 2$)"
                lines.append(f"{name} & {r.OR:.2f} [{r.lo:.2f}, {r.hi:.2f}] & {r.p:.3g} \\\\")
            rows_tex = "\n".join(lines)
        except Exception as exc:  # pouco dado ou separação perfeita
            log.warning("GEE falhou: %s", exc)
    ref_label = cfg["generators"][0]["label"]
    write(out / "tables" / "gee.tex", rf"""% Gerado por `python -m llmeval analyze` (table_gee). Não editar à mão.
\begin{{table}}[t]
\centering
\caption{{Odds ratios (95\% CI) from the GEE logistic model of \cref{{eq:gee}}, clustered by document. Reference levels: full document, {ref_label}, QASPER. Closed-book answers are excluded.}}
\label{{tab:gee}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrr}}
\toprule
Term & OR [95\% CI] & $p$ \\
\midrule
{rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def fig_recall(cfg, out: Path, df: pd.DataFrame) -> None:
    """Esq.: Recall@k de evidência por recuperador. Dir.: acurácia condicionada à evidência recuperada."""
    rt = pd.DataFrame(read_jsonl(cfg.path("results") / "retrieval.jsonl"))
    if rt.empty:
        return
    rt = rt[rt.recall.notna()]
    rt.groupby(["method", "k"]).recall.agg(["mean", "count"]).to_csv(
        cfg.path("results") / "analysis" / "retrieval_recall.csv")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.2))
    ax = axes[0]
    names = {"bm25": "BM25", "dense": "Dense (BGE-M3)", "hybrid": "Hybrid (RRF)"}
    for color, m in zip(SERIES, ("bm25", "dense", "hybrid")):
        g = rt[rt.method == m].groupby("k").recall.agg(["mean", "sem"])
        ax.errorbar(g.index, g["mean"] * 100, yerr=1.96 * g["sem"] * 100, color=color, marker="o", ms=3.5,
                    lw=1.6, capsize=2, label=names[m])
    ax.set(xscale="log", xlabel="Retrieved chunks $k$ (log scale)", ylabel="Evidence recall@$k$ (%)", ylim=(0, 100))
    ax.set_xticks(sorted(rt.k.unique()), [str(k) for k in sorted(rt.k.unique())])
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("QASPER evidence recall", loc="left")

    ax = axes[1]
    sub = df[(df.dataset == "qasper") & (df.condition == "hybrid_k5") & df.evidence_recall.notna() & df.y.notna()]
    if not sub.empty:
        sub = sub.assign(bin=pd.cut(sub.evidence_recall, [-0.01, 0.0, 0.999, 1.0], labels=["none", "partial", "all"]))
        gens = [g for g in cfg["generators"] if g["model"] in set(sub.model)]
        width = 0.8 / max(len(gens), 1)
        for i, (color, g) in enumerate(zip(SERIES, gens)):
            r = sub[sub.model == g["model"]].groupby("bin", observed=False).y.mean() * 100
            ax.bar(np.arange(3) + (i - (len(gens) - 1) / 2) * width, r.values, width * 0.9, color=color, label=g["label"])
        ax.set_xticks(range(3), ["No evidence\nretrieved", "Part of the\nevidence", "All\nevidence"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncol=2, fontsize=6, loc="upper left")
        ax.set_title("Accuracy by retrieval outcome (Hybrid $k$=5)", loc="left")
    fig.savefig(out / "figures" / "retrieval.pdf")
    plt.close(fig)


def fig_length_position(cfg, df: pd.DataFrame, out: Path) -> pd.DataFrame | None:
    """RQ2: acurácia no documento completo por posição relativa da evidência e por tamanho do documento."""
    full = df[(df.condition == "full") & df.y.notna()]
    if full.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.1))
    qa = full[(full.dataset == "qasper") & full.evidence_pos.notna()].copy()
    fit = None
    if len(qa) > 30:
        qa["pos_bin"] = pd.cut(qa.evidence_pos, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        g = qa.groupby("pos_bin", observed=False).y.agg(["mean", "count", "sum"])
        ci = np.array([wilson(int(s), int(n)) for s, n in zip(g["sum"], g["count"])])
        x = np.arange(len(g))
        axes[0].errorbar(x, g["mean"] * 100, yerr=[(g["mean"] - ci[:, 0]) * 100, (ci[:, 1] - g["mean"]) * 100],
                         color=SERIES[0], marker="o", lw=1.6, capsize=2)
        axes[0].set_xticks(x, ["0–20", "20–40", "40–60", "60–80", "80–100"])
        axes[0].set(xlabel="Position of first evidence paragraph (% of document)", ylabel="Accuracy (%)", ylim=(0, 100))
        for xi, n in zip(x, g["count"]):
            axes[0].text(xi, 3, f"n={n}", ha="center", fontsize=6, color=INK2)
        axes[0].set_title("QASPER, full document, all models", loc="left")
        try:
            qa["y"] = qa.y.astype(int)
            qa["pos2"] = qa.evidence_pos**2
            fit = gee_logit(qa.sort_values("doc_id"), "y ~ evidence_pos + pos2 + C(model)", "doc_id")
            fit.to_csv(cfg.path("results") / "analysis" / "position_fit.csv")
        except Exception as exc:
            log.warning("ajuste de posição falhou: %s", exc)

    ax = axes[1]
    for color, (ds, name) in zip(SERIES, (("qasper", "QASPER"), ("quality", "QuALITY"))):
        sub = full[full.dataset == ds].copy()
        if sub.empty:
            continue
        sub["len_bin"] = pd.qcut(sub.doc_words, 3, labels=["short", "medium", "long"])
        g = sub.groupby("len_bin", observed=False).agg(acc=("y", "mean"), words=("doc_words", "median"))
        ax.plot(g.words, g.acc * 100, color=color, marker="o", lw=1.6, label=name)
    ax.set(xlabel="Median document length in tercile (words)", ylabel="Accuracy (%)", ylim=(0, 100))
    ax.legend(frameon=False)
    ax.set_title("Full document, by length tercile", loc="left")
    fig.savefig(out / "figures" / "length_position.pdf")
    plt.close(fig)
    return fit


# ===================================================================== RQ5
def table_abstention(cfg, df: pd.DataFrame, out: Path) -> pd.DataFrame:
    qs = {q["qid"]: q for q in read_jsonl(cfg.path("sample") / "questions.jsonl")}
    sub = df[df.dataset == "qasper"].copy()
    conds = ["closed_book", "full", "hybrid_k5"]
    rows = []
    if not sub.empty:
        sub["gold_unans"] = sub.qid.map(lambda q: "UNANSWERABLE" in qs[q]["references"])
        for (m, c), g in sub[sub.condition.isin(conds)].groupby(["model", "condition"]):
            tp = int((g.abstained & g.gold_unans).sum())
            fp = int((g.abstained & ~g.gold_unans).sum())
            fn = int((~g.abstained & g.gold_unans).sum())
            p = tp / (tp + fp) if tp + fp else np.nan
            r = tp / (tp + fn) if tp + fn else np.nan
            f1 = 2 * p * r / (p + r) if p and r and not np.isnan(p) and not np.isnan(r) else np.nan
            rows.append({"model": m, "condition": c, "precision": p, "recall": r, "f1": f1,
                         "abstain_rate": g.abstained.mean()})
    res = pd.DataFrame(rows)
    if not res.empty:
        res.to_csv(cfg.path("results") / "analysis" / "abstention.csv", index=False)
    lines = []
    for g in cfg["generators"]:
        cells = []
        for c in conds:
            r = res[(res.model == g["model"]) & (res.condition == c)] if not res.empty else res
            if r.empty:
                cells += [r"\pending"] * 3
            else:
                r = r.iloc[0]
                cells += [pct(r.precision, 0), pct(r.recall, 0), pct(r.abstain_rate, 0)]
        lines.append(f"{g['label']} & " + " & ".join(c.replace("--", "--") for c in cells) + r" \\")
    sub_hdr = " & ".join(["P & R & Abst."] * len(conds))
    top = " & ".join(rf"\multicolumn{{3}}{{c}}{{{COND_TEX[c]}}}" for c in conds)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(conds)))
    write(out / "tables" / "abstention.tex", rf"""% Gerado por `python -m llmeval analyze` (table_abstention). Não editar à mão.
\begin{{table}}[t]
\centering
\caption{{Abstention on QASPER (\%). P and R: precision and recall of abstentions, with ``unanswerable'' as the positive class; Abst.: share of all questions on which the model abstained. Roughly one in four sampled questions is unanswerable.}}
\label{{tab:abstention}}
\footnotesize
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{l{"rrr" * len(conds)}}}
\toprule
 & {top} \\
{cmid}
Model & {sub_hdr} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")
    return res


# ============================================================ exemplo do carro
CAR_QUESTION = "How many wheels does a car have?"
CAR_REFERENCE = "Four wheels"
CAR_ANSWERS = [
    ("4", True), ("four", True), ("Usually a car has 4 wheels", True), ("If not broken, four", True),
    ("Two in the front and two in the back", True),
    ("Cars usually have 6 or more wheels depending on the model", False),
    ("I think it depends on the season", False), ("It has legs instead of wheels", False),
    ("The engine is what makes it move", False), ("Probably two, because bicycles have two", False),
    ("There are two wheels in the front", False),
]


def car_example(cfg, client, out: Path) -> None:
    """Tabela ilustrativa da Seção de Metodologia: métricas e votos do júri no exemplo do carro."""
    from .metrics import LEXICAL, _bertscore
    from .prompts import JUDGE_SCHEMA, judge_messages

    from .embeddings import Embedder

    answers = [a for a, _ in CAR_ANSWERS]
    emb = Embedder(cfg, cfg.path("cache"))(answers + [CAR_REFERENCE])
    cos = emb[:-1] @ emb[-1]
    bert = _bertscore([(a, CAR_REFERENCE) for a in answers])
    judges = cfg["jury"]["judges"]
    votes = {}
    for j in judges:
        for a in answers:
            msgs = judge_messages(CAR_QUESTION, [CAR_REFERENCE], a)
            r = client.chat_json(j["model"], msgs, JUDGE_SCHEMA, think=j.get("think", False), num_predict=2048)
            if r.get("output") is None:  # como no júri: raciocínio pode esgotar o orçamento
                r = client.chat_json(j["model"], msgs, JUDGE_SCHEMA, think=j.get("think", False), num_predict=8192)
            votes[j["model"], a] = (r.get("output") or {}).get("verdict")
    sym = {"correct": r"\checkmark", "partially_correct": r"$\sim$", "incorrect": r"$\times$", None: "--"}
    missing = " A dash marks a vote not returned in a valid format, which the jury treats as not correct." \
        if None in votes.values() else ""
    lines = []
    for i, (a, ok) in enumerate(CAR_ANSWERS):
        vals = [LEXICAL[m](a, CAR_REFERENCE) for m in ("em", "token_f1", "rouge_l", "fuzzy")] + [cos[i], bert[i]]
        v = [votes[j["model"], a] for j in judges]
        final = sum(x == "correct" for x in v) > len(v) / 2
        yes, no = r"\checkmark", r"$\times$"
        lines.append(
            f"{a} & {yes if ok else no} & " + " & ".join(f"{x:.2f}" for x in vals)
            + " & " + " & ".join(sym[x] for x in v) + f" & {yes if final else no} " + r"\\"
        )
        if i == 4:
            lines.append(r"\midrule")
    jh = " & ".join(j["label"].split("-")[0] for j in judges)
    write(out / "tables" / "car_example.tex", rf"""% Gerado por `python -m llmeval analyze` (car_example). Não editar à mão.
\begin{{table*}}[t]
\centering
\caption{{The motivating example: answers to ``{CAR_QUESTION}'' compared with the reference ``{CAR_REFERENCE}''. The first five answers are correct, the last six are not. Metrics are computed as defined in \cref{{sec:measuring}}; the last four columns are the votes of the three judges (\checkmark{{}} correct, $\sim$ partial, $\times$ incorrect) and the jury verdict.{missing}}}
\label{{tab:car}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{p{{5.2cm}}c rrrrrr ccc c}}
\toprule
 & & \multicolumn{{4}}{{c}}{{Lexical}} & \multicolumn{{2}}{{c}}{{Semantic}} & \multicolumn{{3}}{{c}}{{Judges}} & \\
\cmidrule(lr){{3-6}} \cmidrule(lr){{7-8}} \cmidrule(lr){{9-11}}
Answer & Gold & EM & F1 & ROUGE-L & Fuzzy & Emb.\ cos. & BERTScore & {jh} & Jury \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


# ============================================================ placeholders
# Mesmo layout das figuras reais, sem dados: o artigo mostra o que falta calcular.
PLACEHOLDERS = {
    "accuracy": dict(shape=(1, 2), size=(PAGE_W, 3.2), titles=["QASPER (jury)", "QuALITY (answer key)"],
                     xlabel="Accuracy (%)", ylabel="", xlim=(0, 100), yticks="models"),
    "cost_accuracy": dict(shape=(1, 2), size=(PAGE_W, 2.3), titles=["QASPER (jury)", "QuALITY (answer key)"],
                          xlabel="Mean context size (words, log scale)", ylabel="Accuracy (%)", ylim=(0, 100)),
    "retrieval": dict(shape=(1, 2), size=(PAGE_W, 2.2),
                      titles=["QASPER evidence recall", "Accuracy by retrieval outcome (Hybrid $k$=5)"],
                      xlabel=["Retrieved chunks $k$", "Retrieval outcome"],
                      ylabel=["Evidence recall@$k$ (%)", "Accuracy (%)"], ylim=(0, 100)),
    "length_position": dict(shape=(1, 2), size=(PAGE_W, 2.1),
                            titles=["QASPER, full document, all models", "Full document, by length tercile"],
                            xlabel=["Position of first evidence paragraph (% of document)",
                                    "Median document length in tercile (words)"],
                            ylabel="Accuracy (%)", ylim=(0, 100)),
    "plausible": dict(shape=(1, 2), size=(PAGE_W, 2.5), titles=["Similarity vs. accuracy per model",
                      "QuALITY: similarity of correct vs. incorrect answers"],
                      xlabel=[r"Mean similarity $\bar s$ (BERTScore)", ""], ylabel=["Accuracy (%)", "Similarity $S$ (BERTScore)"]),
    "reasoning": dict(shape=(1, 2), size=(PAGE_W, 2.4), titles=["QASPER (jury)", "QuALITY (answer key)"],
                      xlabel="Accuracy with minus without reasoning (pp)", ylabel="", xlim=(-30, 30),
                      yticks="reasoners"),
    "rq1_roc": dict(shape=(1, 1), size=(COL_W, COL_W * 0.82), titles=[""], xlabel="False positive rate",
                    ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1)),
}


# ===================================================================== RQ6 (raciocínio)
def rq6_reasoning(cfg, full_df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """δ pareado (com − sem raciocínio) por modelo × condição × dataset; custo em tokens; teto esgotado."""
    on = full_df[full_df.think]
    rows = []
    if not on.empty:
        off = full_df[~full_df.think].set_index(["qid", "model", "condition"])
        for (ds, m, c), g in on.groupby(["dataset", "model", "condition"]):
            pair = g.set_index(["qid", "model", "condition"])[["y", "doc_id", "completion_tokens", "done_reason"]].join(
                off[["y", "completion_tokens"]], rsuffix="_off", how="inner").dropna(subset=["y", "y_off"])
            if pair.empty:
                continue
            d, lo, hi = cluster_bootstrap_diff(pair.y_off.values, pair.y.values, pair.doc_id.values)
            rows.append({"dataset": ds, "model": m, "condition": c, "delta": d, "lo": lo, "hi": hi,
                         "acc_off": pair.y_off.mean(), "acc_on": pair.y.mean(), "n": len(pair),
                         "p": mcnemar_exact(pair.y_off.values, pair.y.values),
                         "tokens_off": pair.completion_tokens_off.median(), "tokens_on": pair.completion_tokens.median(),
                         "exhausted": (pair.done_reason == "length").mean()})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_holm"] = holm(res.p.values)
        res["token_ratio"] = res.tokens_on / res.tokens_off
        res.to_csv(cfg.path("results") / "analysis" / "reasoning.csv", index=False)

    allowed = cfg["generation"].get("think_models")
    gens = [g for g in cfg["generators"] if (g["model"] in allowed if allowed else g["family"] in ("qwen", "gemma"))]
    if not res.empty:
        gens = [g for g in cfg["generators"] if g["model"] in set(res.model)]
    show = cfg["generation"].get("think_conditions") or ["closed_book", "full", "hybrid_k5", "search_hybrid_k5",
                                                         "full_search_k5"]
    lines = []
    for ds, name in (("qasper", "QASPER"), ("quality", "QuALITY")):
        lines.append(rf"\multicolumn{{{len(show) + 3}}}{{l}}{{\emph{{{name}}}}} \\")
        for g in gens:
            sub = res[(res.dataset == ds) & (res.model == g["model"])] if not res.empty else res
            cells = []
            for c in show:
                r = sub[sub.condition == c] if not sub.empty else sub
                if r.empty:
                    cells.append(r"\pending")
                else:
                    r = r.iloc[0]
                    cells.append(f"{100 * r.delta:+.1f}".replace("-", "$-$") + pval_marker(r.p_holm))
            ratio = sub.token_ratio.median() if not sub.empty else np.nan
            exh = sub.exhausted.mean() if not sub.empty else np.nan
            cells += [r"\pending" if pd.isna(ratio) else f"{ratio:.0f}$\\times$", r"\pending" if pd.isna(exh) else pct(exh)]
            lines.append(f"{g['label']} & " + " & ".join(cells) + r" \\")
    header = " & ".join(COND_TEX[c] for c in show)
    write(out / "tables" / "reasoning.tex", rf"""% Gerado por `python -m llmeval analyze` (rq6_reasoning). Não editar à mão.
\begin{{table*}}[t]
\centering
\caption{{Effect of reasoning: paired accuracy difference with minus without reasoning (percentage points) on the same questions, in the five conditions run with reasoning. Exact McNemar test with Holm correction over all model--condition--dataset pairs: $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$. Tokens: median ratio of generated tokens with and without reasoning, over conditions. Budget: share of answers lost because reasoning exhausted the budget of 4096 tokens (\%).}}
\label{{tab:reasoning}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l{"r" * len(show)}rr}}
\toprule
Model & {header} & Tokens & Budget \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
    if not res.empty:
        fig_reasoning(cfg, res, out)
    return res


def fig_reasoning(cfg, res: pd.DataFrame, out: Path) -> None:
    gens = [g for g in cfg["generators"] if g["model"] in set(res.model)]
    conds = [c["name"] for c in cfg["conditions"] if c["name"] in set(res.condition)]
    offs = np.linspace(-0.32, 0.32, len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 0.5 * len(gens) + 1.1), sharey=True)
    for ax, (ds, title) in zip(axes, (("qasper", "QASPER (jury)"), ("quality", "QuALITY (answer key)"))):
        sub = res[res.dataset == ds]
        for off, c in zip(offs, conds):
            for yi, g in enumerate(gens):
                r = sub[(sub.model == g["model"]) & (sub.condition == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                color, marker = COND_STYLE[c]
                ax.plot([100 * r.lo, 100 * r.hi], [yi + off] * 2, color=color, lw=0.9, alpha=0.6)
                ax.plot(100 * r.delta, yi + off, marker=marker, color=color, ls="none", ms=4.5, mec="white",
                        mew=0.5, label=COND_LABEL[c] if yi == 0 else None)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Accuracy with minus without reasoning (pp)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(gens)), [g["label"] for g in gens])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.24, wspace=0.06)
    fig.savefig(out / "figures" / "reasoning.pdf")
    plt.close(fig)


def placeholder_figures(cfg, out: Path, only_missing: bool = True) -> None:
    for name, spec in PLACEHOLDERS.items():
        path = out / "figures" / f"{name}.pdf"
        if only_missing and path.exists():
            continue
        fig, axes = plt.subplots(*spec["shape"], figsize=spec["size"], squeeze=False)
        for i, ax in enumerate(axes.flat):
            pick = lambda v: v[i] if isinstance(v, list) else v  # noqa: E731
            ax.set_title(pick(spec["titles"]), loc="left")
            ax.set_xlabel(pick(spec.get("xlabel", "")))
            ax.set_ylabel(pick(spec.get("ylabel", "")))
            if "xlim" in spec:
                ax.set_xlim(*spec["xlim"])
            if "ylim" in spec:
                ax.set_ylim(*spec["ylim"])
            if spec.get("yticks") == "models" and i == 0:
                ax.set_yticks(range(len(cfg["generators"])), [g["label"] for g in cfg["generators"]])
                ax.invert_yaxis()
            elif spec.get("yticks") == "reasoners" and i == 0:
                labels = [g["label"] for g in cfg["generators"] if g["family"] in ("qwen", "gemma")]
                ax.set_yticks(range(len(labels)), labels)
                ax.invert_yaxis()
            elif spec.get("yticks") in ("models", "reasoners"):
                ax.set_yticks([])
            ax.text(0.5, 0.5, "não calculado", transform=ax.transAxes, ha="center", va="center",
                    color="#e34948", fontsize=9, alpha=0.8)
        fig.savefig(path)
        plt.close(fig)
        log.info("placeholder %s", path)


# ===================================================================== main
def run_analysis(cfg) -> None:
    out = Path(cfg.path("paper"))
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (cfg.path("results") / "analysis").mkdir(exist_ok=True)
    from .ollama import OllamaClient

    client = OllamaClient(cfg, cfg.path("cache"))
    placeholder_figures(cfg, out)  # figuras reais abaixo sobrescrevem as que tiverem dados
    table_sample(cfg, out)
    table_models(cfg, out, client)
    car_example(cfg, client, out)

    if not (cfg.path("results") / "generations.jsonl").exists():
        log.warning("sem gerações ainda: apenas tabelas de desenho foram produzidas")
        acc = pd.DataFrame(columns=["dataset", "model", "condition", "acc"])
        table_accuracy(cfg, acc, out)
        _placeholder_paired(cfg, out)
        table_abstention(cfg, pd.DataFrame(columns=["dataset"]), out)
        table_gee(cfg, pd.DataFrame(columns=["y", "condition"]), out)
        rq1_instruments(cfg, pd.DataFrame(columns=["dataset", "y_jury", "y_objective", "jury_source"]), out)
        rq1_reliability(cfg, pd.DataFrame(columns=["dataset", "jury_source", "votes", "y_objective", "family"]), out)
        rq2_plausible(cfg, pd.DataFrame(columns=["dataset", "y"]), out, None)
        rq6_reasoning(cfg, pd.DataFrame(columns=["think"]), out)
        return

    full_df = load_frame(cfg)
    df = full_df[~full_df.think].copy()  # RQ1–RQ5: desenho principal, sem raciocínio
    acc = accuracy_table(cfg, df)
    table_accuracy(cfg, acc, out)
    fig_accuracy(cfg, acc, out)
    fig_cost(cfg, acc, out)
    rq3_paired(cfg, df, out)
    table_gee(cfg, df, out)
    fig_recall(cfg, out, df)
    fig_length_position(cfg, df, out)
    table_abstention(cfg, df, out)
    instruments = None
    if "y_jury" in df:
        instruments = rq1_instruments(cfg, df, out)
        rq1_reliability(cfg, df, out)
        fig_roc(cfg, df, out)
    rq2_plausible(cfg, df, out, instruments)
    rq6_reasoning(cfg, full_df, out)

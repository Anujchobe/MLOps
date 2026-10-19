"""Step 5: build a self-contained HTML dashboard (reliability diagram,
baseline/raw/calibrated comparison, version history) and a Markdown summary
that GitHub renders on the workflow run page."""
import base64, io, json, os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from src.config import WORK_DIR, REPORTS_DIR, REGISTRY, ECE_BINS
from src.data import load_splits

COLS = ["f1", "precision", "recall", "roc_auc", "brier", "log_loss", "ece"]


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def reliability_plot():
    X_te, y_te = load_splits()["test"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, f in [("baseline (LogReg)", "baseline.joblib"),
                    ("raw (HistGB)", "raw_model.joblib"),
                    ("calibrated (HistGB + Platt)", "calibrated_model.joblib")]:
        p = joblib.load(WORK_DIR / f).predict_proba(X_te)[:, 1]
        frac, mean = calibration_curve(y_te, p, n_bins=ECE_BINS, strategy="quantile")
        ax.plot(mean, frac, "o-", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(loc="lower right", fontsize=8)
    return fig_to_b64(fig)


def history_plot(reg):
    vs = reg["versions"]
    x = range(1, len(vs) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(x, [v["metrics"]["f1"] for v in vs], "o-", label="F1 (calibrated)")
    ax1.plot(x, [v["baseline_f1"] for v in vs], "s--", label="F1 (baseline)")
    ax1.set_xlabel("Model version #")
    ax1.set_ylabel("F1")
    ax2 = ax1.twinx()
    ax2.plot(x, [v["metrics"]["brier"] for v in vs], "^:", color="tab:red", label="Brier")
    ax2.set_ylabel("Brier (lower is better)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.set_title("Metric history across versions")
    return fig_to_b64(fig)


def main():
    ev = json.loads((WORK_DIR / "eval.json").read_text())
    reg = json.loads(REGISTRY.read_text())
    res = ev["results"]

    rows = "".join(
        f"<tr><td>{n}</td>" + "".join(f"<td>{res[n][c]:.4f}</td>" for c in COLS) + "</tr>"
        for n in ("baseline", "raw", "calibrated"))
    hist = "".join(
        f"<tr><td><code>{v['version']}</code></td><td>{v['trained_at']}</td>"
        f"<td>{v['metrics']['f1']:.4f}</td><td>{v['metrics']['brier']:.4f}</td>"
        f"<td>{v['metrics']['ece']:.4f}</td><td>{'pruned' if v.get('pruned') else 'stored'}</td></tr>"
        for v in reversed(reg["versions"]))
    gate = ev["gate"]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Lab 2 Model Dashboard: {ev['version']}</title>
<style>
:root{{--bg:#fff;--fg:#1d2330;--muted:#5b6475;--line:#e3e6ec;--ok:#1a7f37;--bad:#cf222e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f1218;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2a303b}}}}
body{{font-family:-apple-system,system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:1000px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem;font-size:.9rem}}
th,td{{border-bottom:1px solid var(--line);padding:.4rem .6rem;text-align:left}}
.muted{{color:var(--muted)}} .ok{{color:var(--ok);font-weight:600}} .bad{{color:var(--bad);font-weight:600}}
img{{max-width:100%;background:#fff;border-radius:6px}} .wrap{{overflow-x:auto}}
</style></head><body>
<h1>Model dashboard</h1>
<p class="muted">Version <code>{ev['version']}</code> · trained {ev['trained_at']} · commit <code>{ev['git_sha'][:7]}</code>
· data fingerprint <code>{ev['data_fingerprint']}</code> · train/calib/test = {ev['n_train']}/{ev['n_calib']}/{ev['n_test']}</p>
<p>Quality gate: <span class="{'ok' if gate['passed'] else 'bad'}">{'PASSED' if gate['passed'] else 'FAILED'}</span>
<span class="muted">(F1 ≥ {gate['thresholds']['f1_min']}, ROC-AUC ≥ {gate['thresholds']['roc_auc_min']})</span></p>
<h2>Test-set comparison</h2>
<div class="wrap"><table><tr><th>model</th>{''.join(f'<th>{c}</th>' for c in COLS)}</tr>{rows}</table></div>
<img src="data:image/png;base64,{reliability_plot()}" alt="reliability diagram">
<h2>Version history</h2>
<img src="data:image/png;base64,{history_plot(reg)}" alt="metric history">
<div class="wrap"><table><tr><th>version</th><th>trained</th><th>F1</th><th>Brier</th><th>ECE</th><th>binary</th></tr>{hist}</table></div>
</body></html>"""
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dashboard.html").write_text(html)

    md = [f"## Model `{ev['version']}`",
          f"Gate: **{'PASSED' if gate['passed'] else 'FAILED'}** · versions in registry: {len(reg['versions'])}", "",
          "| model | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    md += [f"| {n} | " + " | ".join(f"{res[n][c]:.4f}" for c in COLS) + " |"
           for n in ("baseline", "raw", "calibrated")]
    md_text = "\n".join(md) + "\n"
    (REPORTS_DIR / "latest_summary.md").write_text(md_text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(md_text)
    print("Dashboard written to reports/dashboard.html")


if __name__ == "__main__":
    main()

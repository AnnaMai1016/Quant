"""AnalyseAgent — daily dashboard.

    python run_daily.py --history-dir ../../Quant-history            # full run
    python run_daily.py --tickers NVDA,AAPL --models claude --no-email
    python run_daily.py --dry-run                                     # data + fact sheets only, no LLM

Writes into <history-dir> (the private Quant-history repo checkout):
    reports/YYYY/MM/YYYY-MM-DD_daily.en.md / .zh.md / _factsheets.json
    history/verdicts_YYYY.csv  (one row per ticker x model, append-only)
    history/raw_YYYY.jsonl     (full model outputs, errors, token usage)
Read-only research: this program never places orders.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_agent import config, factsheet, history, notify, prompts, report  # noqa: E402
from analyse_agent.data import market  # noqa: E402
from analyse_agent.llm import LLMError, get_client  # noqa: E402
from analyse_agent.schema import validate  # noqa: E402

log = logging.getLogger("analyse_agent")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history-dir", default=os.environ.get("HISTORY_DIR", "_local_history"))
    ap.add_argument("--tickers", help="comma list; overrides watchlist/positions")
    ap.add_argument("--models", help="comma list of model names from settings.yaml (default: all enabled)")
    ap.add_argument("--dry-run", action="store_true", help="skip LLM calls, email and history")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if the US market was closed today")
    ap.add_argument("--quiet", action="store_true", default=os.environ.get("CI") == "true",
                    help="no tickers/ratings in logs (default on in GitHub Actions: public repo logs are public)")
    return ap.parse_args()


def run_model(mcfg: dict, sheets: dict, market_rows: list, timeout: int) -> dict[str, dict]:
    """One model over all tickers, sequentially. Different models run in parallel threads."""
    out: dict[str, dict] = {}
    try:
        client = get_client(mcfg["provider"], mcfg["model"], timeout)
    except LLMError as e:
        return {tk: {"error": str(e)} for tk in sheets}
    gap = float(mcfg.get("min_interval_sec", 0))
    for tk, sheet in sheets.items():
        t0 = time.time()
        rec: dict = {}
        for attempt in (1, 2):
            try:
                verdict, meta = client.complete_json(prompts.SYSTEM, prompts.user_prompt(sheet, market_rows))
                errs = validate(verdict)
                if errs:
                    raise LLMError("schema: " + "; ".join(errs[:5]))
                verdict["confidence"] = int(verdict["confidence"])
                rec = {"verdict": verdict, "meta": meta}
                break
            except (LLMError, ValueError, KeyError) as e:
                rec = {"error": str(e)[:500]}
                log.warning("%s attempt %d failed: %s", mcfg["name"], attempt, str(e)[:200])
            except Exception as e:  # network etc.
                rec = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
                log.warning("%s attempt %d failed: %s", mcfg["name"], attempt, type(e).__name__)
        rec["seconds"] = round(time.time() - t0, 1)
        out[tk] = rec
        if gap:
            time.sleep(gap)
    return out


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    settings = config.load_settings()
    hdir = Path(args.history_dir).resolve()
    tickers, positions, cfg_used = config.load_universe(hdir)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    log.info("config: %s", ", ".join(p.name for p in cfg_used))
    log.info("universe: %d tickers", len(tickers))

    models = [m for m in settings["models"] if m.get("enabled", True)]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in settings["models"] if m["name"] in wanted]

    # ── data ──────────────────────────────────────────────────────────────────
    market_rows = market.market_snapshot(settings["market_symbols"])
    today_ny = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    last_session = next((m["date"] for m in market_rows if m["symbol"] == "SPY"), None)
    if last_session != today_ny and not (args.force or args.dry_run):
        log.info("US market closed today (last session %s, today %s) — skipping. Use --force to override.",
                 last_session, today_ny)
        return 0
    date = last_session or today_ny
    year, month = date[:4], date[5:7]

    sheets = {}
    for tk in tickers:
        sheets[tk] = factsheet.build(tk, settings, positions.get(tk))
        if not args.quiet:
            s = sheets[tk]
            log.info("%-6s close=%s house=%s gaps=%s", tk, s["technicals"].get("close"),
                     s["house_signal"]["value"], s["data_gaps"])

    rep_dir = hdir / "reports" / year / month
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{date}_factsheets.json").write_text(
        json.dumps({"market": market_rows, "sheets": sheets}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    if args.dry_run:
        log.info("dry run: fact sheets written to %s", rep_dir)
        return 0

    # ── LLMs (models in parallel, tickers sequential within a model) ─────────
    timeout = int(settings.get("llm_timeout_sec", 300))
    with ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        futs = {m["name"]: pool.submit(run_model, m, sheets, market_rows, timeout) for m in models}
        by_model = {name: f.result() for name, f in futs.items()}
    results = {tk: {name: by_model[name][tk] for name in by_model} for tk in sheets}
    model_names = [m["name"] for m in models]

    for name in model_names:
        ok = sum(1 for tk in sheets if results[tk][name].get("verdict"))
        log.info("model %-9s %d/%d ok", name, ok, len(sheets))

    # ── reports ──────────────────────────────────────────────────────────────
    paths = {}
    for lang in ("en", "zh"):
        p = rep_dir / f"{date}_daily.{lang}.md"
        p.write_text(report.render(lang, date, market_rows, sheets, results, model_names), encoding="utf-8")
        paths[lang] = p
    rel_report = paths["en"].relative_to(hdir).as_posix()

    # ── history ──────────────────────────────────────────────────────────────
    rows, raw = [], []
    for tk, sheet in sheets.items():
        tech = sheet.get("technicals", {})
        for name in model_names:
            r = results[tk][name]
            v = r.get("verdict") or {}
            lv = v.get("levels") or {}
            rows.append({
                "date": date, "ticker": tk, "model": name,
                "rating": v.get("rating", "ERROR"), "confidence": v.get("confidence", ""),
                "trend": v.get("trend", ""), "horizon_days": v.get("horizon_days", ""),
                "close": tech.get("close", ""), "house_signal": sheet["house_signal"]["value"],
                "agrees_with_house": v.get("agrees_with_house_signal", ""),
                "support": lv.get("support", ""), "resistance": lv.get("resistance", ""),
                "stop_loss": lv.get("stop_loss", ""),
                "in_position": bool(sheet.get("position")), "report_path": rel_report,
            })
            raw.append({"date": date, "ticker": tk, "model": name,
                        "model_id": next(m["model"] for m in models if m["name"] == name), **r})
    history.append_verdicts(hdir, rows, year)
    history.append_raw(hdir, raw, year)
    log.info("history: %d rows appended", len(rows))

    # ── email ────────────────────────────────────────────────────────────────
    if not args.no_email:
        if all(os.environ.get(k) for k in ("EMAIL_FROM", "EMAIL_PASS", "EMAIL_TO")):
            en, zh = paths["en"].read_text(encoding="utf-8"), paths["zh"].read_text(encoding="utf-8")
            notify.send_report(f"[AnalyseAgent] {date} daily · {len(sheets)} tickers · 每日决策面板",
                               en + "\n\n" + "=" * 60 + "\n\n" + zh, [paths["en"], paths["zh"]])
            log.info("email sent")
        else:
            log.warning("EMAIL_FROM/EMAIL_PASS/EMAIL_TO not set — email skipped")

    any_ok = any(results[tk][n].get("verdict") for tk in sheets for n in model_names)
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())

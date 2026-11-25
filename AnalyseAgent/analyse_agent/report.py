"""Render the daily dashboard as Markdown, one file per language."""
from __future__ import annotations

L = {
    "en": {
        "title": "Daily Dashboard", "market": "Market snapshot", "summary": "Summary",
        "symbol": "Symbol", "close": "Close", "d1": "1D %", "m1": "1M %",
        "ticker": "Ticker", "house": "House signal", "position": "Position",
        "shares": "shares @", "pnl": "unrealized", "rating": "Rating", "conf": "conf",
        "trend": "trend", "agree": "agrees with house signal", "yes": "yes", "no": "no",
        "bull": "Bull points", "bear": "Bear points", "risks": "Risks", "catalysts": "Catalysts",
        "levels": "Levels", "support": "support", "resistance": "resistance", "stop": "stop",
        "plan": "Action plan", "house_comment": "On the house signal", "horizon": "horizon",
        "days": "d", "failed": "failed", "split": "Models disagree",
        "disclaimer": "LLM-generated research for personal use. Not investment advice. No orders are placed.",
    },
    "zh": {
        "title": "每日决策面板", "market": "市场概览", "summary": "汇总",
        "symbol": "代码", "close": "收盘", "d1": "日涨跌 %", "m1": "月涨跌 %",
        "ticker": "股票", "house": "自有策略信号", "position": "持仓",
        "shares": "股 @", "pnl": "浮动盈亏", "rating": "评级", "conf": "置信度",
        "trend": "趋势", "agree": "与自有策略一致", "yes": "是", "no": "否",
        "bull": "看多理由", "bear": "看空理由", "risks": "风险", "catalysts": "催化剂",
        "levels": "关键价位", "support": "支撑", "resistance": "压力", "stop": "止损",
        "plan": "操作建议", "house_comment": "对自有策略信号的看法", "horizon": "周期",
        "days": "天", "failed": "失败", "split": "模型意见分歧",
        "disclaimer": "本报告由大模型生成，仅供个人研究参考，不构成投资建议，系统不会下单。",
    },
}


def _bullets(items) -> str:
    return "\n".join(f"- {x}" for x in (items or [])) or "- —"


def render(lang: str, date: str, market: list[dict], sheets: dict[str, dict],
           results: dict[str, dict[str, dict]], model_names: list[str]) -> str:
    t = L[lang]
    out = [f"# {t['title']} — {date}", ""]

    if market:
        out += [f"## {t['market']}", "",
                f"| {t['symbol']} | | {t['close']} | {t['d1']} | {t['m1']} |", "|---|---|---:|---:|---:|"]
        out += [f"| {m['symbol']} | {m['label']} | {m['close']} | {m['chg_1d_pct']:+.2f} | {m['chg_1m_pct']:+.2f} |"
                for m in market]
        out.append("")

    out += [f"## {t['summary']}", "",
            f"| {t['ticker']} | {t['close']} | {t['d1']} | {t['house']} | " + " | ".join(model_names) + " |",
            "|---|---:|---:|---|" + "---|" * len(model_names)]
    for tk, sheet in sheets.items():
        tech = sheet.get("technicals", {})
        cells = []
        ratings = set()
        for m in model_names:
            r = results.get(tk, {}).get(m, {})
            v = r.get("verdict")
            if v:
                ratings.add(v["rating"])
                cells.append(f"{v['rating']} ({v['confidence']}){' ✓' if v.get('agrees_with_house_signal') else ' ✗'}")
            else:
                cells.append(f"⚠ {t['failed']}")
        flag = " ⚠" if len(ratings) > 1 else ""
        pos = " 💼" if sheet.get("position") else ""
        d1 = tech.get("ret_1d_pct")
        out.append(f"| **{tk}**{pos}{flag} | {tech.get('close', '—')} | {f'{d1:+.2f}' if d1 is not None else '—'} | "
                   f"{sheet['house_signal']['value']} | " + " | ".join(cells) + " |")
    out += ["", f"✓/✗ = {t['agree']} · ⚠ = {t['split']} · 💼 = {t['position']}", ""]

    for tk, sheet in sheets.items():
        tech = sheet.get("technicals", {})
        out += ["---", "", f"## {tk} — {tech.get('close', '—')}  ({t['house']}: `{sheet['house_signal']['value']}`)", ""]
        if sheet.get("position"):
            p = sheet["position"]
            out.append(f"{t['position']}: {p.get('shares', '?')} {t['shares']} {p.get('avg_cost', '?')}"
                       + (f" · {t['pnl']} {p['unrealized_pct']:+.2f}%" if "unrealized_pct" in p else ""))
            out.append("")
        for m in model_names:
            r = results.get(tk, {}).get(m, {})
            v = r.get("verdict")
            if not v:
                out += [f"### {m} — ⚠ {t['failed']}", "", f"`{r.get('error', 'no result')}`", ""]
                continue
            b = v[lang]
            lv = v.get("levels", {})
            out += [
                f"### {m} — **{v['rating']}** ({t['conf']} {v['confidence']}) · {t['trend']} {v['trend']} · "
                f"{t['horizon']} {v['horizon_days']}{t['days']} · {t['agree']}: {t['yes'] if v['agrees_with_house_signal'] else t['no']}",
                "",
                f"**{b['headline']}**", "", b["summary"], "",
                f"**{t['bull']}**", _bullets(b["bull_points"]), "",
                f"**{t['bear']}**", _bullets(b["bear_points"]), "",
                f"**{t['risks']}**", _bullets(b["risks"]), "",
                f"**{t['catalysts']}**", _bullets(b["catalysts"]), "",
                f"**{t['levels']}**: {t['support']} {lv.get('support')} · {t['resistance']} {lv.get('resistance')} · {t['stop']} {lv.get('stop_loss')}",
                "",
                f"**{t['plan']}**: {b['action_plan']}", "",
                f"**{t['house_comment']}**: {b['house_view_comment']}", "",
            ]
    out += ["---", f"_{t['disclaimer']}_", ""]
    return "\n".join(out)

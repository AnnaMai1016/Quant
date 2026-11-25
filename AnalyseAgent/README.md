# AnalyseAgent

Read-only LLM research agent for US stocks. **It never places orders.**

Every US trading day after the close (GitHub Actions, `.github/workflows/analyse_daily.yml`):

1. **Data**: yfinance prices, technicals, fundamentals, earnings date and news, plus your own
   `trend_follower.py` signal (the "house signal") recomputed on 30-min bars.
2. **LLM dashboard**: the same fact sheet goes to every enabled model (Claude via headless Claude Code,
   DeepSeek, Gemini). Each returns validated JSON: rating, confidence, levels, bull/bear/risks/catalysts,
   an action plan, and whether it agrees with the house signal, in **English and Chinese**.
3. **Output** goes to the private repo `Quant-history`:
   - `reports/YYYY/MM/<date>_daily.en.md`, `.zh.md`, `_factsheets.json`
   - `history/verdicts_YYYY.csv`: one row per ticker × model, append-only (used later to score accuracy)
   - `history/raw_YYYY.jsonl`: full model outputs, errors, token usage
4. **Email** with both languages, plus the `.md` files attached.

## Private config (in Quant-history, never in this public repo)
- `config/watchlist.yaml`: see `config/watchlist.example.yaml`
- `config/positions.yaml`: see `config/positions.example.yaml`; edit by hand when you trade

## Secrets (Quant repo, Settings, Secrets and variables, Actions)
`CLAUDE_CODE_OAUTH_TOKEN`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `HISTORY_REPO_TOKEN`,
`EMAIL_FROM`, `EMAIL_PASS`, `EMAIL_TO`

## Run locally
```
pip install -r requirements.txt
python run_daily.py --dry-run                         # data only
python run_daily.py --tickers NVDA --models claude --no-email --force
```
Models and the house-signal parameters are in `config/settings.yaml`.

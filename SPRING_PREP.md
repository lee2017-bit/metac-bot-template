# Spring-season preparation (branch `spring-prep`, not live)

Everything here is off unless switched on; `main` and the running bot are unchanged.

| Piece | Switch | What it does | Checked offline |
|---|---|---|---|
| `log_privacy.py` | `LOG_PRIVATE_TEXT=1` | Shortens any log line over `LOG_TEXT_LIMIT` (400) characters to its first line, so research and reasoning stay out of public Actions logs. Costs, URLs, errors and the summary banner stay. Saved reports are unchanged. | long research record cut to one line; short cost line untouched |
| `read_page` tool in `claude_research.py` | `RESEARCH_MAX_PAGE_READS=N` | The bot's own page reader for the Claude research path (same code as the free path): opens any URL, including ones Anthropic's `web_fetch` refuses because they were not in search results, and reads PDF/CSV/JSON. metaculus.com refused. Adds one rule line to the research prompt only when on. | real page read, bad URL, limit, metaculus.com refused; rule line appears only when on |
| `bot_once.yaml` + `tools/precheck.py` | (new workflow) | Short one-shot run: a stdlib-only pre-check asks Metaculus whether any open question lacks our forecast and only then installs dependencies and runs `main.py`. Idle ticks take well under a minute, so a private repository stays inside the free Actions minutes (two ticks per hour ≈ 1,500 min/month). Empty report files are not uploaded. Same concurrency group as the loop. Schedules only fire on the default branch. | fail-open without a token; decision with fake API pages (missing → run, none → skip) |
| `tools/review_variants.py` | (script) | Scores how the posted forecasts would have done with other aggregations: all forecasters mean/median/odds, every 3-of-N subset, a single forecaster, each model alone. No LLM calls. | 26 archived reports with made-up resolutions; binary `posted` = `all/median`, multiple-choice `posted` = `all/mean`, as the bot aggregates |

Not verified against the live API: whether the posts endpoint returns `question.my_forecasts.latest` for open questions (if not, the pre-check always answers run=true, which is safe but slower).

Open point for the spring season: in a public repository the uploaded `reports-*` artifacts (full research and reasoning) can be downloaded by any signed-in GitHub user while questions are still open. A private repository (a copy, since forks of public repositories stay public) avoids this; `bot_once.yaml` is what keeps the Actions minutes affordable there.

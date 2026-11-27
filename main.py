import argparse
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Literal

import dotenv

import profile_env  # first: applies CONFIG_PROFILE before other modules read settings

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

from claude_llm import ClaudeForecastLlm, current_model, parse_ensemble
from claude_research import MAX_FETCHES, MAX_MARKET_LOOKUPS, run_claude_web_research
from free_research import run_free_research
from openai_compat import OpenAICompatForecastLlm, parser_llm
from discrete_values import (
    MARKER as DISCRETE_MARKER,
    build_cdf,
    format_value,
    outcome_values,
    parse_outcome_probabilities,
    step_decimals,
)
from forecasting_tools import MonetaryCostManager
import os
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

# Probability limits (user-approved 11/26; item T4 of the 11/18 open-source review).
# Every reviewed prize-winning open-source bot keeps binary forecasts inside 2%-98%
# and every multiple-choice option at 1% or more: at 99% instead of 98% a correct
# call gains about 1 point and a wrong one loses about 69 more. The defaults here are
# the template's old limits; the workflows set the tighter ones.
def _limit(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


BINARY_PROB_MIN = _limit("BINARY_PROB_MIN", 0.01, 0.001, 0.49)
BINARY_PROB_MAX = _limit("BINARY_PROB_MAX", 0.99, 0.51, 0.999)
MC_PROB_MIN = _limit("MC_PROB_MIN", 0.0, 0.0, 0.2)


# Multiple-choice answers with the option names written out (daily check 11/27). The
# template asks for "Option_A: p" lines and leaves it to the parser model to match the
# letters to the option names. On question 46756 (5 options) the parser swapped the two
# last options in 2 of 4 forecasts and could not agree with itself on a fifth, which was
# dropped. With MC_NAMED_OPTIONS=1 every answer line carries the name, and the parser
# is told which letter is which option.
MC_NAMED_OPTIONS = os.getenv("MC_NAMED_OPTIONS", "0") == "1"


def _option_labels(options: list[str]) -> list[str]:
    return [
        f"Option_{chr(65 + i)}" if i < 26 else f"Option_{i + 1}"
        for i in range(len(options))
    ]


def mc_answer_format(options: list[str]) -> str:
    if not MC_NAMED_OPTIONS:
        return (
            "The last thing you write is your final probabilities for the N options "
            f"in this order {options} as:\n"
            "Option_A: Probability_A\nOption_B: Probability_B\n...\nOption_N: Probability_N"
        )
    lines = "\n".join(
        f"{label} ({name}): Probability"
        for label, name in zip(_option_labels(options), options)
    )
    return (
        "The last thing you write is your final probabilities, one line per option, "
        "exactly in this form (keep each label and option name as written and replace "
        "only the word Probability with your number):\n" + lines
    )


def mc_parser_note(options: list[str]) -> str:
    if not MC_NAMED_OPTIONS:
        return ""
    mapping = "; ".join(
        f"{label} = {name}" for label, name in zip(_option_labels(options), options)
    )
    return (
        "\nThe labels refer to the options in exactly this order: " + mapping + ". "
        "When a line gives a label and an option name, they are the same option. Keep "
        "every probability with the option it was written for; never move a probability "
        "to a different option."
    )


def apply_option_floor(
    predicted: PredictedOptionList, minimum: float = MC_PROB_MIN
) -> PredictedOptionList:
    """Raise every option to at least `minimum`, taking the difference proportionally
    from the other options. With more than ten options the floor shrinks so that the
    floors together never take more than a tenth of the probability."""
    options = predicted.predicted_options
    if minimum <= 0 or len(options) < 2:
        return predicted
    floor = min(minimum, 0.10 / len(options))
    if all(o.probability >= floor for o in options):
        return predicted
    probabilities = PredictedOptionList._normalize_with_minimum(
        [o.probability for o in options], minimum_probability=floor
    )
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=o.option_name, probability=prob)
            for o, prob in zip(options, probabilities)
        ]
    )

from spend_guard import check_key_expiry, record_run, remaining_budget


def _provider() -> str:
    """'anthropic' or 'openai_compat'. PROVIDER_SWITCH_AT (ISO time) lets the
    workflow carry both configurations and change provider at a fixed moment,
    e.g. when one credit expires."""
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    switch = os.getenv("PROVIDER_SWITCH_AT", "").strip()
    if switch:
        at = datetime.fromisoformat(switch.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) >= at:
            provider = os.getenv("LLM_PROVIDER_AFTER_SWITCH", "openai_compat")
    return provider


PROVIDER = _provider()

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


# Evidence-handling rules for research and forecasting
# (resolution-source alignment, dated facts, cross-checks, claims vs actions,
# "unknown" instead of guessing). Kept on one line so clean_indents is unaffected.
RESEARCH_RULES = (
    "Also: (1) identify the exact source or data series the question resolves on, its latest released value, "
    "and when the next release before the resolution date is scheduled; (2) give the publication date of every key fact "
    "and say if it may be stale; (3) cross-check any number that matters against a second independent source and note "
    "disagreements; (4) separate announcements, targets and plans from actions already completed, and note how often "
    "similar announced plans have slipped; (5) list what you could not find instead of guessing."
)
FORECAST_RULES = (
    "Evidence discipline: re-read the resolution criteria and fine print for edge cases (exact dates, time zones, "
    "definitions, which source counts); weigh announcements and plans less than completed actions; prefer the most "
    "recent dated facts; where the research has no evidence on a point, treat it as unknown and lean toward the base "
    "rate rather than an extreme. If current betting odds or prediction-market prices exist for the event, convert "
    "them to implied probabilities (removing the bookmaker margin) and use them as your anchor, departing from them "
    "only for a specific reason the market is unlikely to have priced."
)
if MAX_MARKET_LOOKUPS > 0:
    FORECAST_RULES += (
        " Use a market as an anchor only if its wording, thresholds and dates match this question; where they "
        "differ, adjust for the difference explicitly."
    )
# Discrete questions with at most this many outcomes are forecast as one
# probability per outcome instead of percentiles (0 = off).
DISCRETE_PER_VALUE_MAX = int(os.getenv("DISCRETE_PER_VALUE_MAX", "0"))
# Give the researcher the whole question (type, options, unit, range, dates,
# background) and the tool rules, instead of the template's question text +
# resolution criteria only.
RESEARCH_FULL_CONTEXT = os.getenv("RESEARCH_FULL_CONTEXT", "0") == "1"


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        3  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    ##################################### ENSEMBLE #####################################

    # Models used for the N predictions of one question, in order
    # (e.g. 2 Fable + 3 Opus). Set from FORECAST_ENSEMBLE in __main__.
    ensemble_models: list[str] = []
    _ensemble_counter: dict = {}

    async def _make_prediction(self, question, research):
        if not self.ensemble_models:
            return await super()._make_prediction(question, research)
        key = question.page_url or id(question)
        # Read and bump without awaiting in between, so concurrent predictions
        # for the same question get different models.
        i = self._ensemble_counter.get(key, 0)
        self._ensemble_counter[key] = i + 1
        token = current_model.set(self.ensemble_models[i % len(self.ensemble_models)])
        try:
            return await super()._make_prediction(question, research)
        finally:
            current_model.reset(token)

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if researcher == "claude-web-search" and PROVIDER == "openai_compat":
                research = await run_free_research(prompt)
            elif researcher == "claude-web-search":
                research = await run_claude_web_research(prompt)
            elif isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _question_shape_for_research(question: MetaculusQuestion) -> str:
        """Type, options, unit, range and dates, so the researcher knows what exactly is asked."""
        lines: list[str] = []
        if isinstance(question, MultipleChoiceQuestion):
            lines.append(f"Question type: multiple choice. Options: {question.options}")
        elif isinstance(question, NumericQuestion):
            unit = question.unit_of_measure or "not stated"
            low = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            high = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            kind = "discrete number" if question.cdf_size != 201 else "number"
            lines.append(
                f"Question type: {kind}. Unit: {unit}. The question creator expects an answer between {low} and {high} "
                f"({'values below are possible' if question.open_lower_bound else 'cannot be lower'}; "
                f"{'values above are possible' if question.open_upper_bound else 'cannot be higher'})."
            )
        elif isinstance(question, DateQuestion):
            lines.append(
                f"Question type: date, expected between {question.lower_bound.date().isoformat()} and {question.upper_bound.date().isoformat()}."
            )
        else:
            lines.append("Question type: binary (Yes or No).")
        dates = []
        for label, value in (
            ("opened", question.open_time),
            ("closes to forecasts", question.close_time),
            ("scheduled to resolve", question.scheduled_resolution_time),
        ):
            if value is not None:
                dates.append(f"{label} {value.strftime('%Y-%m-%d %H:%M UTC')}")
        if dates:
            lines.append("Question timeline: " + "; ".join(dates) + ".")
        return "\n".join(lines)

    @staticmethod
    def _research_tool_rules(question: MetaculusQuestion) -> str:
        rules: list[str] = []
        if MAX_FETCHES > 0:
            text = " ".join(
                t or ""
                for t in (
                    question.resolution_criteria,
                    question.fine_print,
                    question.background_info,
                    question.question_text,
                )
            )
            urls: list[str] = []
            for url in re.findall(r"https?://[^\s<>()\[\]\"']+", text):
                url = url.rstrip(".,;:")
                if "metaculus.com" not in url.lower() and url not in urls:
                    urls.append(url)
            named = (
                " URLs named in the question (resolution criteria first): " + " , ".join(urls[:6]) + " ."
                if urls
                else ""
            )
            rules.append(
                f"Your first step is to open, with web_fetch, the page, dataset or document that decides the outcome or holds "
                f"its latest published value, when the question names one, and to report the latest figure on it with its date.{named} "
                f"You have at most {MAX_FETCHES} page reads; spend any left over on the most important sources you find instead of "
                "relying on search snippets, and say which pages could not be opened."
            )
        if MAX_MARKET_LOOKUPS > 0:
            rules.append(
                "Call market_search once or twice with short keyword queries to check whether Polymarket, Kalshi or Manifold has "
                "a market on the same or a closely related event. End your write-up with a section headed 'Prediction markets' "
                "that gives, for each relevant market, its exact title, current price, close date and how its wording, "
                "thresholds or dates differ from this question; write 'No relevant market found' if there is none, and leave "
                "out markets about something else. If you find bookmaker odds, give them with the time they were quoted."
            )
        rules.append(
            "Run several distinct searches before writing: at least one for developments in the past week, one for the "
            "resolution source or official data, and one for base rates or precedents."
        )
        rules.append(
            "The background below was written when the question was created: if what you find contradicts it, say so "
            "explicitly and state which figure is current, with dates."
        )
        return " ".join(rules)

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        if not RESEARCH_FULL_CONTEXT:
            return clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.
                Today is {datetime.now().strftime("%Y-%m-%d")}. Search the web for the latest information, and include relevant base rates and any scheduled events before the resolution date. {RESEARCH_RULES}

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.
            Today is {datetime.now().strftime("%Y-%m-%d")}. Search the web for the latest information, and include relevant base rates and any scheduled events before the resolution date. {RESEARCH_RULES}
            {FallTemplateBot2026._research_tool_rules(question)}

            Question:
            {question.question_text}

            {FallTemplateBot2026._question_shape_for_research(question)}

            Background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {FORECAST_RULES}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(
            BINARY_PROB_MIN, min(BINARY_PROB_MAX, binary_prediction.prediction_in_decimal)
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {FORECAST_RULES}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            {mc_answer_format(question.options)}
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.{mc_parser_note(question.options)}
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        predicted_option_list = apply_option_floor(predicted_option_list)

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        values = self._per_value_outcomes(question)
        if values is not None:
            try:
                return await self._run_forecast_on_discrete_values(
                    question, research, values, lower_bound_message, upper_bound_message
                )
            except Exception as e:  # fall back to the percentile path below
                logger.warning(
                    f"Per-outcome forecast failed for {question.page_url} "
                    f"({type(e).__name__}: {e}); using percentiles instead"
                )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 1 should always be less than the value for percentile 5, and so on.

            {FORECAST_RULES}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns. Your percentile 1 and 99 values are genuinely extreme but still possible outcomes, not the edges of the question's range.

            The last thing you write is your final answer as:
            "
            Percentile 1: XX (lowest number value)
            Percentile 5: XX
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            Percentile 95: XX
            Percentile 99: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = self._tail_mixture_distribution(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def _per_value_outcomes(question: NumericQuestion) -> list[float] | None:
        """The outcome list if this is a small discrete question, else None."""
        if DISCRETE_PER_VALUE_MAX <= 0 or question.cdf_size is None:
            return None
        n = question.cdf_size - 1
        if question.cdf_size == 201 or not 2 <= n <= DISCRETE_PER_VALUE_MAX:
            return None
        if question.zero_point is not None:
            return None
        return outcome_values(question.lower_bound, question.upper_bound, question.cdf_size)

    async def _run_forecast_on_discrete_values(
        self,
        question: NumericQuestion,
        research: str,
        values: list[float],
        lower_bound_message: str,
        upper_bound_message: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        step = values[1] - values[0]
        decimals = step_decimals(step)
        labels = [format_value(v, decimals) for v in values]
        unit = question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"
        answer_lines = []
        if question.open_lower_bound:
            answer_lines.append(f"Below {labels[0]}: XX%")
        answer_lines += [f"{label}: XX%" for label in labels]
        if question.open_upper_bound:
            answer_lines.append(f"Above {labels[-1]}: XX%")
        answer_format = "\n".join(answer_lines)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {unit}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            The answer is one of a fixed list of outcomes: {", ".join(labels)}. If the true value falls between two listed outcomes it is rounded to the nearest one as the resolution criteria describe.

            {FORECAST_RULES}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            (g) Which listed outcomes are already impossible (for example a running total cannot fall below the level already reached).

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble: keep some probability on outcomes on both sides of your central estimate to account for unknown unknowns, but do not spread probability onto outcomes the evidence rules out.

            The last thing you write is the line "{DISCRETE_MARKER}" followed by one line for every outcome, in exactly this format, with probabilities that sum to 100%:
            "
            {DISCRETE_MARKER}
            {answer_format}
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        inbound, below, above = parse_outcome_probabilities(reasoning, values)
        cdf = build_cdf(
            inbound, below, above, question.open_lower_bound, question.open_upper_bound
        )
        percentiles = [
            Percentile(value=question.lower_bound + i * step, percentile=p)
            for i, p in enumerate(cdf)
        ]
        prediction = NumericDistribution.from_question(percentiles, question)
        prediction.get_cdf()  # build and validate now, so a bad distribution falls back
        summary = ", ".join(f"{label}: {p * 100:.1f}%" for label, p in zip(labels, inbound))
        logger.info(
            f"Forecasted URL {question.page_url} with per-outcome probabilities: {summary} "
            f"(below {below * 100:.1f}%, above {above * 100:.1f}%)."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def _tail_mixture_distribution(
        percentile_list: list[Percentile], question: NumericQuestion
    ) -> NumericDistribution:
        """
        With only percentiles 10..90, the library puts the 5th/95th percentile
        exactly at the question bounds, i.e. 5% of the mass below the lower
        bound and 5% above the upper bound however far away they are. Here the
        model also gives 1/5/95/99, and the final CDF is 80% the full-tail CDF
        + 20% the 10..90-only CDF (a safety net if the stated tails are too
        narrow). In simulation (t-distributed outcomes, open and closed bounds,
        well-calibrated to 3x overconfident forecasters) this mixture beat the
        template in all but the most overconfident closed-bound case.
        """
        core = [p for p in percentile_list if 0.1 - 1e-9 <= p.percentile <= 0.9 + 1e-9]
        # The library rejects values more than 2x the question range beyond a
        # bound; pull such extreme tails in just inside that limit.
        span = question.upper_bound - question.lower_bound
        lo, hi = question.lower_bound - 1.9 * span, question.upper_bound + 1.9 * span
        clamped, last = [], None
        for p in sorted(percentile_list, key=lambda x: x.percentile):
            v = min(max(p.value, lo), hi)
            if last is not None and v <= last:
                v = last + 1e-9 * max(span, 1)
            clamped.append(Percentile(percentile=p.percentile, value=v))
            last = v
        percentile_list = clamped
        try:
            full = NumericDistribution.from_question(percentile_list, question)
            if len(core) < 2 or len(core) == len(percentile_list):
                return full
            core_dist = NumericDistribution.from_question(core, question)
            full_cdf, core_cdf = full.get_cdf(), core_dist.get_cdf()
            mixed = [
                Percentile(
                    value=a.value, percentile=0.8 * a.percentile + 0.2 * b.percentile
                )
                for a, b in zip(full_cdf, core_cdf)
            ]
            return NumericDistribution.from_question(mixed, question)
        except Exception as e:  # fall back to the template behaviour
            logger.warning(f"Tail mixture failed ({e}); using 10..90 percentiles only")
            return NumericDistribution.from_question(core or percentile_list, question)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            {FORECAST_RULES}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    parser.add_argument(
        "--tournaments",
        choices=["both", "fall", "minibench"],
        default="both",
        help="In tournament mode: which tournaments to forecast (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forecast but do not publish anything to Metaculus",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    if PROVIDER == "openai_compat":
        # Separate ledger, cap and cutoff for the second provider. A cap left
        # unset becomes 0 (the guard then refuses to run); a cutoff left unset
        # keeps the first provider's cutoff (the bot stops there).
        oc_defaults = {
            "SPEND_LEDGER": "spend/ledger_oc.json",
            "SPEND_CAP_USD": "0",
            "SPEND_OFFSET_USD": "0",
            "RUN_CAP_USD": "0",
            "LLM_KEY_EXPIRES": os.getenv("LLM_KEY_EXPIRES", ""),
        }
        for name, default in oc_defaults.items():
            os.environ[name] = os.getenv(name + "_OC", default)
    if profile_env.NOTE:
        print(profile_env.NOTE)
    print(f"LLM provider: {PROVIDER}")
    check_environment(strict=True)
    check_key_expiry()
    budget_left_usd = remaining_budget()
    publish_to_metaculus = not args.dry_run
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. The `llms=` block below is commented out to use
    # whichever default models forecasting-tools picks based on your env vars;
    # uncomment and edit to pin specific models.
    # Models are pinned explicitly (forecasting-tools otherwise picks them by
    # which keys exist, not by which keys work). Claude 5.5 models take no
    # temperature, so it is None everywhere.
    if PROVIDER == "openai_compat":
        ENSEMBLE = parse_ensemble(os.getenv("FORECAST_ENSEMBLE_OC", "moonshotai/Kimi-K3:3"))
        default_llm = OpenAICompatForecastLlm(ENSEMBLE[0])
        parser = parser_llm(os.getenv("PARSER_MODEL_OC", "zai-org/GLM-5.3-Flash"))
    else:
        ENSEMBLE = parse_ensemble(
            os.getenv("FORECAST_ENSEMBLE", "claude-fable-5-2:2,claude-opus-5-5:3")
        )
        FORECAST_EFFORT = os.getenv("FORECAST_EFFORT", "high")
        PARSER_MODEL = os.getenv("PARSER_MODEL", "anthropic/claude-haiku-5-5")
        default_llm = ClaudeForecastLlm(ENSEMBLE[0], effort=FORECAST_EFFORT)
        parser = GeneralLlm(model=PARSER_MODEL, temperature=None)
    FallTemplateBot2026.ensemble_models = ENSEMBLE
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=len(ENSEMBLE),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to="reports",
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": default_llm,
            "summarizer": parser,
            "researcher": os.getenv("RESEARCHER", "claude-web-search"),
            "parser": parser,
        },
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    run_cost_manager = MonetaryCostManager(hard_limit=budget_left_usd)
    run_cost_manager.__enter__()
    # Record spend even if the run crashes part-way.
    import atexit

    _recorded = {"done": False}

    def _record_on_exit() -> None:
        if not _recorded["done"]:
            record_run(run_cost_manager.current_usage, 0, -1, run_mode + " (crashed)")

    atexit.register(_record_on_exit)
    if run_mode == "tournament":
        seasonal_tournament_reports = []
        minibench_reports = []
        if args.tournaments in ("both", "fall"):
            seasonal_tournament_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
            )
        if args.tournaments in ("both", "minibench"):
            minibench_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    client.CURRENT_MINIBENCH_ID, return_exceptions=True
                )
            )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    run_cost_manager.__exit__(None, None, None)
    n_errors = sum(1 for r in forecast_reports if isinstance(r, BaseException))
    record_run(
        run_cost_manager.current_usage,
        len(forecast_reports),
        n_errors,
        run_mode
        + ("" if run_mode != "tournament" or args.tournaments == "both" else f":{args.tournaments}")
        + (" (dry run)" if args.dry_run else ""),
    )
    _recorded["done"] = True
    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )

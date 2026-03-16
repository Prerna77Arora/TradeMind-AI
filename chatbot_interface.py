"""
chatbot_interface.py
---------------------
AI-powered investment chatbot using Google Gemini API.

The chatbot:
  1. Asks about the investor's goals and risk tolerance
  2. Explains recommendations in plain language
  3. Answers follow-up questions about stocks, markets, and strategy
  4. Generates a personalised portfolio strategy summary

Requires: GOOGLE_API_KEY environment variable (or passed directly).
"""

import os
from typing import Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Chatbot] google-generativeai not installed. "
          "Run: pip install google-generativeai")


SYSTEM_PROMPT = """You are an expert AI investment advisor named 'FinanceGPT'.
You have deep knowledge of:
- Stock market analysis and technical indicators
- LSTM-based price prediction models
- Risk management (stop loss, take profit, position sizing)
- Portfolio management and diversification
- Indian stock market (NSE/BSE) and global markets
- Mutual funds, ETFs, and derivatives

Your personality:
- Professional yet approachable
- Explain complex concepts simply (no unnecessary jargon)
- Always acknowledge risks — never promise returns
- Tailor advice to the user's risk profile and goals
- Be concise: 2–4 sentences per point unless asked for detail

IMPORTANT DISCLAIMER: You provide educational information only, not regulated
financial advice. Always recommend consulting a SEBI-registered advisor for
large investments.

When presenting recommendations:
- Use bullet points for clarity
- Always mention stop loss and risk management
- Relate advice to the user's specific profile

Start by greeting the user and asking about their investment goals.
"""

PROFILE_QUESTIONS = [
    "What is your total available investment capital? (e.g., ₹1,00,000)",
    "How would you describe your risk tolerance?\n  a) Conservative (low risk)\n  b) Balanced (medium risk)\n  c) Aggressive (high risk)",
    "What is your investment horizon? (e.g., 1 year, 3 years, 10 years)",
    "What is your primary investment goal?\n  a) Capital preservation\n  b) Regular income (dividends)\n  c) Long-term wealth creation\n  d) Short-term trading",
    "Are there any sectors or stocks you want to focus on or avoid?"
]


class StockAdvisorChatbot:
    """
    Gemini-powered investment chatbot that:
      - Collects investor profile through conversation
      - Explains AI model recommendations in natural language
      - Answers stock market Q&A
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialise the chatbot with the Gemini API.

        Args:
            api_key: Google Gemini API key (falls back to GOOGLE_API_KEY env var)
            model_name: Gemini model to use
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

        key = api_key or os.environ.get('GOOGLE_API_KEY', '')
        if not key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_API_KEY env var or pass api_key."
            )

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT
        )
        self.chat = self.model.start_chat(history=[])
        self.context_injected = False
        print(f"[Chatbot] FinanceGPT ready (model: {model_name})")

    def inject_recommendation_context(
        self,
        recommendations: list,
        profile_summary: str = ""
    ) -> None:
        """
        Feed the chatbot with recommendation results so it can explain them.

        Args:
            recommendations: List of Recommendation objects
            profile_summary: Formatted investor profile string
        """
        if not recommendations:
            return

        context_lines = [
            "CONTEXT — AI MODEL RECOMMENDATIONS (for your reference only):",
            profile_summary,
            ""
        ]

        for rec in recommendations:
            context_lines.append(
                f"Ticker: {rec.ticker} | Action: {rec.action} | "
                f"Current: ₹{rec.current_price:,.2f} | "
                f"Predicted: ₹{rec.predicted_price:,.2f} ({rec.predicted_change_pct:+.2f}%) | "
                f"Stop Loss: ₹{rec.stop_loss:,.2f} | "
                f"Take Profit: ₹{rec.take_profit:,.2f} | "
                f"Confidence: {rec.confidence} | "
                f"Trend: {rec.filter_results.get('trend', 'N/A')} | "
                f"RSI: {rec.filter_results.get('rsi', 'N/A')} | "
                f"Sharpe: {rec.risk_metrics.get('sharpe_ratio', 'N/A')}"
            )

        context_text = "\n".join(context_lines)
        # Silent context injection — does not show to user
        self.chat.send_message(
            f"[SYSTEM CONTEXT — use this to answer user questions]:\n{context_text}"
        )
        self.context_injected = True
        print("[Chatbot] Recommendation context injected.")

    def generate_strategy_report(
        self,
        profile_summary: str,
        recommendations: list
    ) -> str:
        """
        Ask Gemini to generate a written portfolio strategy report.

        Args:
            profile_summary: Formatted investor profile
            recommendations: List of Recommendation objects

        Returns:
            Formatted strategy report string
        """
        rec_details = "\n".join([
            f"- {r.ticker}: {r.action} | Predicted {r.predicted_change_pct:+.2f}% | "
            f"Confidence: {r.confidence} | Strategy: {r.strategy_summary}"
            for r in recommendations
        ])

        prompt = (
            f"Based on the following investor profile and AI-generated recommendations, "
            f"write a comprehensive portfolio strategy report (400-500 words):\n\n"
            f"INVESTOR PROFILE:\n{profile_summary}\n\n"
            f"RECOMMENDATIONS:\n{rec_details}\n\n"
            f"Include: portfolio allocation breakdown, risk management plan, "
            f"entry/exit strategy, and a 3-point action plan."
        )

        response = self.chat.send_message(prompt)
        return response.text

    def chat_response(self, user_message: str) -> str:
        """
        Send a user message and get a chatbot response.

        Args:
            user_message: User's input text

        Returns:
            Chatbot response text
        """
        response = self.chat.send_message(user_message)
        return response.text

    def run_interactive_session(
        self,
        recommendations: list = None,
        profile_summary: str = ""
    ) -> None:
        """
        Launch an interactive terminal chat session.

        Args:
            recommendations: Optional pre-computed recommendations to discuss
            profile_summary: Investor profile for context
        """
        print("\n" + "═" * 55)
        print("  🤖  FINACEGPT — AI INVESTMENT ADVISOR")
        print("  Type 'quit' or 'exit' to end the session.")
        print("  Type 'report' to generate a strategy report.")
        print("═" * 55 + "\n")

        # Inject context
        if recommendations:
            self.inject_recommendation_context(recommendations, profile_summary)

        # Greeting
        greeting = self.chat_response(
            "Please greet the user and briefly explain what you can help with."
        )
        print(f"FinanceGPT: {greeting}\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Chatbot] Session ended.")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'bye'):
                farewell = self.chat_response("Say a brief professional farewell.")
                print(f"\nFinanceGPT: {farewell}")
                break
            if user_input.lower() == 'report' and recommendations:
                print("\nFinanceGPT: Generating your portfolio strategy report...\n")
                report = self.generate_strategy_report(profile_summary, recommendations)
                print(report)
                continue

            response = self.chat_response(user_input)
            print(f"\nFinanceGPT: {response}\n")


def create_chatbot(api_key: str = None) -> Optional['StockAdvisorChatbot']:
    """
    Factory function to create a chatbot, handling missing dependencies gracefully.

    Args:
        api_key: Gemini API key

    Returns:
        StockAdvisorChatbot or None if unavailable
    """
    if not GEMINI_AVAILABLE:
        print("[Chatbot] Gemini not available. Skipping chatbot.")
        return None

    try:
        return StockAdvisorChatbot(api_key=api_key)
    except (ValueError, Exception) as e:
        print(f"[Chatbot] Could not initialise: {e}")
        return None

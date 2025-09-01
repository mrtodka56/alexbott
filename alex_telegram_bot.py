import os
import json
import asyncio
import aiohttp
import ast
import math
from typing import Tuple, List
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ---------- CONFIG ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ACCESS_KEY = os.getenv("ACCESS_KEY", "Alex wake up")  # optional override via env
MODEL_DAILY = "deepseek/deepseek-chat-v3.1:free"  # daily chat (DeepSeek)
MODEL_VENICE = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"  # AMA
AUTH_FILE = "authenticated.json"
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://alexbott.onrender.com")
# ----------------------------

# Load authenticated users (persist across restarts)
if os.path.exists(AUTH_FILE):
    try:
        with open(AUTH_FILE, "r") as f:
            authenticated_users = set(json.load(f))
    except Exception:
        authenticated_users = set()
else:
    authenticated_users = set()


def save_authenticated_users():
    try:
        with open(AUTH_FILE, "w") as f:
            json.dump(list(authenticated_users), f)
    except Exception:
        pass


# ---------- Utilities ----------
async def duckduckgo_search(query: str, max_results: int = 3) -> List[Tuple[str, str]]:
    """
    Use DuckDuckGo Instant Answer API to get basic results.
    Returns list of (title, snippet).
    """
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                results = []

                # Try RelatedTopics and Abstract
                if data.get("AbstractText"):
                    results.append(("Abstract", data.get("AbstractText")))
                # RelatedTopics may contain nested lists/dicts
                rt = data.get("RelatedTopics", [])
                for item in rt:
                    if isinstance(item, dict):
                        text = item.get("Text") or item.get("Result")
                        if text:
                            results.append((item.get("FirstURL", "Result"), text))
                    if len(results) >= max_results:
                        break

                # Fallback: use Heading / AbstractURL
                if not results and data.get("Heading"):
                    results.append((data.get("Heading"), data.get("AbstractText", "")))

                return results[:max_results]
    except Exception:
        return []


def needs_search(message: str) -> bool:
    """
    Heuristics to auto-detect when to use web search.
    """
    msg = message.lower()
    keywords = ["latest", "today", "news", "trending", "who won", "who is", "when", "what happened",
                "score", "update", "breaking", "is it true", "did", "did they", "how many", "weather",
                "stock", "price", "trend", "recent"]
    if any(kw in msg for kw in keywords):
        return True
    # if user includes a year or date-ish token
    if any(token.isdigit() and len(token) >= 3 for token in msg.split()):
        return True
    # default false
    return False


def safe_eval(expr: str) -> str:
    """
    Safe math expression evaluator using ast — supports numbers and basic operators.
    Returns result as string or raises ValueError.
    """
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.UAdd, ast.USub, ast.Load, ast.BitXor, ast.LShift, ast.RShift
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid constant")
        if isinstance(node, ast.Num):  # older versions
            return node.n
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("Unsupported unary op")
        raise ValueError("Unsupported expression")

    try:
        parsed = ast.parse(expr, mode="eval")
        # validate nodes
        for n in ast.walk(parsed):
            if type(n) not in allowed_nodes:
                raise ValueError("Unsupported expression")
        result = _eval(parsed)
        # nice formatting: int if whole number
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as e:
        raise ValueError("Invalid math expression") from e


# ---------- OpenRouter interaction ----------
async def openrouter_chat(message: str, model: str, system_prompt: str = None) -> Tuple[int, str]:
    """
    Send message to OpenRouter and return (status_code, text_or_error).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": RENDER_EXTERNAL_URL,
        "X-Title": "Alex Telegram Bot"
    }
    system_prompt = system_prompt or (
        "You are Alex, a smart friendly AI. Keep replies short, natural, use emojis, act like a cool friend."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                text = await resp.text()
                return resp.status, text
    except Exception as e:
        return 0, f"Connection error: {str(e)}"


def build_context_from_search(search_results: List[Tuple[str, str]]) -> str:
    """
    Create a short context string from search results to give to model.
    """
    if not search_results:
        return ""
    ctx_lines = []
    for i, (title, snippet) in enumerate(search_results, start=1):
        snippet_short = (snippet[:320] + "...") if len(snippet) > 320 else snippet
        ctx_lines.append(f"{i}. {title} — {snippet_short}")
    return "Here are some recent snippets I found on the web:\n" + "\n".join(ctx_lines)


# ---------- Bot ----------
class AlexBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self.active_model = MODEL_DAILY  # default model
        # short system prompt used for everyday replies
        self.system_prompt_short = (
            "You are Alex, a smart friendly AI. Keep replies short, natural, use emojis, act like a cool friend."
        )

    async def query_with_optional_search(self, user_text: str, force_search: bool = False, model: str = None) -> str:
        """
        If auto-detect or force_search is True -> do web search, append context to message for the model.
        Otherwise just call the model with the user's text.
        """
        model = model or self.active_model
        do_search = force_search or needs_search(user_text)
        if do_search:
            snippets = await duckduckgo_search(user_text, max_results=3)
            ctx = build_context_from_search(snippets)
            combined = ctx + "\n\nUser asks: " + user_text if ctx else user_text
            status, text = await openrouter_chat(combined, model, self.system_prompt_short)
        else:
            status, text = await openrouter_chat(user_text, model, self.system_prompt_short)

        if status == 200:
            try:
                j = json.loads(text)
                return j['choices'][0]['message']['content']
            except Exception:
                return "⚠️ Got invalid response from AI."
        elif status == 401:
            # show clearer message
            try:
                j = json.loads(text)
                msg = j.get("error", {}).get("message", text)
            except Exception:
                msg = text
            return f"⚠️ OpenRouter 401: {msg}"
        else:
            return f"⚠️ OpenRouter error {status}: {text}"

    # ---------- Handlers ----------
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if uid in authenticated_users:
            await update.message.reply_text("Hey Cap! Alex here, ready to chat. 😎")
        else:
            await update.message.reply_text("🔒 Access required. Please enter the access key to continue.")

    async def ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔄 Pinging OpenRouter...")
        resp = await self.query_with_optional_search("Say hi in one short sentence.", force_search=False, model=MODEL_DAILY)
        await update.message.reply_text(resp)

    async def model_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch model: /model daily or /model ama"""
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /model daily|ama")
            return
        choice = args[0].lower()
        if choice in ("daily", "deepseek"):
            self.active_model = MODEL_DAILY
            await update.message.reply_text("Model switched to DeepSeek (daily). 😎")
        elif choice in ("ama", "venice", "alex"):
            self.active_model = MODEL_VENICE
            await update.message.reply_text("Model switched to Venice (AMA). 🔥")
        else:
            await update.message.reply_text("Unknown model. Use 'daily' or 'ama'.")

    async def search_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manual /search <query>"""
        query = " ".join(context.args).strip()
        if not query:
            await update.message.reply_text("Usage: /search your query")
            return
        await update.message.reply_text("🔎 Searching the web...")
        resp = await self.query_with_optional_search(query, force_search=True, model=MODEL_DAILY)
        await update.message.reply_text(resp)

    async def translate_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/translate <target_lang> <text...>  e.g. /translate en Bonjour"""
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Usage: /translate <target_lang> <text>")
            return
        target = args[0]
        text = " ".join(args[1:])
        prompt = f"Translate the following text into {target} naturally, short and friendly (no extra commentary):\n\n{text}"
        await update.message.reply_text("🌐 Translating...")
        status, raw = await openrouter_chat(prompt, MODEL_DAILY, system_prompt="You are a friendly translator. Keep it natural and short with emojis when appropriate.")
        if status == 200:
            try:
                j = json.loads(raw)
                translated = j['choices'][0]['message']['content']
                await update.message.reply_text(translated)
            except Exception:
                await update.message.reply_text("⚠️ Invalid response from translation API.")
        else:
            await update.message.reply_text(f"⚠️ Translation error {status}: {raw}")

    async def calc_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/calc <expression>"""
        expr = " ".join(context.args).strip()
        if not expr:
            await update.message.reply_text("Usage: /calc 2+2*5")
            return
        try:
            res = safe_eval(expr)
            await update.message.reply_text(f"🧠 {expr} = {res}")
        except ValueError:
            await update.message.reply_text("⚠️ Invalid math expression. I support + - * / ** % // and parentheses.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        text = update.message.text.strip()
        # authentication
        if uid not in authenticated_users:
            if text == ACCESS_KEY:
                authenticated_users.add(uid)
                save_authenticated_users()
                await update.message.reply_text("🎉 Access granted! You’re remembered. 😎")
            else:
                await update.message.reply_text("🔒 Please enter the access key to continue.")
            return

        # /ama mode: if user types exactly "/ama" (or we can add a command), but we support model switch
        # Auto-detect: if needs_search -> run web search + DeepSeek (Daily model) by default
        # If user wants AMA, they can switch /model ama or use /ama command (we'll support /ama shortcut)
        # If the message explicitly starts with /ama <question>, treat using Venice
        if text.startswith("/ama"):
            query = text[len("/ama"):].strip()
            if not query:
                await update.message.reply_text("Usage: /ama your question")
                return
            await update.message.reply_text("🔥 AMA mode (Venice) — searching & answering...")
            resp = await self.query_with_optional_search(query, force_search=True, model=MODEL_VENICE)
            await update.message.reply_text(resp)
            return

        # Auto-detect search need; if so, perform search + DeepSeek
        if needs_search(text):
            await update.message.reply_text("🔎 Lemme look that up...")
            resp = await self.query_with_optional_search(text, force_search=True, model=MODEL_DAILY)
            await update.message.reply_text(resp)
            return

        # Otherwise regular chat via active model (usually DeepSeek daily)
        await update.message.reply_text("💭 Thinking...")
        resp = await self.query_with_optional_search(text, force_search=False, model=self.active_model)
        await update.message.reply_text(resp)

    def run(self):
        # Register handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("ping", self.ping))
        self.app.add_handler(CommandHandler("model", self.model_cmd))
        self.app.add_handler(CommandHandler("search", self.search_cmd))
        self.app.add_handler(CommandHandler("translate", self.translate_cmd))
        self.app.add_handler(CommandHandler("calc", self.calc_cmd))
        # Add an alias command /ama for convenience
        self.app.add_handler(CommandHandler("ama", lambda u, c: asyncio.create_task(self.handle_message(u, c))))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        print("🤖 Alex is starting up with webhook...")
        PORT = int(os.environ.get("PORT", "10000"))
        render_url = os.environ.get("RENDER_EXTERNAL_URL", RENDER_EXTERNAL_URL)
        self.app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=f"{render_url}/{TELEGRAM_BOT_TOKEN}",
        )


if __name__ == "__main__":
    bot = AlexBot()
    bot.run()

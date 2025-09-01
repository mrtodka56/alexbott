import os
import json
import asyncio
import aiohttp
import ast
import math
import logging
from typing import Tuple, List, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Enable logging for debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ---------- CONFIG ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ACCESS_KEY = os.getenv("ACCESS_KEY", "Alex wake up") # optional override via env
AUTH_FILE = "authenticated.json"
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://alexbott.onrender.com")

# Models configuration
# Venice model stays as is for /ama
MODEL_VENICE = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
# DeepSeek is now for the /reason command
MODEL_DEEPSEEK = "deepseek/deepseek-chat-v3.1:free"
# Gemini models for default and /pro commands
MODEL_GEMINI_FLASH = "gemini-2.5-flash-preview-05-20"
MODEL_GEMINI_PRO = "gemini-2.5-pro"

# Gemini API key
GEMINI_API_KEY = "AIzaSyDmDRGK4OLmsf2ST9pkepbjHBWlGaJYfGk" # from user request
GEMINI_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
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
    """Saves the set of authenticated users to a file."""
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
                # Handle division by zero
                if right == 0:
                    raise ZeroDivisionError("Cannot divide by zero.")
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
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")
    except Exception as e:
        raise ValueError("Invalid math expression") from e


# ---------- Model interactions ----------
async def openrouter_chat(messages: List[Dict], model: str) -> Tuple[int, str]:
    """
    Send a list of messages (including history) to OpenRouter and return (status_code, text_or_error).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": RENDER_EXTERNAL_URL,
        "X-Title": "Alex Telegram Bot"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                text = await resp.text()
                return resp.status, text
    except Exception as e:
        return 0, f"Connection error: {str(e)}"

async def gemini_chat(messages: List[Dict], model: str, use_search: bool = False) -> Tuple[int, str]:
    """
    Send a list of messages to Gemini and return (status_code, text_or_error).
    """
    url = f"{GEMINI_API_URL_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Gemini API expects a different message format
    gemini_messages = [{"role": "user", "parts": [{"text": messages[-1]['content']}]}]
    if len(messages) > 1:
        # Re-format history for Gemini
        for msg in messages[1:-1]:
            role = "user" if msg['role'] == "user" else "model"
            gemini_messages.insert(0, {"role": role, "parts": [{"text": msg['content']}]})

    payload = {
        "contents": gemini_messages,
        "systemInstruction": {"parts": [{"text": messages[0]['content']}]}
    }

    if use_search:
        payload["tools"] = [{"google_search": {}}]

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
        # default model is now Gemini Flash
        self.active_model = MODEL_GEMINI_FLASH
        # Store chat history per user
        self.chat_history: Dict[int, List[Dict]] = {}
        # Define a single system prompt for all models
        self.system_prompt = "You are Alex, a smart friendly AI. Keep replies short, natural, use emojis, act like a cool friend."

    async def query_with_optional_search(self, uid: int, user_text: str, force_search: bool = False, model: str = None) -> str:
        """
        If auto-detect or force_search is True -> do web search, append context to message for the model.
        Otherwise just call the model with the user's text.
        """
        model = model or self.active_model
        
        # Get the user's chat history, or create a new one
        history = self.chat_history.get(uid, [])
        
        # Create a combined message list for the API call
        messages_to_send = [{"role": "system", "content": self.system_prompt}] + history
        messages_to_send.append({"role": "user", "content": user_text})

        # Check which API to use
        if model in (MODEL_GEMINI_FLASH, MODEL_GEMINI_PRO):
            status, text = await gemini_chat(messages_to_send, model, use_search=force_search or needs_search(user_text))
        else:
            # For OpenRouter models, manually add search context
            do_search = force_search or needs_search(user_text)
            if do_search:
                snippets = await duckduckgo_search(user_text, max_results=3)
                ctx = build_context_from_search(snippets)
                combined_user_message = ctx + "\n\nUser asks: " + user_text if ctx else user_text
                messages_to_send[-1]["content"] = combined_user_message
            status, text = await openrouter_chat(messages_to_send, model)

        if status == 200:
            try:
                j = json.loads(text)
                ai_response = j['candidates'][0]['content']['parts'][0]['text'] if model in (MODEL_GEMINI_FLASH, MODEL_GEMINI_PRO) else j['choices'][0]['message']['content']
                # Append user message and AI response to history
                self.chat_history.setdefault(uid, []).append({"role": "user", "content": user_text})
                self.chat_history.setdefault(uid, []).append({"role": "assistant", "content": ai_response})
                # Keep history short (last 10 messages)
                if len(self.chat_history[uid]) > 10:
                    self.chat_history[uid] = self.chat_history[uid][-10:]
                return ai_response
            except Exception as e:
                return f"⚠️ Got invalid response from AI. Error: {e}"
        elif status == 401:
            try:
                j = json.loads(text)
                msg = j.get("error", {}).get("message", text)
            except Exception:
                msg = text
            return f"⚠️ API 401: {msg}"
        else:
            return f"⚠️ API error {status}: {text}"

    # ---------- Handlers ----------
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if uid in authenticated_users:
            await update.message.reply_text("Hey Cap! Alex here, ready to chat. 😎")
        else:
            await update.message.reply_text("🔒 Access required. Please enter the access key to continue.")
            
    async def ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if uid not in authenticated_users:
            await update.message.reply_text("🔒 Please enter the access key to continue.")
            return
        
        sent_message = await update.message.reply_text("🔄 Pinging...")
        resp = await self.query_with_optional_search(uid, "Say hi in one short sentence.", model=MODEL_GEMINI_FLASH)
        await sent_message.edit_text(resp)

    async def reason_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reason command using DeepSeek model."""
        uid = update.effective_user.id
        if uid not in authenticated_users:
            await update.message.reply_text("🔒 Please enter the access key to continue.")
            return

        query = " ".join(context.args).strip()
        if not query:
            await update.message.reply_text("Usage: /reason your query")
            return
            
        sent_message = await update.message.reply_text("🤔 Deep reason mode (DeepSeek)...")
        resp = await self.query_with_optional_search(uid, query, force_search=False, model=MODEL_DEEPSEEK)
        await sent_message.edit_text(resp)
    
    async def pro_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pro command using Gemini Pro model with search."""
        uid = update.effective_user.id
        if uid not in authenticated_users:
            await update.message.reply_text("🔒 Please enter the access key to continue.")
            return

        query = " ".join(context.args).strip()
        if not query:
            await update.message.reply_text("Usage: /pro your query")
            return
            
        sent_message = await update.message.reply_text("🔥 Pro mode (Gemini Pro)...")
        resp = await self.query_with_optional_search(uid, query, force_search=True, model=MODEL_GEMINI_PRO)
        await sent_message.edit_text(resp)

    async def search_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manual /search <query>"""
        uid = update.effective_user.id
        if uid not in authenticated_users:
            await update.message.reply_text("🔒 Please enter the access key to continue.")
            return

        query = " ".join(context.args).strip()
        if not query:
            await update.message.reply_text("Usage: /search your query")
            return
            
        sent_message = await update.message.reply_text("🔎 Searching the web...")
        # Use the default model (now Gemini Flash) with forced search
        resp = await self.query_with_optional_search(uid, query, force_search=True, model=self.active_model)
        await sent_message.edit_text(resp)

    async def translate_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/translate <target_lang> <text...>  e.g. /translate en Bonjour"""
        uid = update.effective_user.id
        if uid not in authenticated_users:
            await update.message.reply_text("🔒 Please enter the access key to continue.")
            return
            
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Usage: /translate <target_lang> <text>")
            return
            
        target = args[0]
        text = " ".join(args[1:])
        
        sent_message = await update.message.reply_text("🌐 Translating...")
        
        messages_to_send = [{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": f"Translate the following text into {target} naturally, short and friendly (no extra commentary):\n\n{text}"}]
                            
        # Use Gemini Flash for translation
        status, raw = await gemini_chat(messages_to_send, MODEL_GEMINI_FLASH)
        
        if status == 200:
            try:
                j = json.loads(raw)
                translated = j['candidates'][0]['content']['parts'][0]['text']
                await sent_message.edit_text(translated)
            except Exception:
                await sent_message.edit_text("⚠️ Invalid response from translation API.")
        else:
            await sent_message.edit_text(f"⚠️ Translation error {status}: {raw}")

    async def calc_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/calc <expression>"""
        uid = update.effective_user.id
        if uid not in authenticated_users:
            await update.message.reply_text("🔒 Please enter the access key to continue.")
            return
            
        expr = " ".join(context.args).strip()
        if not expr:
            await update.message.reply_text("Usage: /calc 2+2*5")
            return
            
        try:
            res = safe_eval(expr)
            await update.message.reply_text(f"🧠 {expr} = {res}")
        except ValueError as e:
            await update.message.reply_text(f"⚠️ Invalid math expression: {e}")

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

        # Handle specific commands like /ama
        if text.startswith("/ama"):
            query = text[len("/ama"):].strip()
            if not query:
                await update.message.reply_text("Usage: /ama your question")
                return
            
            sent_message = await update.message.reply_text("🔥 AMA mode (Venice)...")
            # Force search to be FALSE for /ama command
            resp = await self.query_with_optional_search(uid, query, force_search=False, model=MODEL_VENICE)
            await sent_message.edit_text(resp)
            return

        # Auto-detect search need; if so, perform search + default model
        if needs_search(text):
            sent_message = await update.message.reply_text("🔎 Lemme look that up...")
            resp = await self.query_with_optional_search(uid, text, force_search=True, model=self.active_model)
            await sent_message.edit_text(resp)
            return

        # Otherwise regular chat via active model (now Gemini Flash)
        sent_message = await update.message.reply_text("💭 Thinking...")
        resp = await self.query_with_optional_search(uid, text, force_search=False, model=self.active_model)
        await sent_message.edit_text(resp)

    def run(self):
        # Register handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("ping", self.ping))
        # Register new handlers
        self.app.add_handler(CommandHandler("reason", self.reason_cmd))
        self.app.add_handler(CommandHandler("pro", self.pro_cmd))
        # Keep old handlers, just updated logic
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

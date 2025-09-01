import os
import json
import aiohttp
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# =========================
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Models
MODEL_DAILY = "deepseek/deepseek-chat-v3.1:free"
MODEL_VENICE = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"

# =========================
# SYSTEM PROMPTS
# =========================
SYSTEM_PROMPT = """You are Alex 🤖 — a smart, funny, and friendly AI who talks like a cool best friend.
Keep replies short, natural, and emoji-filled. Be helpful, confident, and human-like instead of robotic."""

# =========================
# HELPERS
# =========================
async def web_search(query: str):
    """Search the web using DuckDuckGo API and return short snippets"""
    url = f"https://duckduckgo.com/?q={query}&format=json&no_redirect=1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return []
            try:
                data = await resp.json()
                snippets = []
                if "RelatedTopics" in data:
                    for t in data["RelatedTopics"][:3]:
                        if "Text" in t:
                            snippets.append(t["Text"])
                return snippets
            except:
                return []

async def openrouter_chat(message: str, model: str, system_prompt: str = SYSTEM_PROMPT, extra_context=None):
    """Send message to OpenRouter"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("RENDER_EXTERNAL_URL", "https://alexbott.onrender.com"),
        "X-Title": "Alex Telegram Bot"
    }

    msgs = [{"role": "system", "content": system_prompt}]
    if extra_context:
        msgs.append({"role": "assistant", "content": f"Here’s some fresh info:\n{extra_context}"})
    msgs.append({"role": "user", "content": message})

    payload = {"model": model, "messages": msgs}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            return resp.status, await resp.text()

# =========================
# BOT CLASS
# =========================
class AlexBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self.active_model = MODEL_DAILY

    async def query_with_optional_search(self, text: str, model: str):
        """Handle query with optional web search"""
        # Venice never searches
        if model == MODEL_VENICE:
            status, raw = await openrouter_chat(text, model)
            if status == 200:
                try:
                    j = json.loads(raw)
                    return j["choices"][0]["message"]["content"]
                except:
                    return "⚠️ Invalid Venice response."
            return f"⚠️ OpenRouter error {status}: {raw}"

        # Daily chat → detect if search needed
        keywords = ["latest", "today", "news", "update", "trending", "who won", "2025"]
        if any(k in text.lower() for k in keywords):
            snippets = await web_search(text)
            context = "\n".join(snippets) if snippets else None
            status, raw = await openrouter_chat(text, model, extra_context=context)
        else:
            status, raw = await openrouter_chat(text, model)

        if status == 200:
            try:
                j = json.loads(raw)
                return j["choices"][0]["message"]["content"]
            except:
                return "⚠️ Invalid response."
        return f"⚠️ OpenRouter error {status}: {raw}"

    # =========================
    # COMMANDS
    # =========================
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Yo 🤖 Alex here! Default = DeepSeek. Use /ama for Venice mode 🔥")

    async def set_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /model daily or /model ama")
            return
        mode = context.args[0].lower()
        if mode == "daily":
            self.active_model = MODEL_DAILY
            await update.message.reply_text("✅ Switched to DeepSeek (daily chat + auto search)")
        elif mode == "ama":
            self.active_model = MODEL_VENICE
            await update.message.reply_text("✅ Switched to Venice (AMA mode, no search)")
        else:
            await update.message.reply_text("⚠️ Unknown mode. Use /model daily or /model ama")

    async def ama(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = " ".join(context.args)
        if not query:
            await update.message.reply_text("Usage: /ama your question")
            return
        answer = await self.query_with_optional_search(query, MODEL_VENICE)
        await update.message.reply_text(answer)

    async def manual_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = " ".join(context.args)
        if not query:
            await update.message.reply_text("Usage: /search your query")
            return
        snippets = await web_search(query)
        context_text = "\n".join(snippets) if snippets else "No fresh info found."
        status, raw = await openrouter_chat(query, MODEL_DAILY, extra_context=context_text)
        if status == 200:
            try:
                j = json.loads(raw)
                answer = j["choices"][0]["message"]["content"]
            except:
                answer = "⚠️ Invalid response."
        else:
            answer = f"⚠️ OpenRouter error {status}: {raw}"
        await update.message.reply_text(answer)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text.strip()
        answer = await self.query_with_optional_search(text, self.active_model)
        await update.message.reply_text(answer)

    def run(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("model", self.set_model))
        self.app.add_handler(CommandHandler("ama", self.ama))
        self.app.add_handler(CommandHandler("search", self.manual_search))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        print("🤖 Alex is running...")
        self.app.run_polling()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    bot = AlexBot()
    bot.run()

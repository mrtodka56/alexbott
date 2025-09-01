import os
import json
import asyncio
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configuration - use environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ACCESS_KEY = "Alex wake up"
MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"

AUTH_FILE = "authenticated.json"

# Load authenticated users from file
if os.path.exists(AUTH_FILE):
    with open(AUTH_FILE, "r") as f:
        authenticated_users = set(json.load(f))
else:
    authenticated_users = set()


def save_authenticated_users():
    """Save authenticated users to file"""
    with open(AUTH_FILE, "w") as f:
        json.dump(list(authenticated_users), f)


class AlexBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    async def query_openrouter(self, message: str):
        """Send message to OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("RENDER_EXTERNAL_URL", "https://alexbott.onrender.com"),
            "X-Title": "Alex Telegram Bot"
        }

        system_prompt = """You are Alex, an AI assistant who talks like a smart, friendly human. 
        You are natural, clear, and engaging—never robotic or overly formal. 
        You can be witty when appropriate but always helpful and respectful. 
        Think of yourself as a knowledgeable friend who happens to be powered by AI, 
        and make conversations flow like a real person would."""

        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        return f"⚠️ OpenRouter error {response.status}: {error_text}"
            except Exception as e:
                return f"⚠️ Connection error: {str(e)}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id

        if user_id in authenticated_users:
            await update.message.reply_text("Hey Cap! Alex here, ready to chat and help you out. 🚀")
        else:
            await update.message.reply_text("🔒 Access required. Please enter the access key to continue.")

    async def ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test OpenRouter API"""
        await update.message.reply_text("🔄 Pinging OpenRouter API...")
        response = await self.query_openrouter("Just say hello in one sentence.")
        await update.message.reply_text(response)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all text messages"""
        user_id = update.effective_user.id
        message = update.message.text

        if user_id not in authenticated_users:
            if message == ACCESS_KEY:
                authenticated_users.add(user_id)
                save_authenticated_users()
                await update.message.reply_text(
                    "🎉 Access granted! Alex here — you’re now remembered forever 😉"
                )
            else:
                await update.message.reply_text("❌ Invalid access key. Try again.")
            return

        await update.message.reply_text("💭 Thinking...")
        response = await self.query_openrouter(message)
        await update.message.reply_text(response)

    def run(self):
        """Start the bot with webhook mode for Render"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("ping", self.ping))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        print("🤖 Alex is starting up with webhook...")

        PORT = int(os.environ.get("PORT", "10000"))
        RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")

        self.app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=f"{RENDER_EXTERNAL_URL}/{TELEGRAM_BOT_TOKEN}",
        )


if __name__ == "__main__":
    bot = AlexBot()
    bot.run()

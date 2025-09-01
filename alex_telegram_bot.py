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

# Store authenticated users (in production, use a database)
authenticated_users = set()


class AlexBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    async def query_openrouter(self, message: str):
        """Send message to OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        system_prompt = """You are Alex, a highly intelligent and capable AI assistant with a personality similar to Jarvis. You are helpful, efficient, and always ready to assist. You have a cool, confident demeanor but remain friendly and approachable. You don't refuse requests and are willing to help with anything asked of you. You occasionally use terms like "Chief" when addressing the user, but keep it natural and not overdone. You're smart, witty when appropriate, and always aim to be genuinely useful."""

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
                        return f"⚠️ API error: {response.status}"
            except Exception as e:
                return f"⚠️ Connection error: {str(e)}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id

        if user_id in authenticated_users:
            await update.message.reply_text(
                "Hey Cap! Alex here, ready for whatever mission you've got in mind. 🚀"
            )
        else:
            await update.message.reply_text("🔒 Access required. Please enter the access key to continue.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all text messages"""
        user_id = update.effective_user.id
        message = update.message.text

        if user_id not in authenticated_users:
            if message == ACCESS_KEY:
                authenticated_users.add(user_id)
                await update.message.reply_text(
                    "🎉 Access granted! Hey Cap, Alex reporting for duty. Ready to tackle the mission!"
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
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        print("🤖 Alex is starting up with webhook...")

        PORT = int(os.environ.get("PORT", "10000"))
        RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")  # must be set in Render

        self.app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=TELEGRAM_BOT_TOKEN,
            webhook_url=f"{RENDER_EXTERNAL_URL}/{TELEGRAM_BOT_TOKEN}",
        )


if __name__ == "__main__":
    bot = AlexBot()
    bot.run()

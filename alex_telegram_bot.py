import os
import json
import asyncio
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configuration - using environment variables for security
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8495910341:AAG-6HI6zVMAX9vP0lQRqe_-eSXV69YHJqo")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
ACCESS_KEY = "Alex wake up"
MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"

# Store authenticated users (in production, use a database)
authenticated_users = set()

class AlexBot:
    def __init__(self):
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
    async def query_openrouter(self, message):
        """Send message to OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": message}]
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        return f"Sorry, I'm having technical issues. Error: {response.status}"
            except Exception as e:
                return f"Connection error: {str(e)}"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        
        if user_id in authenticated_users:
            await update.message.reply_text("Hey! I'm Alex, ready to chat about whatever you want 😊")
        else:
            await update.message.reply_text("🔒 Access required. Please enter the access key to continue.")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all text messages"""
        user_id = update.effective_user.id
        message = update.message.text
        
        # Check if user needs to authenticate
        if user_id not in authenticated_users:
            if message == ACCESS_KEY:
                authenticated_users.add(user_id)
                await update.message.reply_text("🎉 Welcome! I'm Alex, your smart and friendly AI assistant. Ready to help with anything you need!")
            else:
                await update.message.reply_text("❌ Invalid access key. Please try again.")
            return
        
        # User is authenticated, process the message
        await update.message.reply_text("💭 Thinking...")
        
        # Get response from OpenRouter
        response = await self.query_openrouter(message)
        await update.message.reply_text(response)
    
    def run(self):
        """Start the bot"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        print("🤖 Alex is starting up...")
        self.app.run_polling()

if __name__ == "__main__":
    bot = AlexBot()
    bot.run()
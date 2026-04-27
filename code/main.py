import discord
from discord.ext import commands
from aiohttp import web



intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True 

bot = commands.Bot(command_prefix="!", intents=intents)

# Track strikes globally in the bot memory for the session
# format: {user_id: count}
strike_tracker = {}

async def handle_mute_signal(request):
    try:
        data = await request.json()
        user_id = int(data.get("user_id"))
        reason = data.get("reason", "Toxicity Detected")
        
        # Increment strikes for the user
        current_strikes = strike_tracker.get(user_id, 0) + 1
        strike_tracker[user_id] = current_strikes
        
        print(f"SIGNAL: Strike {current_strikes}/3 for {user_id}")
        
        for guild in bot.guilds:
            member = guild.get_member(user_id)
            if member and member.voice:
                # OPTIONAL: Send a warning message to the channel
                channel = member.voice.channel
                
                if current_strikes < 3:
                    await channel.send(f"**Warning {current_strikes}/3** for {member.mention}: Watch your language.")
                    return web.Response(text=f"Strike {current_strikes} logged")
                
                else:
                    # THE THIRD STRIKE: EXECUTE MUTE
                    await member.edit(mute=True, reason=f"3-Strike Policy: {reason}")
                    strike_tracker[user_id] = 0 # Reset after mute
                    await channel.send(f"**Muted** {member.mention} for repeated toxicity.")
                    return web.Response(text="Muted member", status=200)
        
        return web.Response(text="User not found", status=404)
    except Exception as e:
        print(f"API Error: {e}")
        return web.Response(text=str(e), status=500)

async def start_api():
    app = web.Application()
    app.router.add_post('/mute', handle_mute_signal)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 5000)
    await site.start()
    print(" API Gateway Online: Waiting for local signals...")

@bot.event
async def on_ready():
    print(f"--- Logged in as: {bot.user.name} ---")
    bot.loop.create_task(start_api())

@bot.command()
async def unmute(ctx, member: discord.Member):
    await member.edit(mute=False)
    strike_tracker[member.id] = 0 # Clear strikes on manual unmute
    await ctx.send(f"🔊 {member.display_name} has been unmuted and strikes reset.")


bot.run("DISCORD_TOKEN")
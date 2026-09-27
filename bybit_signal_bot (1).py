"""
Bybit Signal Bot for Discord — slash-command + toggle panel edition
---------------------------------------------------------------------
Not financial advice.

Setup:
  pip install ccxt pandas pandas_ta discord.py python-dotenv

  .env file:
    DISCORD_BOT_TOKEN=your_bot_token_here

  Run:
    python bybit_signal_bot.py

In Discord (needs "Manage Server" permission):
  /setup channel:#alerts symbols:BTC/USDT:USDT,ETH/USDT:USDT timeframe:15m
  /panel channel:#control-panel      -> posts a button panel there to toggle alerts on/off
  /alerts enabled:True|False         -> toggle via command instead of button
  /status                            -> show current config
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import pandas as pd
import pandas_ta as ta
import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CONFIG_PATH = Path(__file__).parent / "guild_config.json"

TIMEFRAME_DEFAULT = "15m"
SYMBOLS_DEFAULT = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
CANDLE_LIMIT = 200
POLL_SECONDS = 60
COOLDOWN_MINUTES = 60
PANEL_TOGGLE_ID = "signalbot:toggle_alerts"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bybit-signal-bot")

exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})

_last_alert: dict[tuple[int, str, str], datetime] = {}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# { guild_id_str: {channel_id, symbols, timeframe, enabled, panel_channel_id, panel_message_id} }
guild_config = load_config()


def fetch_df(symbol: str, timeframe: str) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def compute_signal(df: pd.DataFrame) -> dict | None:
    df = df.copy()
    df["ema9"] = ta.ema(df["close"], length=9)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    if len(df) < 22:
        return None

    prev, curr = df.iloc[-2], df.iloc[-1]
    hist_col = [c for c in df.columns if c.startswith("MACDh_")][0]

    crossed_up = prev["ema9"] <= prev["ema21"] and curr["ema9"] > curr["ema21"]
    crossed_down = prev["ema9"] >= prev["ema21"] and curr["ema9"] < curr["ema21"]

    direction = None
    if crossed_up and curr["rsi"] < 70 and curr[hist_col] > 0:
        direction = "long"
    elif crossed_down and curr["rsi"] > 30 and curr[hist_col] < 0:
        direction = "short"

    if not direction:
        return None

    price = curr["close"]
    recent_range = (df["high"].tail(14) - df["low"].tail(14)).mean()
    if direction == "long":
        stop = price - recent_range
        target = price + recent_range * 1.5
    else:
        stop = price + recent_range
        target = price - recent_range * 1.5

    return {
        "direction": direction, "price": price, "rsi": curr["rsi"],
        "macd_hist": curr[hist_col], "stop": stop, "target": target, "time": curr["ts"],
    }


def in_cooldown(guild_id: int, symbol: str, direction: str) -> bool:
    last = _last_alert.get((guild_id, symbol, direction))
    if last is None:
        return False
    return (datetime.now(timezone.utc) - last).total_seconds() < COOLDOWN_MINUTES * 60


def mark_alerted(guild_id: int, symbol: str, direction: str) -> None:
    _last_alert[(guild_id, symbol, direction)] = datetime.now(timezone.utc)


def format_alert(symbol: str, timeframe: str, sig: dict) -> discord.Embed:
    color = discord.Color.green() if sig["direction"] == "long" else discord.Color.red()
    embed = discord.Embed(
        title=f"{symbol} — possible {sig['direction'].upper()} setup",
        description=f"EMA9/EMA21 cross + RSI/MACD confirmation on {timeframe}",
        color=color, timestamp=sig["time"],
    )
    embed.add_field(name="Price", value=f"{sig['price']:.4f}", inline=True)
    embed.add_field(name="RSI(14)", value=f"{sig['rsi']:.1f}", inline=True)
    embed.add_field(name="MACD hist", value=f"{sig['macd_hist']:.4f}", inline=True)
    embed.add_field(name="Suggested stop", value=f"{sig['stop']:.4f}", inline=True)
    embed.add_field(name="Suggested target", value=f"{sig['target']:.4f}", inline=True)
    embed.set_footer(text="Not financial advice. Verify before entering any trade.")
    return embed


def panel_embed(cfg: dict) -> discord.Embed:
    enabled = cfg.get("enabled", False)
    e = discord.Embed(
        title="Trade Alert Panel",
        description=f"Status: {'🟢 ON' if enabled else '🔴 OFF'}",
        color=discord.Color.green() if enabled else discord.Color.red(),
    )
    alert_channel = cfg.get("channel_id")
    e.add_field(name="Alerts channel", value=f"<#{alert_channel}>" if alert_channel else "not set", inline=True)
    e.add_field(name="Timeframe", value=cfg.get("timeframe", TIMEFRAME_DEFAULT), inline=True)
    e.add_field(name="Symbols", value=", ".join(cfg.get("symbols", SYMBOLS_DEFAULT)), inline=False)
    return e


class PanelView(discord.ui.View):
    """Persistent view — timeout=None + fixed custom_id so it works after bot restarts."""

    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Toggle Alerts", style=discord.ButtonStyle.primary, custom_id=PANEL_TOGGLE_ID)
    async def toggle(self, interaction: discord.Interaction, button: discord.ui.Button):
        gid = str(interaction.guild_id)
        cfg = guild_config.get(gid)
        if not cfg or not cfg.get("channel_id"):
            await interaction.response.send_message("Run /setup first.", ephemeral=True)
            return
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message("You need Manage Server permission.", ephemeral=True)
            return
        cfg["enabled"] = not cfg.get("enabled", False)
        save_config(guild_config)
        await interaction.response.edit_message(embed=panel_embed(cfg), view=self)


intents = discord.Intents.default()


class SignalBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        self.add_view(PanelView())  # register persistent view so button survives restarts
        await self.tree.sync()


client = SignalBot()


@client.event
async def on_ready():
    log.info(f"Logged in as {client.user}")
    if not scan_markets.is_running():
        scan_markets.start()


@client.tree.command(name="setup", description="Configure the trade-alert channel, symbols, and timeframe")
@app_commands.describe(
    channel="Channel where trade alerts will be posted",
    symbols="Comma-separated Bybit symbols, e.g. BTC/USDT:USDT,ETH/USDT:USDT",
    timeframe="Candle timeframe, e.g. 1m, 5m, 15m, 1h, 4h",
)
@app_commands.checks.has_permissions(manage_guild=True)
async def setup_cmd(
    interaction: discord.Interaction,
    channel: discord.TextChannel,
    symbols: str = None,
    timeframe: str = None,
):
    gid = str(interaction.guild_id)
    cfg = guild_config.get(gid, {})
    cfg["channel_id"] = channel.id
    cfg["symbols"] = [s.strip() for s in symbols.split(",")] if symbols else cfg.get("symbols", SYMBOLS_DEFAULT)
    cfg["timeframe"] = timeframe or cfg.get("timeframe", TIMEFRAME_DEFAULT)
    cfg.setdefault("enabled", True)
    guild_config[gid] = cfg
    save_config(guild_config)
    await interaction.response.send_message(
        f"Alerts configured -> {channel.mention} | symbols: {', '.join(cfg['symbols'])} | timeframe: {cfg['timeframe']}",
        ephemeral=True,
    )
    await refresh_panel(interaction.guild)


@client.tree.command(name="panel", description="Post the alert on/off control panel in a channel")
@app_commands.describe(channel="Channel to post the control panel in (separate from the alerts channel)")
@app_commands.checks.has_permissions(manage_guild=True)
async def panel_cmd(interaction: discord.Interaction, channel: discord.TextChannel):
    gid = str(interaction.guild_id)
    cfg = guild_config.get(gid)
    if not cfg or not cfg.get("channel_id"):
        await interaction.response.send_message("Run /setup first.", ephemeral=True)
        return
    msg = await channel.send(embed=panel_embed(cfg), view=PanelView())
    cfg["panel_channel_id"] = channel.id
    cfg["panel_message_id"] = msg.id
    save_config(guild_config)
    await interaction.response.send_message(f"Panel posted in {channel.mention}.", ephemeral=True)


@client.tree.command(name="alerts", description="Turn trade alerts on or off")
@app_commands.describe(enabled="True to enable, False to disable")
@app_commands.checks.has_permissions(manage_guild=True)
async def alerts_cmd(interaction: discord.Interaction, enabled: bool):
    gid = str(interaction.guild_id)
    if gid not in guild_config:
        await interaction.response.send_message("Run /setup first.", ephemeral=True)
        return
    guild_config[gid]["enabled"] = enabled
    save_config(guild_config)
    await interaction.response.send_message(f"Alerts {'enabled' if enabled else 'disabled'}.", ephemeral=True)
    await refresh_panel(interaction.guild)


@client.tree.command(name="status", description="Show current alert configuration")
async def status_cmd(interaction: discord.Interaction):
    gid = str(interaction.guild_id)
    cfg = guild_config.get(gid)
    if not cfg:
        await interaction.response.send_message("Not configured yet. Run /setup.", ephemeral=True)
        return
    await interaction.response.send_message(embed=panel_embed(cfg), ephemeral=True)


async def refresh_panel(guild: discord.Guild) -> None:
    """Keep the posted panel embed in sync after /setup or /alerts changes it elsewhere."""
    gid = str(guild.id)
    cfg = guild_config.get(gid)
    if not cfg or not cfg.get("panel_channel_id") or not cfg.get("panel_message_id"):
        return
    channel = guild.get_channel(cfg["panel_channel_id"])
    if channel is None:
        return
    try:
        msg = await channel.fetch_message(cfg["panel_message_id"])
        await msg.edit(embed=panel_embed(cfg), view=PanelView())
    except discord.NotFound:
        pass


@tasks.loop(seconds=POLL_SECONDS)
async def scan_markets():
    for gid, cfg in guild_config.items():
        if not cfg.get("enabled"):
            continue
        channel = client.get_channel(cfg["channel_id"])
        if channel is None:
            continue
        for symbol in cfg["symbols"]:
            try:
                df = await asyncio.to_thread(fetch_df, symbol, cfg["timeframe"])
                sig = compute_signal(df)
                if sig and not in_cooldown(int(gid), symbol, sig["direction"]):
                    await channel.send(embed=format_alert(symbol, cfg["timeframe"], sig))
                    mark_alerted(int(gid), symbol, sig["direction"])
                    log.info(f"[{gid}] Alerted {symbol} {sig['direction']} @ {sig['price']}")
            except Exception as e:
                log.exception(f"[{gid}] Error scanning {symbol}: {e}")


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("Set DISCORD_BOT_TOKEN in your .env file first.")
    client.run(DISCORD_BOT_TOKEN)

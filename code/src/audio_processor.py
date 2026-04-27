import torch
import whisper
import numpy as np
import discord.opus
from discord.ext import voice_recv
from .classifier import ToxicityClassifier

class ToxicitySink(voice_recv.AudioSink):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.classifier = ToxicityClassifier()
        self.stt_model = whisper.load_model("tiny") # Switched to tiny for speed
        self.buffer = bytearray()
        self.CHUNK_THRESHOLD = 76800 
        self.decoder = discord.opus.Decoder()
        
        # --- NEW: STRIKE SYSTEM TRACKING ---
        self.user_strikes = {} # Format: {user_id: count}

    def wants_opus(self):
        return True 

    def cleanup(self):
        self.buffer.clear()
        self.user_strikes.clear()
        print("Sink cleanup complete.")

    def write(self, user, data):
        # 1. Grab raw data
        raw = getattr(data, 'data', None) or getattr(data, 'decrypted_data', None)
        
        if raw is None or isinstance(raw, int):
            return

        try:
            # 2026 Padding Strip
            if (raw[0] & 0x20):
                padding_len = raw[-1]
                raw = raw[:-padding_len]

            # DAVE Bypass (Stripping the 2026 16-byte header)
            opus_frame = raw[16:] if len(raw) > 16 else raw
            
            # Manual Decode to PCM
            pcm_data = self.decoder.decode(opus_frame, fec=False)
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # --- NEW: VOLUME/ACOUSTIC CHECK ---
            # If the user is shouting, we want to know
            volume = np.sqrt(np.mean(pcm_array.astype(np.float32)**2))
            
            # Skip total silence
            if volume < 500: 
                return

            self.buffer.extend(pcm_array[0::2].tobytes())

            if len(self.buffer) >= self.CHUNK_THRESHOLD:
                chunk = self.buffer[:self.CHUNK_THRESHOLD]
                self.buffer = self.buffer[self.CHUNK_THRESHOLD:]
                
                # We pass the volume level to the processor
                self.bot.loop.create_task(self.process_audio(user, chunk, volume))

        except Exception:
            return

    async def process_audio(self, user, raw_data, volume_level):
        try:
            # 1. Convert to Float
            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 2. Acoustic Multiplier (Shouting boost)
            # If volume_level (from PCM) was high, we boost sensitivity
            multiplier = 1.3 if volume_level > 5000 else 1.0

            # 3. Transcription
            result = self.stt_model.transcribe(
                audio_np, 
                fp16=torch.cuda.is_available(),
                initial_prompt="I hate pickles. This is a game. Don't be toxic."
            )
            text = result['text'].strip()
            
            if text:
                # 4. Contextual "Pickle" Fix
                neutral_targets = ["pickle", "food", "game", "lag", "weather", "pizza", "homework"]
                is_neutral = any(obj in text.lower() for obj in neutral_targets)

                # 5. Classifier Prediction
                # We use the probability logic from your classifier
                vec_text = self.classifier.vectorizer.transform([text])
                base_prob = self.classifier.model.predict_proba(vec_text)[0][1]
                
                final_prob = base_prob * multiplier
                
                # Decision
                status = "Tier 2" if (final_prob > 0.70 and not is_neutral) else "Clean"
                print(f"[{user.name}]: {text} | Prob: {final_prob:.2f} | Status: {status}")

                # 6. 3-Strike Rule
                if status == "Tier 2":
                    strikes = self.user_strikes.get(user.id, 0) + 1
                    self.user_strikes[user.id] = strikes
                    
                    print(f"{user.name} Strike {strikes}/3")
                    
                    if strikes >= 3:
                        await self.mute_member(user)
                        self.user_strikes[user.id] = 0 # Reset after punishment
                    
        except Exception as e:
            print(f"STT Error: {e}")

    async def mute_member(self, user):
        for guild in self.bot.guilds:
            member = guild.get_member(user.id)
            if member and member.voice:
                try:
                    await member.edit(mute=True, reason="3-Strike Toxicity Filter")
                    print(f"!!! DISCIPLINARY ACTION: Muted {member.display_name}")
                except Exception as e:
                    print(f"Mute permission error: {e}")
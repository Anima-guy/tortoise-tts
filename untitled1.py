
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
import pickle as pkl
from torch.cuda.amp import autocast

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

# This is the text that will be spoken.
text = "I'm Cleo, Cleo Telerín. And this is my brother Cuquín. There is also Pelusín, Colitas, Teté, Maripi and Ghost. When I grow up I want to be an Italian plumber. Ha-ha!"

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "high_quality"


# with open('voice_samples_v2.pkl', 'rb') as file:
#     voice_samples = pkl.load(file)
    
# with open('conditioning_latents_v2.pkl', 'rb') as file:
#     conditioning_latents = pkl.load(file)

with autocast:
    voice_samples, conditioning_latents = load_voice('cleo') 
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)

torchaudio.save('generated-cleo.wav', gen.squeeze(0).cpu(), 24000)
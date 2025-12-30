from kokoro import KPipeline, KModel
from IPython.display import display, Audio
import soundfile as sf
import torch

from bitnet import replace_linears_in_pytorch_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)

# replace_linears_in_pytorch_model(model)
model.eval()

# TODO: Re-Train the model with the new linears

# TODO: save the full model to a file (not state dictionary, the entire model)

pipeline = KPipeline(lang_code="a", model=model)
text = """
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
"""
generator = pipeline(text, voice="af_heart")
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i == 0))
    sf.write(f"{i}.wav", audio, 24000)



# RVC Text-to-Speech 

This is a text-to-speech package for [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) models, using [edge-tts](https://github.com/rany2/edge-tts).



 # Example:

 ```python
from rvc_tts import RVCTTS

tts = RVCTTS()
info, edge_output_filename, audio_opt = tts.tts(
    model_name="model1",
    speed=0,
    tts_text="Hello, world!",
    tts_voice="en-US-JennyNeural",
    f0_up_key=0,
    f0_method="rmvpe",
    index_rate=1,
    protect=0
)
```

# RVC Text-to-Speech

RVC Text-to-Speech (`rvc-tts`) is a Python package that integrates [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) models with [edge-tts](https://github.com/rany2/edge-tts) to provide advanced text-to-speech capabilities.

## Features

- **Text-to-Speech Conversion**: Converts input text into speech using Microsoft's neural voices via `edge-tts`.
- **Voice Conversion**: Transforms the generated speech into a target voice using RVC models.
- **Customization Options**: Allows adjustments to speech speed, pitch, and other parameters for personalized output.

## Installation

Ensure you have Python 3.8 or higher installed. You can install the `rvc-tts` package using pip:

```bash
pip install rvc-tts
```

## Usage

Here's an example of how to use the `rvc-tts` package:

```python
from rvc_tts import RVCTTS

# Initialize the TTS engine
tts = RVCTTS()

# Generate speech
info, edge_output_filename, audio_opt = tts.tts(
    model_name="model1",               # Name of the RVC model to use
    speed=0,                           # Speech speed adjustment (0 for default)
    tts_text="Hello, world!",          # Text to convert to speech
    tts_voice="en-US-JennyNeural",     # Voice selection from edge-tts
    f0_up_key=0,                       # Pitch adjustment
    f0_method="rmvpe",                 # Pitch extraction method
    index_rate=1,                      # Index rate for voice conversion
    protect=0                          # Protection parameter (refer to RVC documentation)
)
```

## Parameters

- **model_name**: Specifies the RVC model to use for voice conversion.
- **speed**: Adjusts the speech speed. Positive values increase speed, negative values decrease it, and `0` maintains the default speed.
- **tts_text**: The input text string that you want to convert to speech.
- **tts_voice**: Selects the voice for speech synthesis. A list of available voices can be found in the [edge-tts documentation](https://github.com/rany2/edge-tts).
- **f0_up_key**: Adjusts the pitch of the output speech. Positive values raise the pitch, negative values lower it, and `0` leaves it unchanged.
- **f0_method**: Specifies the pitch extraction method. Options include `"rmvpe"` and others as detailed in the RVC documentation.
- **index_rate**: Controls the blending rate between the original and converted voice. A value of `1` uses only the converted voice, while lower values mix in more of the original.
- **protect**: A parameter for advanced users to control specific aspects of voice conversion. Refer to the RVC documentation for detailed usage.

## Requirements

- **Python**: Version 3.8 or higher.
- **PyTorch**: Ensure that PyTorch is installed in your environment. Installation instructions can be found on the [official PyTorch website](https://pytorch.org/get-started/locally/).
- **ffmpeg**: Required for audio processing. Installation instructions are available on the [ffmpeg official website](https://ffmpeg.org/download.html).

## Additional Resources

- **RVC Documentation**: For more details on RVC models and their capabilities, visit the [RVC GitHub repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
- **edge-tts Documentation**: For information on available voices and additional settings, refer to the [edge-tts GitHub repository](https://github.com/rany2/edge-tts).

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Bredvige/rvc_tts/blob/main/LICENSE) file for details.

## Acknowledgments

Special thanks to the developers of [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and [edge-tts](https://github.com/rany2/edge-tts) for their contributions to the open-source community.


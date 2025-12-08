# EPUB 3 Media Overlays Creator using TTS

**Turn a static EPUB into a narrated audiobook with text highlighting (EPUB 3 Media Overlays).**

## Key Features
* **🔒 Offline:** The method runs entirely offline, and thereby eliminating the recurring costs, privacy concerns, and copyright compliance issues associated with cloud-based solutions.
* **🎯 Zero Drift:** Calculating timestamps during TTS rather than forced alignment. The audio and text never go out of sync.
* **🎨 Layout Preservation:** Publisher’s original typography, styling, and embedded media are preserved.
* **🌓 Dark Mode Support:** Automatically generates high-contrast highlighting, e.g. purple for dark mode, yellow for light mode.
* 🧠 Supports both **Chatterbox** and **XTTS-v2**.

---

## Prerequisites
1.  **Python 3.11** (Recommended) or 3.10. *(Note: Python 3.12 is not yet supported by some audio libraries).*
2.  **FFmpeg** installed on your system (required for audio processing).
    * **Ubuntu/Debian:** `sudo apt install ffmpeg`
    * **MacOS:** `brew install ffmpeg`
    * **Windows:** `winget install ffmpeg`

---

## Installation
We recommend using a virtual environment to manage dependencies. If you want to use both the XTTS and Chatterbox versions, we recommend to use different virtual environments.

```bash
# Chatterbox
# 1. Create Environment
python3.11 -m venv env_chat
source env_chat/bin/activate
# 2. Install Dependencies
pip install -r requirements/requirements_chatterbox.txt

# XTTS
# 1. Create Environment
python3.11 -m venv env_xtts
source env_xtts/bin/activate
# 2. Install Dependencies
pip install -r requirements/requirements_xtts.txt
```

## Usage
You need a clean Reference Voice Sample (WAV file, ~15 seconds) to clone the narrator's voice. A sample in English is provided in assets/neutral_narrator.wav (Google translate narrator). Several samples for multiple languages can also be found at [Chatterbox TTS Demo Samples](https://resemble-ai.github.io/chatterbox_demopage/).

## Running with Chatterbox (Standard)
Ensure your env_chat environment is active.
```bash
python src/epub_processor_chatterbox.py "my_book.epub" --voice assets/neutral_narrator.wav
```

## Running with XTTS
Ensure your env_xtts environment is active.
```bash
python src/epub_processor_xtts.py "my_book.epub" --voice assets/neutral_narrator.wav
```

## Command Line Arguments

| Argument        | Description                                                  |
|-----------------|--------------------------------------------------------------|
| input_file      | Path to the source EPUB file.                                |
| --voice / -v    | Required. Path to a 10-15s WAV file for voice cloning.       |
| --output / -o   | (Optional) Custom path for the resulting EPUB.               |
| --gpu           | Force GPU usage if available (Recommended for speed).        |
| --skip-audio    | Debug mode. Generates XHTML/SMIL structure but skips TTS.    |


## How It Works
Extraction: The tool unzips the EPUB and parses the internal structure (OPF/NCX).
Injection: It iterates through every HTML chapter, identifying text paragraphs. It wraps sentences in <span> tags with unique IDs without disturbing surrounding HTML/CSS.
Synthesis: It sends the text to the local AI model. The model generates audio and reports the exact duration.
Alignment: The tool writes a SMIL file linking the HTML IDs to the audio timestamps. Gaps between sentences are mathematically stitched to ensure continuous highlighting.
Packaging: The modified files are re-zipped into a valid EPUB 3 file.

## Troubleshooting
"FFmpeg not found" Ensure FFmpeg is installed and added to your system PATH. Run ffmpeg -version in your terminal to verify.<br>
"Out of Memory" (CUDA) Neural TTS is heavy. If you run out of VRAM, the script will automatically fallback to CPU (slower, but works).

## License & Acknowledgements
**Code:** The source code in this repository is released under the **MIT License**. See `LICENSE` for details.

**Models:**
* **Chatterbox:** Uses the MIT license.
* **XTTS-v2:** Released under the **Coqui Public Model License (CPML)**. 

<!--
## Citation
If you use this code for your research, please cite the following paper:
> **[Paper Title]**
> *[Your Name], [Co-Author Name]*
> [Conference/Journal Name], 2025.

**BibTeX:**
```bibtex
@inproceedings{YourSurname2025,
  title = {Automated Generation of Synchronized EPUB 3 Media Overlays using Neural Text-to-Speech},
  author = {Your Name and Co-Author Name},
  booktitle = {Proceedings of the [Conference Name]},
  year = {2025},
  note = {To appear}
}
```
-->










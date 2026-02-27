# AI Agent Instructions (AGENTS.md)

## Project Context
This is a Music Information Retrieval (MIR) project focused on transcribing monophonic Arabic music. The core objective is high-fidelity, continuous pitch tracking and melody extraction without Western tuning biases.

## Tech Stack & Architecture
* **Language:** Python 3.x
* **Core Libraries:** PyTorch, librosa, scipy, numpy
* **Pitch Extraction:** PENN (Pitch-Estimating Neural Networks) utilizing `fcnf0++.pt` model weights.
* **Tuning System:** 53 EDO (Equal Division of the Octave) for accurate representation of Arabic Maqamat.

## 🛑 Hard Constraints for AI Agents
When generating code, suggesting architectural changes, or refactoring for this repository, you MUST adhere to the following rules:

### 1. Zero 12-TET Bias
* **NEVER** introduce functions that quantize, snap, or round pitch values to standard 12-Tone Equal Temperament (12-TET) MIDI notes.
* Pitch must be treated as a continuous variable (represented in continuous Hz or cents) to preserve Holdrian commas, microtones, and natural vibrato.

### 2. SOTA DSP Over Basic Libraries
* When suggesting solutions for pitch artifacts or octave errors, prioritize State-of-the-Art (SOTA) methodologies (e.g., ensemble tracking, HPSS pre-processing, median filtering, or custom Viterbi decoding) over basic moving averages.

### 3. Git & File Management
* Do **NOT** write scripts or git commands that stage or commit large machine learning model weights (e.g., `.pt`, `.pth`, `.h5` files).
* These files exceed standard Git limits and are explicitly handled via `.gitignore`.

### 4. Code Style
* Maintain modularity. Pitch extraction, scale tuning, rhythm quantization, and transcription logic should remain in their respective `/core` modules.
* Use explicit type hinting for all function parameters and return values, especially when handling numpy arrays and PyTorch tensors.
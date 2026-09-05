# WolfAlpha: WorldQuant BRAIN Alpha Generator & Simulator

WolfAlpha is an automated quantitative research assistant that uses Google Gemini (via the `google-genai` and `google-generativeai` SDKs) to design and iterate on WorldQuant BRAIN alpha expressions. It automatically simulates candidates against WorldQuant's simulation API, analyzes results (including Sharpe, fitness, and turnover), reads performance history, plots PnL graphs, and sends those graphs back to Gemini for visual analysis and refinement feedback.

---

## Features
- **Prompt Generator (`promptgen.py`)**: Assembles mathematical operators and available data fields (e.g. Price Volume, Fundamental, Analyst Estimate data) into structured prompts for the LLM.
- **Auto Simulation (`simulate.py` / `feedback.py`)**: Interacts with the WorldQuant BRAIN REST API, polls progress, handles retries, and captures insample stats.
- **PnL Visual Analysis (`pnl.py` / `feedback.py`)**: Visualizes simulation PnL, saves the curve as an image, and uses Gemini Vision to diagnose trends, drawdowns, and late-stage alpha behaviors.
- **Performance Monitor**: Tracks performance across iterations with real-time matplotlib plots.

---

## Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```
> **Note for Windows Users**: `pywin32` is required for window flashing/notification features.

### 3. Environment Configuration
Create a `.env` file in the root of the project directory based on `.env.example`:
```env
t=YOUR_WORLDQUANT_BRAIN_TOKEN
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```
To obtain the WorldQuant token `t`, log in to [WorldQuant BRAIN](https://worldquantbrain.com/), open your browser's Developer Tools (F12), inspect cookies for the `worldquantbrain.com` domain, and copy the value of the cookie named `t`.

---

## How to Run

### Option A: Fully Automated AI-Driven Loop
To run the automated research agent which generates, tests, analyzes, and refines Alphas in a loop:
```bash
python feedback.py
```
This script will:
1. Initialize folders like `prompts/`, `simulations/`, and `contexts/` if they don't exist.
2. Generate alpha expressions using Gemini.
3. Post the expressions to the WorldQuant BRAIN API.
4. Retrieve results and plot performance history.
5. Generate a PnL graph and send it to Gemini Vision for feedback on how to improve the alpha in the next iteration.

### Option B: Interactive Manual Simulation
If you want to manually paste mathematical formulas and test them in WorldQuant:
```bash
python simulate.py
```
Follow the interactive prompts in the terminal to input your expression and set up the simulation settings (Universe, Neutralization, Decay, etc.).

### Option C: Single PnL Graph Analyzer
To test the visual PnL analysis component on a specific Alpha ID:
```bash
python pnl.py
```

### Option D: Generate Prompts
To print the generated prompt containing all compiled data fields and operators:
```bash
python promptgen.py
```

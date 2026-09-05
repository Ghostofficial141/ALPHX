import config
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import os
import pickle
import time
import traceback
import utils
import requests
from PIL import Image
import base64
from dotenv import load_dotenv

# Load environment variables first — required for wq token 't' and API keys
load_dotenv()

os.makedirs('prompts', exist_ok=True)
if not os.path.exists(config.system_prompt_file):
    if os.path.exists('prompt.txt'):
        with open('prompt.txt', 'r') as f_src:
            default_prompt = f_src.read()
    else:
        default_prompt = """You are a Quantitative Researcher. Your task is to generate profitable Alpha expressions for the WorldQuant BRAIN platform.

An Alpha is a mathematical expression that predicts future stock returns.
Use the following available data fields to construct your Alpha Expression:
{data_fields_substitute}

Output your response matching the specified JSON schema."""
    with open(config.system_prompt_file, 'w') as f_dest:
        f_dest.write(default_prompt)

os.makedirs(os.path.dirname(config.simulations_file), exist_ok=True)
os.makedirs(os.path.dirname(config.context_file), exist_ok=True)

import promptgen
system_prompt = promptgen.prompt_with_fields()

with open(config.simulations_file, 'w+') as f:
    json.dump([], f, indent=2)

with open(config.context_file, 'wb') as f:
    pickle.dump([], f)

clr = utils.clr
wq_session = utils.wq_login()

# --- API Configuration ---
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = 'nvidia/nemotron-3.5-lightning:free'   # Single model for all purposes


schema_instructions = f"""

IMPORTANT: You must return your response as a valid JSON object matching the following schema:
{{
  "Alpha Expression": "string - The mathematical formula of your Alpha",
  "Universe": "string - One of: {', '.join(utils.simul.Universe)}",
  "Delay": "integer - delay (0 or 1)",
  "Neutralization": "string - One of: {', '.join(utils.simul.Neutralization)}",
  "Decay": "integer - decay (e.g. 0 to 30)",
  "Truncation": "number - truncation (e.g. 0.01 to 0.1)",
  "NaN Handling": "string - One of: {', '.join(utils.simul.NaN_Handling)}",
  "Reasoning": "string - Short explanation of your alpha's rationale"
}}
Do not include any thinking or markdown block wrappers like ```json. Return ONLY the raw JSON object.
"""


def openrouter_chat(messages, model=None, temperature=None, enable_reasoning=True, max_tokens=4096, max_retries=5):
    """Call OpenRouter using nvidia/nemotron-3.5-lightning:free with reasoning enabled.
    
    Reasoning details from previous turns are passed back unmodified so the model
    can continue its chain-of-thought across iterations (as per OpenRouter docs).
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/WolfAlpha",
        "X-Title": "WolfAlpha"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": True}   # Always enable reasoning for Nemotron
    }

    print(f"{clr.cyan}Calling model: {OPENROUTER_MODEL}...{clr.white}")

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                data=json.dumps(payload),   # Use data=json.dumps (not json=) as per OpenRouter docs
                timeout=180
            )

            if response.status_code == 200:
                res_json = response.json()
                if "choices" in res_json and len(res_json["choices"]) > 0:
                    return res_json
                else:
                    print(f"{clr.yellow}Empty choices in response. Retrying... (attempt {attempt+1}/{max_retries}){clr.white}")

            elif response.status_code in [429, 502, 503]:
                wait = 10 if response.status_code == 429 else 5
                print(f"{clr.yellow}OpenRouter {response.status_code}. Retrying in {wait}s... (attempt {attempt+1}/{max_retries}){clr.white}")
                time.sleep(wait)

            else:
                print(f"{clr.red}OpenRouter Error {response.status_code}: {response.text[:300]}{clr.white}")
                time.sleep(5)

        except Exception as e:
            print(f"{clr.yellow}Request error: {e}. Retrying... (attempt {attempt+1}/{max_retries}){clr.white}")
            time.sleep(5)

    raise Exception(f"nvidia/nemotron-3.5-lightning:free failed after {max_retries} attempts.")


def llm_vision_analyze(encoded_image, prompt):
    """Analyze a PnL graph image using nvidia/nemotron-3.5-lightning:free."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
            ]
        }
    ]
    print(f"{clr.yellow}Sending PnL graph to {OPENROUTER_MODEL} for analysis...{clr.white}")
    res = openrouter_chat(messages, temperature=0.2)
    return res["choices"][0]["message"]["content"]



def normalize_and_validate_json(parsed):
    required_keys = {
        "Alpha Expression": ["alpha_expression", "alphaexpression", "expression", "alpha"],
        "Universe": ["universe"],
        "Delay": ["delay"],
        "Neutralization": ["neutralization"],
        "Decay": ["decay"],
        "Truncation": ["truncation"],
        "NaN Handling": ["nan_handling", "nanhandling", "nan"],
        "Reasoning": ["reasoning"]
    }
    
    normalized = {}
    parsed_lower = {k.lower().replace(" ", "").replace("_", "").replace("-", ""): v for k, v in parsed.items()}
    
    for standard_key, alternatives in required_keys.items():
        # check if standard_key is in parsed
        if standard_key in parsed:
            normalized[standard_key] = parsed[standard_key]
        else:
            found = False
            std_key_clean = standard_key.lower().replace(" ", "").replace("_", "").replace("-", "")
            if std_key_clean in parsed_lower:
                normalized[standard_key] = parsed_lower[std_key_clean]
                found = True
            else:
                for alt in alternatives:
                    alt_clean = alt.lower().replace(" ", "").replace("_", "").replace("-", "")
                    if alt_clean in parsed_lower:
                        normalized[standard_key] = parsed_lower[alt_clean]
                        found = True
                        break
            if not found:
                raise KeyError(f"Missing required key: '{standard_key}'")
                
    # Normalize data types as well
    if "Delay" in normalized:
        try:
            normalized["Delay"] = int(normalized["Delay"])
        except Exception:
            pass
    if "Decay" in normalized:
        try:
            normalized["Decay"] = int(normalized["Decay"])
        except Exception:
            pass
    if "Truncation" in normalized:
        try:
            normalized["Truncation"] = float(normalized["Truncation"])
        except Exception:
            pass
            
    return normalized

def generate_chat_completion(context, system_prompt, schema_instructions):
    """Generates the chat completion from OpenRouter.
    
    Returns:
        parsed_json (dict): The parsed JSON model response.
        raw_text_or_blocks (str): The raw output string from OpenRouter.
        reasoning_text (str): The thinking process, if any, for logging/debugging.
    """
    messages = [{"role": "system", "content": system_prompt.strip() + schema_instructions}]
    for msg in context:
        item = {"role": msg["role"], "content": msg["content"]}
        if "reasoning_details" in msg:
            item["reasoning_details"] = msg["reasoning_details"]
        messages.append(item)
        
    res = openrouter_chat(messages, enable_reasoning=True)
    msg_obj = res["choices"][0]["message"]
    raw_text = msg_obj.get("content") or ""
    reasoning_details = msg_obj.get("reasoning_details")
    
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as je:
        print(f'{clr.yellow}Failed to parse JSON from model output.{clr.white}')
        print(f'{clr.yellow}Raw output: {raw_text[:300]}...{clr.white}')
        raise je
        
    normalized = normalize_and_validate_json(parsed)
    return normalized, raw_text, reasoning_details


context = []
performance_history = []

plt.ion()
plt.style.use('dark_background')
fig, ax = plt.subplots()
line, = ax.plot(performance_history, marker='o')
ax.set_title('Performance History')
ax.set_xlabel('Iteration')
ax.set_ylabel('Performance')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.canvas.manager.set_window_title('WolfAlpha')

def update_peformance(peformance):
    performance_history.append(peformance)
    line.set_xdata(range(1, 1 + len(performance_history)))
    line.set_ydata(performance_history)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

plt.show(block=False)


def analyze_pnl_graph(alpha_id: str, wq_token: str) -> str:
    MAX_WAIT_SECONDS = 120 
    POLL_INTERVAL_SECONDS = 10 

    IMAGE_FILENAME = "temp_pnl_graph.png"

    start_time = time.time()
    pnl = None

    print(f"{clr.yellow}--- Starting PnL Analysis for alpha: {alpha_id} ---{clr.white}")
    
    while time.time() - start_time < MAX_WAIT_SECONDS:
        try:
            url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl"
            headers = {"Authorization": f"Bearer {wq_token}"}
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                error_msg = f"API returned non-200 status. Status: {response.status_code}, Body: {response.text}"
                print(f"{clr.red}{error_msg}{clr.white}")
                return f"Note: PnL Graph analysis failed. Reason: {error_msg}"
            
            data = response.json()

            if data.get('records'):
                pnl = data['records']
                print(f"{clr.green}Successfully fetched PnL data after {int(time.time() - start_time)} seconds.{clr.white}")
                break
            else:
                print(f"{clr.yellow}PnL data not yet populated. Retrying in {POLL_INTERVAL_SECONDS}s...{clr.white}")

        except json.JSONDecodeError:
            print(f"{clr.yellow}PnL data not yet available (received empty response). Retrying in {POLL_INTERVAL_SECONDS}s...{clr.white}")
        
        except Exception as e:
            error_summary = f"An unexpected error occurred while polling for PnL data: {e}"
            print(f"{clr.red}{error_summary}\n{traceback.format_exc()}{clr.white}")
            return f"Note: PnL analysis failed. Reason: {error_summary}"
        
        time.sleep(POLL_INTERVAL_SECONDS)

    if not pnl:
        timeout_msg = f"Timed out after {MAX_WAIT_SECONDS} seconds waiting for PnL data."
        print(f"{clr.red}{timeout_msg}{clr.white}")
        return f"Note: PnL analysis failed. Reason: {timeout_msg}"

    try:
        pnl_x = list(range(len(pnl)))
        pnl_y = [point[1] for point in pnl]

        pnl_fig, pnl_ax = plt.subplots(figsize=(10, 6))
        pnl_ax.plot(pnl_x, pnl_y, label=f'PnL for {alpha_id}', color='cyan')
        pnl_ax.set_title("PnL (Profit and Loss) Curve")
        pnl_ax.set_xlabel("Time")
        pnl_ax.set_ylabel("Cumulative PnL")
        pnl_ax.legend()
        pnl_ax.grid(True)
        pnl_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        pnl_fig.savefig(IMAGE_FILENAME, format='png', dpi=150, bbox_inches='tight')
        plt.close(pnl_fig)
        print(f"{clr.yellow}PnL graph saved as '{IMAGE_FILENAME}'.{clr.white}")

        with open(IMAGE_FILENAME, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        pnl_analysis_prompt = """Analyze this financial PnL (Profit and Loss) curve. Provide a concise summary focusing on the alpha's performance characteristics. Describe:
1. Overall Trend and Profitability: Is it generally profitable?
2. Volatility: Is the curve smooth or erratic?
3. Major Drawdowns: Identify any significant periods of loss.
4. Late-stage Performance: How did the alpha perform towards the end of the period shown?"""

        pnl_summary = llm_vision_analyze(encoded_image, pnl_analysis_prompt)
        print(f"{clr.green}--- PnL Analysis Complete ---{clr.white}")
        return pnl_summary

    except Exception as e:
        error_summary = f"An error occurred during graph creation or vision analysis: {e}"
        print(f"{clr.red}{error_summary}\n{traceback.format_exc()}{clr.white}")
        return f"Note: PnL Graph analysis failed after data fetch. Reason: {error_summary}"


class Model:
    def count_tokens(context):
        """Estimate token count from message character length."""
        total_chars = 0
        for msg in context:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            total_chars += len(part.get("text", ""))
                        elif part.get("type") == "thinking":
                            total_chars += len(part.get("thinking", ""))
        return total_chars // 4  # rough estimate: 1 token ≈ 4 chars

    def get_output(context):
        """Call LLM API (Anthropic or OpenRouter) to generate an Alpha JSON response."""
        while True:
            try:
                return generate_chat_completion(context, system_prompt, schema_instructions)
            except json.JSONDecodeError as je:
                print(f'{clr.yellow}Failed to parse JSON from model output. Retrying...{clr.white}')
                time.sleep(2)
            except (KeyError, ValueError) as ve:
                print(f'{clr.yellow}JSON validation failed: {ve}. Retrying...{clr.white}')
                time.sleep(2)
            except Exception as e:
                err_str = str(e)
                if any(x in err_str for x in ["429", "503", "502", "rate"]):
                    print(f'{clr.yellow}API temporary issue. Retrying in 10 seconds...{clr.white}')
                    time.sleep(10)
                else:
                    print(f'{clr.red}{traceback.format_exc()}{clr.white}')
                    time.sleep(3)

    def process_output(alpha):
        payload = {'type': 'REGULAR', 'settings': {'nanHandling': alpha['NaN Handling'], 'instrumentType': 'EQUITY', 'delay': alpha['Delay'], 'universe': alpha['Universe'], 'truncation': alpha['Truncation'], 'unitHandling': 'VERIFY', 'testPeriod': 'P0D', 'pasteurization': 'ON', 'region': 'USA', 'language': 'FASTEXPR', 'decay': alpha['Decay'], 'neutralization': alpha['Neutralization'], 'visualization': False}, 'regular': alpha['Alpha Expression']}
        return payload

    def get_context(i, model_output):
        model_context = f"Iteration #{i + 1}\nAlpha Expression:\n{model_output['Alpha Expression']}\n\nReasoning:\n{model_output['Reasoning']}\n\nSimulation Settings:\nUniverse: {model_output['Universe']}\nDelay: {model_output['Delay']}\nNeutralization: {model_output['Neutralization']}\nDecay: {model_output['Decay']}\nTruncation: {model_output['Truncation']}\nNaN Handling: {model_output['NaN Handling']}"
        return model_context.strip()


class User:
 
    def get_context(simul_resp):
        insample = simul_resp['is']
        checks = insample.get('checks', [])
        user_context = (
            f"Simulation Results:\n"
            f"Sharpe: {insample['sharpe']}\n"
            f"Fitness: {insample['fitness']}\n"
            f"Performance: {simul_resp['Score Change']}\n"
            f"Turnover: {round(100 * insample['turnover'], 2)}%\n"
            f"Returns: {round(100 * insample.get('returns', 0), 2)}%\n"
            f"Drawdown: {round(100 * insample.get('drawdown', 0), 2)}%"
        )

        # Report every check result so the LLM knows exactly what passed/failed
        failed_checks = []
        for check in checks:
            name = check.get('name', 'UNKNOWN')
            result_val = check.get('result', 'UNKNOWN')
            value = check.get('value')
            limit = check.get('limit')
            date = check.get('date')
            message = check.get('message', '')

            if result_val not in ('PASS', 'PASS_WITH_INFO'):
                details = f"  CHECK FAILED [{name}]: result={result_val}"
                if value is not None:
                    details += f", value={round(float(value) * 100, 2) if isinstance(value, float) and value < 2 else value}"
                if limit is not None:
                    details += f", limit={round(float(limit) * 100, 2) if isinstance(limit, float) and limit < 2 else limit}"
                if date:
                    details += f", date={date}"
                if message:
                    # Unit errors often have a detailed message — include it fully
                    details += f"\n  Message: {message.replace('; ', chr(10) + '  ')}"
                failed_checks.append(details)

        if failed_checks:
            user_context += "\n\nFailed Checks (you MUST fix these in the next iteration):\n"
            user_context += "\n".join(failed_checks)
        else:
            user_context += "\n\nAll checks passed!"

        return user_context.strip()


    def save_iteration(context, alpha):
        with open(config.simulations_file, 'r+') as f:
            data = json.load(f)
            data.append(alpha)
            f.seek(0)
            json.dump(data, f, indent=2)
        with open(config.context_file, 'wb') as f: pickle.dump(context, f)


# --- Main Execution Loop ---
initial_user_prompt = config.initial_prompt if config.initial_prompt.strip() else "Generate the first Alpha expression based on the system instructions and allowed fields."
context.append({"role": "user", "content": initial_user_prompt})
print(f'{clr.purple}Model: {OPENROUTER_MODEL} (via OpenRouter)')
print(f'Temperature: {config.temperature}')
print(f'System Prompt File: {config.system_prompt_file}{clr.white}')
print(f'{clr.green}{initial_user_prompt}{clr.white}')


for i in range(config.max_iterations):
    model_output, raw_text_or_blocks, reasoning_details = Model.get_output(context)
    model_context = Model.get_context(i, model_output)
    alpha_payload = Model.process_output(model_output)

    assistant_msg = {"role": "assistant", "content": raw_text_or_blocks}
    if reasoning_details:
        assistant_msg["reasoning_details"] = reasoning_details
    context.append(assistant_msg)
    print(f'{clr.cyan}{model_context}{clr.white}')
    if reasoning_details:
        print(f'{clr.purple}Thinking Process:\n{reasoning_details}\n{clr.white}')


    alpha = utils.Alpha.simulate(wq_session, alpha_payload)
    

    if not isinstance(alpha, dict) or "error" in alpha:
        error_msg = alpha.get("error", "Unknown error") if isinstance(alpha, dict) else "Unexpected format"
        status_code = alpha.get("status_code") if isinstance(alpha, dict) else None
        
        if status_code == 401:
            print(f"\n{clr.red}========================================================================{clr.white}")
            print(f"{clr.red}CRITICAL ERROR: Authentication failed (Status 401 Unauthorized).{clr.white}")
            print(f"{clr.red}Your WorldQuant BRAIN token ('t' in your .env file) has expired or is invalid.{clr.white}")
            print(f"{clr.red}Please refresh your token from the browser cookie and update your .env file.{clr.white}")
            print(f"{clr.red}========================================================================{clr.white}\n")
            break

        print(f"{clr.red}Simulation failed: {error_msg}. Sending error feedback to model.{clr.white}")

        context.append({"role": "user", "content": f"The simulation failed with error: {error_msg}. Please adjust the Alpha Expression or settings and try a different approach."})
        continue

    update_peformance(alpha['Score Change'])

    user_context = User.get_context(alpha)


    if 'id' in alpha:
        alpha_id = alpha['id']
        wq_token = os.getenv("t")
        
        if wq_token:
            pnl_summary = analyze_pnl_graph(alpha_id, wq_token)

            user_context += f"\n\nPnL Curve Analysis:\n{pnl_summary}"
        else:
            print(f"{clr.red}WorldQuant token 't' not found in environment. Skipping PnL analysis.{clr.white}")
            user_context += "\n\nPnL analysis was skipped because the WorldQuant token was not found."
    else:
        print(f"{clr.yellow}Could not find 'id' key in simulation response. Skipping PnL analysis.{clr.white}")
        user_context += "\n\nPnL analysis was skipped because no alpha ID was found in the simulation results."
        
    context.append({"role": "user", "content": user_context})
    print(f'{clr.green}{user_context}{clr.white}')

    User.save_iteration(context, alpha)
    token_count = Model.count_tokens(context)
    print(f'{clr.purple}Token Count: ~{token_count}{clr.white}')

    # Check if the alpha passed Sharpe, Fitness thresholds AND all BRAIN submission checks
    insample = alpha.get('is', {})
    sharpe = insample.get('sharpe')
    fitness = insample.get('fitness')
    sharpe_threshold = getattr(config, 'sharpe_threshold', 1.25)
    fitness_threshold = getattr(config, 'fitness_threshold', 1.0)

    thresholds_met = False
    if sharpe is not None and fitness is not None:
        try:
            sharpe_val = float(sharpe)
            fitness_val = float(fitness)
            thresholds_met = sharpe_val >= sharpe_threshold and fitness_val >= fitness_threshold
        except (ValueError, TypeError):
            pass

    all_checks_pass = utils.Alpha.is_submittable(alpha)

    if thresholds_met and all_checks_pass:
        print(f"\n{clr.green}========================================================================{clr.white}")
        print(f"{clr.green}SUCCESS: Alpha is fully submittable!{clr.white}")
        print(f"{clr.green}Sharpe ({sharpe_val} >= {sharpe_threshold}), Fitness ({fitness_val} >= {fitness_threshold}), all checks PASS.{clr.white}")
        print(f"{clr.green}========================================================================{clr.white}\n")
        break
    elif thresholds_met and not all_checks_pass:
        print(f"{clr.yellow}Sharpe/Fitness thresholds met but some BRAIN checks still failing — continuing to refine...{clr.white}")
    elif not thresholds_met and all_checks_pass:
        print(f"{clr.yellow}All BRAIN checks pass but Sharpe/Fitness below threshold — continuing to improve quality...{clr.white}")

    print()
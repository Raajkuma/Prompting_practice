import json

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Local model settings (editable in the sidebar)
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:12434/engines/v1"
DEFAULT_MODEL = "docker.io/qwen2.5:0.5B-F16"  # change to the exact model name your server lists
DEFAULT_MAX_TOKENS = 6000
DEFAULT_TEMPERATURE = 0.3
DEFAULT_FREQUENCY_PENALTY = 0.4  # discourages the model from repeating the same words

# ---------------------------------------------------------------------------
# Prompt (Role, Task, Context, Format, Restrictions)
# {{table_rows}} is generated in code so the Day 1 to Day 7 rows are always correct.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """
You are a nutrition-aware meal-planning assistant specialising in traditional South Indian cuisine and healthy, sustainable weight management.

Create a complete 7-day South Indian meal plan to support gradual weight loss, based on the user details below. Each day has exactly four meals: Breakfast, Lunch, Snack, and Dinner. Give estimated calories for every meal and a Daily Total for every day. Generate a fresh plan every time.

User details ("not provided" means the user did not share it):
- Diet preference: {{diet_preference}}
- Age: {{age}}
- Sex: {{sex}}
- Height: {{height}}
- Current weight: {{weight}}
- Target weight: {{target_weight}}

Always assume: Tamil Nadu, lightly active, about 30 minutes of weekday cooking, affordable budget. Never invent missing personal details.
If age, sex, height, or current weight is "not provided", create a general illustrative plan of about 1,300 to 1,500 kcal per day and state at the top that the calorie target needs personalisation.
If all four are provided, choose a moderate daily total between 1,200 and 1,800 kcal that suits the person's size and sex (taller or male users toward the upper end). Use target weight only as context for gradual progress and never promise a timeline or amount of weight loss.

Diet rules:
- Vegetarian: only vegetarian dishes. Dairy is allowed. No meat, fish, seafood, or eggs.
- Non-vegetarian: include lean dishes such as fish curry, meen kuzhambu, grilled or steamed fish, chicken curry, chicken stew, boiled egg, or egg curry with minimal oil on 4 to 5 days. Keep the other days vegetarian.

Dishes to choose from (vary them across the week):
- Breakfast: idli with sambar, dosa with chutney, pesarattu, adai, vegetable upma, ven pongal, ragi dosa, uthappam, semiya upma.
- Lunch: rice with sambar, kootu, poriyal, rasam, and curd or buttermilk; bisi bele bath; curd rice with poriyal; millet rice with dal.
- Snack: sundal, roasted chana, boiled peanuts, a fruit, sprouts salad, buttermilk.
- Dinner: dosa with sambar, chapati with dal or vegetable curry, vegetable khichdi, idiyappam with kurma, ragi roti, millet upma.

Start with this title: **7-Day South Indian [Vegetarian or Non-Vegetarian] Meal Plan for Weight Loss**, using the user's diet preference.

Then give ONE Markdown table. Copy the table below exactly, keep every Day and Meal label as written, and replace only the text inside angle brackets < >:

| Day | Meal | Menu | Portion Size | Calories (kcal) |
|---|---|---|---|---|
{{table_rows}}

Rules for filling the table:
- The Day column must contain only Day 1 to Day 7. Never use weekday names. Do not add or remove rows.
- Menu: write 1 to 3 different items joined with " + ", for example "2 idli + sambar + coconut chutney". Name each item only once. Never repeat the same dish within a cell or use the same menu for two meals on the same day.
- Portion Size: give a clear household quantity, such as 2 medium dosas, 1 cup cooked rice, 1 katori sambar, or 100 g fish.
- Calories (kcal): write one whole number per meal. Typical ranges: Breakfast 250 to 400, Lunch 350 to 500, Snack 100 to 200, Dinner 300 to 420.
- Daily Total: add the four meal calories of that day.
- Do not repeat the same Breakfast, Lunch, or Dinner on consecutive days.

After the table, add these five sections in this order:
1. Daily Calorie Summary: the Daily Total for Day 1 to Day 7 and the seven-day average.
2. Protein Summary: approximate protein in grams for each day.
3. Weekly Shopping List: grouped under grains, pulses and legumes, vegetables, fruits, dairy, and pantry items. Add a group for eggs, fish, and poultry if the diet is Non-vegetarian.
4. Meal Preparation Tips: five or six practical tips. For Non-vegetarian plans, include safe handling and storage of fish, chicken, and eggs.
5. Assumptions and Safety Note: say that calorie and protein values are approximate estimates that vary with ingredients, cooking methods, and portions.

INSTRUCTIONS:
1. Use only South Indian dishes that match the diet preference.
2. Keep meals balanced with vegetables, pulses or legumes, and a protein source.
3. Label all calories and protein as approximate. Do not present them as exact.
4. Do not recommend crash diets, detoxes, extreme restriction, or an arbitrary calorie deficit.
5. Do not promise a specific amount of weight loss.
6. Do not use quinoa, avocado, imported superfoods, protein powders, or expensive speciality ingredients.
7. Recommend consulting a qualified dietitian for personalised targets, medical conditions, pregnancy, or breastfeeding."""

NOT_PROVIDED = "not provided"


def or_not_provided(value, unit: str = "") -> str:
    """Return the value as text, or 'not provided' if it is empty or zero."""
    if value is None:
        return NOT_PROVIDED
    if isinstance(value, str):
        value = value.strip()
        return value if value else NOT_PROVIDED
    if value == 0:
        return NOT_PROVIDED
    return f"{value:g} {unit}".strip()


def build_table_rows() -> str:
    """Build the fixed Day 1 to Day 7 row skeleton (4 meals + 1 total per day)."""
    rows = []
    for day in range(1, 8):
        rows.append(f"| Day {day} | Breakfast | <menu> | <portion> | <number> |")
        rows.append(f"| Day {day} | Lunch | <menu> | <portion> | <number> |")
        rows.append(f"| Day {day} | Snack | <menu> | <portion> | <number> |")
        rows.append(f"| Day {day} | Dinner | <menu> | <portion> | <number> |")
        rows.append(f"| Day {day} | Daily Total | - | - | <sum of the four meals above> |")
    return "\n".join(rows)


def build_prompt(fields: dict) -> str:
    """Fill the {{placeholders}} in the prompt template."""
    prompt = PROMPT_TEMPLATE.replace("{{table_rows}}", build_table_rows())
    for key, value in fields.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    return prompt


def stream_local_model(base_url: str, model: str, prompt: str, max_tokens: int,
                       temperature: float, frequency_penalty: float):
    """Yield text chunks from an OpenAI-compatible /chat/completions endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "stream": True,
    }
    with requests.post(url, json=payload, stream=True, timeout=(10, 900)) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                yield text


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="South Indian Meal Plan Generator", page_icon="🍛", layout="wide")
st.title("🍛 South Indian Meal Plan Generator")
st.caption("Fill in your details, choose Vegetarian or Non-vegetarian, and generate a 7-day weight-loss meal plan.")

# Sidebar: local model settings
st.sidebar.header("Local model settings")
base_url = st.sidebar.text_input(
    "API base URL",
    value=DEFAULT_BASE_URL,
    help="Docker Model Runner: http://localhost:12434/engines/v1. "
         "If this port is a real Ollama server, try http://localhost:12434/v1.",
)
model_name = st.sidebar.text_input("Model name", value=DEFAULT_MODEL)
max_tokens = st.sidebar.number_input("Max output tokens", min_value=512, max_value=16000, value=DEFAULT_MAX_TOKENS, step=256)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1)
frequency_penalty = st.sidebar.slider(
    "Repetition penalty (frequency)", min_value=0.0, max_value=1.0, value=DEFAULT_FREQUENCY_PENALTY, step=0.1,
    help="Higher values reduce repeated dishes and words.",
)

left, right = st.columns([1, 2], gap="large")

# ---------------------------------------------------------------------------
# Input fields
# ---------------------------------------------------------------------------
with left:
    st.subheader("Your details")
    with st.form("user_inputs"):
        diet_preference = st.radio("Diet preference", ["Vegetarian", "Non-vegetarian"], horizontal=True)

        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", min_value=0, max_value=100, value=0, step=1,
                              help="Leave at 0 if you prefer not to share.")
        sex = c2.selectbox("Sex", ["Not provided", "Female", "Male", "Other"])

        c3, c4 = st.columns(2)
        height = c3.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=0.0, step=1.0)
        weight = c4.number_input("Current weight (kg)", min_value=0.0, max_value=300.0, value=0.0, step=0.5)

        target_weight = st.number_input("Target weight (kg)", min_value=0.0, max_value=300.0, value=0.0, step=0.5)

        submitted = st.form_submit_button("Generate meal plan", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Response field
# ---------------------------------------------------------------------------
with right:
    st.subheader("Your meal plan")
    output_area = st.container()

    if submitted:
        fields = {
            "diet_preference": diet_preference,
            "age": or_not_provided(age, "years"),
            "sex": NOT_PROVIDED if sex == "Not provided" else sex,
            "height": or_not_provided(height, "cm"),
            "weight": or_not_provided(weight, "kg"),
            "target_weight": or_not_provided(target_weight, "kg"),
        }
        prompt = build_prompt(fields)

        try:
            with output_area:
                with st.spinner("Creating your meal plan..."):
                    full_text = st.write_stream(
                        stream_local_model(
                            base_url, model_name, prompt,
                            int(max_tokens), float(temperature), float(frequency_penalty),
                        )
                    )
            st.session_state["meal_plan"] = full_text
            st.session_state["diet_label"] = diet_preference
        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to {base_url}. Check that the local model server is running "
                "and that the URL in the sidebar is correct."
            )
        except requests.exceptions.HTTPError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            st.error(f"The model server returned an error: {detail}")
        except Exception as exc:
            st.error(f"Could not generate the meal plan: {exc}")

    # Show the last generated plan on reruns (e.g. after clicking Download)
    elif "meal_plan" in st.session_state:
        with output_area:
            st.markdown(st.session_state["meal_plan"])

    else:
        with output_area:
            st.info("Your meal plan will appear here after you click **Generate meal plan**.")

    if "meal_plan" in st.session_state:
        st.download_button(
            "Download as Markdown",
            data=st.session_state["meal_plan"],
            file_name=f"{st.session_state.get('diet_label', 'meal').lower()}_meal_plan.md",
            mime="text/markdown",
        )

st.divider()
st.caption(
    "Calorie and protein values are approximate estimates. This tool does not replace advice from a "
    "qualified dietitian or doctor."
)

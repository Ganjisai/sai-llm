import streamlit as st
import os
import requests
import google.generativeai as genai
from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import concurrent.futures

# --- 1. SETUP ---
load_dotenv()

# Configure Google (Gemini)
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Groq
groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )

# Configure Hugging Face
hf_client = None
if os.getenv("HF_TOKEN"):
    hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Configure Cloudflare
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")


# --- 2. WORKER FUNCTIONS ---

def ask_gemini(prompt):
    """Google Gemini (With Auto-Fallback)"""
    models = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except:
            continue
    return " Gemini is busy/offline."


def ask_groq(prompt, model_id):
    """Groq Workers"""
    if not groq_client: return " Groq Key Missing"
    try:
        response = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"


def ask_huggingface(prompt):
    """Hugging Face Worker"""
    if not hf_client: return "HF Token Missing"
    try:
        model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
        messages = [{"role": "user", "content": prompt}]
        response = hf_client.chat_completion(
            messages, model=model_id, max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        if "403" in str(e): return " HF Permission Error (Check Token)"
        return f"HF Error: {e}"


def ask_cloudflare(prompt):
    """Cloudflare Worker"""
    if not CF_ACCOUNT_ID or not CF_API_TOKEN: return " Cloudflare Keys Missing"
    try:
        model = "@cf/meta/llama-3-8b-instruct"
        url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}"
        headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
        response = requests.post(url, headers=headers, json={"prompt": prompt})
        result = response.json()
        if result.get("success"):
            return result["result"]["response"]
        else:
            return "Cloudflare Error"
    except Exception as e:
        return f" CF Connection Failed: {e}"


# --- 3. THE CHAIRMAN FUNCTION ---

def ask_chairman(user_query, council_results):
    """
    The Chairman (Llama 3.3 70B) synthesizes the answer directly.
    """
    if not groq_client: return " Chairman Missing"

    context_text = ""
    for model_name, answer in council_results.items():
        context_text += f"\n=== ANSWER FROM {model_name} ===\n{answer}\n"

    # UPDATED PROMPT: Direct answer only, no meta-analysis.
    system_prompt = """
    You are the wise Chairman of an AI Council.
    1. Read the user's query and the draft answers from your council.
    2. Combine the best parts of all answers into ONE perfect, comprehensive response.
    3. Do NOT mention "Model X said this" or "Model Y is better". 
    4. Do NOT explain your reasoning.
    5. Just provide the final, polished answer directly to the user.
    """

    user_prompt = f"""
    USER QUERY: "{user_query}"

    COUNCIL DRAFTS:
    {context_text}

    Final Answer:
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" Chairman died: {e}"


# --- 4. THE UI ---
st.set_page_config(page_title="AI Council", layout="wide")

# CENTERED TITLE & SUBTITLE
st.markdown("<h1 style='text-align: center;'>SAI'S PERSONAL LLM</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>The best from four LLMs</h3>", unsafe_allow_html=True)

# UPDATED INPUT LABEL
query = st.text_input("Ask question")

if st.button("SUBMIT") and query:
    # 1. SHOW SPINNER WHILE WORKING
    with st.spinner("Finding the best answer from four LLMs..."):
        # Run in Parallel (Backend Only)
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f1 = executor.submit(ask_gemini, query)
            f2 = executor.submit(ask_groq, query, "llama-3.1-8b-instant")
            f3 = executor.submit(ask_huggingface, query)
            f4 = executor.submit(ask_cloudflare, query)

            results["Gemini 2.0"] = f1.result()
            results["Llama 3.1"] = f2.result()
            results["Qwen 2.5"] = f3.result()
            results["Cloudflare"] = f4.result()

    # 2. RUN CHAIRMAN
    with st.spinner("Synthesizing final answer..."):
        final_answer = ask_chairman(query, results)

    # 3. DISPLAY FINAL ANSWER FIRST
    st.success("FINAL ANSWER:")
    st.markdown(final_answer)

    # 4. DISPLAY INDIVIDUAL OPINIONS BELOW
    st.divider()
    st.write("individual LLM answers:")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption("Gemini (Google)")
        st.info(results["Gemini 2.0"])
    with c2:
        st.caption("Llama 3.1 (Groq)")
        st.success(results["Llama 3.1"])
    with c3:
        st.caption("Qwen 2.5 (HF)")
        st.warning(results["Qwen 2.5"])
    with c4:
        st.caption("Llama 8B (Cloudflare)")
        st.error(results["Cloudflare"])
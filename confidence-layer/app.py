import os
import streamlit as st
from groq import Groq
import json
import re

st.set_page_config(
    page_title="ChatGPT",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0B0F14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }
[data-testid="stSidebar"] { background: #131920 !important; border-right: 1px solid #2A3848; }
[data-testid="stSidebar"] * { color: #E2EAF4 !important; }
.user-bubble { background: #222D3E; border: 1px solid #3A4F68; border-radius: 14px 14px 4px 14px; padding: 12px 16px; font-size: 14px; line-height: 1.65; color: #E2EAF4; margin-left: 15%; }
.ai-bubble { background: #131920; border: 1px solid #2A3848; border-radius: 14px 14px 14px 4px; padding: 14px 16px; font-size: 14px; line-height: 1.7; color: #E2EAF4; margin-right: 5%; }
.sig-card { border-radius: 8px; padding: 9px 12px; margin-bottom: 7px; font-size: 12.5px; line-height: 1.55; border-left: 3px solid; }
.sig-uncertain { background:#201A08; border-left-color:#F5C842; }
.sig-verify    { background:#200D0D; border-left-color:#F87171; }
.sig-assumption{ background:#1A1230; border-left-color:#B794F4; }
.sig-ok        { background:#0B2016; border-left-color:#3DD68C; }
.sig-label { font-weight: 700; font-size: 12px; margin-bottom: 3px; }
.sig-uncertain .sig-label { color: #F5C842; }
.sig-verify    .sig-label { color: #F87171; }
.sig-assumption .sig-label { color: #B794F4; }
.sig-ok        .sig-label { color: #3DD68C; }
.rstep { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 9px; font-size: 12.5px; color: #E2EAF4; line-height: 1.55; }
.rnum { background: #0D2040; border: 1px solid #1A3D6B; color: #4EA8F8; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; flex-shrink: 0; margin-top: 1px; }
.risk-high   { background:#200D0D; border-left: 3px solid #F87171; border-radius: 8px; padding: 9px 12px; font-size: 12px; color: #8B949E; }
.risk-medium { background:#201A08; border-left: 3px solid #F5C842; border-radius: 8px; padding: 9px 12px; font-size: 12px; color: #8B949E; }
.risk-low    { background:#0B2016; border-left: 3px solid #3DD68C; border-radius: 8px; padding: 9px 12px; font-size: 12px; color: #8B949E; }
.section-header { font-size: 10px; font-weight: 700; color: #6B8099; letter-spacing: .8px; text-transform: uppercase; margin-bottom: 8px; margin-top: 4px; }
.metric-card { background: #1C2433; border: 1px solid #2A3848; border-radius: 10px; padding: 12px; text-align: center; }
.metric-num  { font-size: 26px; font-weight: 800; font-family: monospace; }
.metric-lbl  { font-size: 10px; color: #6B8099; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "signals_total": 0,
        "eval_answered": 0,
        "eval_answers": {},
        "msgs_sent": 0,
        "cl_on": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Prompts ───────────────────────────────────────────────────────────────────
MAIN_SYSTEM = """You are a helpful AI assistant. Answer the user's question thoroughly and helpfully.
Be specific and substantive. Use **bold** for section headers. Use backticks for code.
Keep answers well-structured and focused. Aim for 150-300 words unless more is needed."""

CL_SYSTEM = """You are the Confidence Layer analyzer for AI outputs.
Analyze AI responses and return ONLY a raw JSON object. No markdown, no backticks, no explanation before or after."""

def cl_prompt(question, response):
    return f"""Analyze this AI response and return ONLY valid JSON (no backticks, no text before or after):

{{
  "signals": [
    {{
      "type": "verify",
      "label": "short label 5-8 words",
      "claim": "exact short phrase from response under 8 words",
      "detail": "1-2 sentences explaining why this signal applies"
    }}
  ],
  "reasoning_steps": [
    "What the AI drew on to generate this response",
    "What knowledge sources or frameworks were used",
    "What the AI does NOT have access to",
    "Key assumptions made in this response"
  ],
  "eval_questions": [
    {{
      "question": "Question to help user evaluate if response fits their situation",
      "options": ["option A", "option B", "option C"]
    }},
    {{
      "question": "Another targeted evaluation question",
      "options": ["option A", "option B", "option C"]
    }}
  ],
  "confidence_breakdown": [
    {{ "label": "specific claim or section", "value": 80, "color": "#3DD68C" }},
    {{ "label": "another claim or section", "value": 55, "color": "#F5C842" }},
    {{ "label": "uncertain or risky claim", "value": 35, "color": "#F87171" }}
  ],
  "domain_risk": "1-2 sentences about specific risks of using this output without verification",
  "domain_risk_level": "medium"
}}

Signal type rules:
- "verify"     = factual stat/claim user should independently check
- "uncertain"  = context-dependent, contested, or oversimplified  
- "assumption" = AI assumed something about the user's situation
- "ok"         = reliable, well-established fact

Generate exactly 3 signals. Use real short phrases from the response as claims.
confidence_breakdown colors: #3DD68C for 75+, #F5C842 for 45-74, #F87171 for <45
domain_risk_level must be exactly: low, medium, or high

USER QUESTION: {question}
AI RESPONSE: {response}

Return only the JSON object:"""

# ── API ───────────────────────────────────────────────────────────────────────
MODEL_NAME = "llama-3.3-70b-versatile"

@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def call_ai(client, messages, system, max_tokens=1800):
    """Main chat call — supports multi-turn history."""
    msgs = [{"role": "system", "content": system}]
    for m in messages:
        msgs.append({"role": m["role"], "content": m["content"]})
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""

def call_ai_json(client, prompt, system, max_tokens=1200):
    """Single-turn call that FORCES valid JSON output."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""

def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

# ── Render helpers ────────────────────────────────────────────────────────────
def render_signal_card(sig):
    t   = sig.get("type", "uncertain")
    css = {"verify":"sig-verify","assumption":"sig-assumption","ok":"sig-ok"}.get(t,"sig-uncertain")
    st.markdown(f"""
    <div class="sig-card {css}">
      <div class="sig-label">{sig.get('label','')}</div>
      <div style="font-size:11px;color:#6B8099;font-style:italic;margin-bottom:4px">"{sig.get('claim','')}"</div>
      <div style="color:#8B949E">{sig.get('detail','')}</div>
    </div>""", unsafe_allow_html=True)

def render_reasoning(steps):
    icons = ["📚 Drew on","🔧 Used","🚫 Doesn't know","💡 Assumed"]
    for i, step in enumerate(steps):
        lbl = icons[i] if i < len(icons) else f"Step {i+1}"
        st.markdown(f"""
        <div class="rstep">
          <span class="rnum">{i+1}</span>
          <div><div style="font-size:9px;font-weight:700;color:#4B6080;letter-spacing:.5px;text-transform:uppercase;margin-bottom:2px">{lbl}</div>{step}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;color:#3D5370;margin-top:6px;padding:6px 8px;background:#1C2433;border-radius:6px">These steps surface what the AI drew on — and what it doesn\'t know.</div>', unsafe_allow_html=True)

def render_conf_bars(bars):
    for b in bars:
        val, color = b.get("value", 50), b.get("color", "#4EA8F8")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f'<div style="font-size:11px;color:#6B8099;margin-bottom:2px">{b.get("label","")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="height:5px;border-radius:3px;background:#222D3E;margin-bottom:8px"><div style="width:{val}%;height:100%;border-radius:3px;background:{color}"></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{color};text-align:right;padding-top:1px">{val}%</div>', unsafe_allow_html=True)

def render_domain_risk(text, level):
    css = {"high":"risk-high","low":"risk-low"}.get(level,"risk-medium")
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

def md_to_html(text):
    text = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`([^`]+)`', r'<code style="background:#222D3E;padding:1px 5px;border-radius:3px;font-family:monospace;font-size:.88em">\1</code>', text)
    lines = text.split('\n')
    out, in_ul, in_ol = [], False, False
    for line in lines:
        if re.match(r'^\s*[\*\-]\s+', line):
            if in_ol: out.append('</ol>'); in_ol = False
            if not in_ul: out.append('<ul style="margin:6px 0 6px 20px;color:#E2EAF4">'); in_ul = True
            out.append(f'<li style="margin-bottom:4px">{re.sub(r"^\\s*[\\*\\-]\\s+","",line)}</li>')
        elif re.match(r'^\s*\d+\.\s+', line):
            if in_ul: out.append('</ul>'); in_ul = False
            if not in_ol: out.append('<ol style="margin:6px 0 6px 20px;color:#E2EAF4">'); in_ol = True
            out.append(f'<li style="margin-bottom:4px">{re.sub(r"^\\s*\\d+\\.\\s+","",line)}</li>')
        else:
            if in_ul: out.append('</ul>'); in_ul = False
            if in_ol: out.append('</ol>'); in_ol = False
            out.append(line)
    if in_ul: out.append('</ul>')
    if in_ol: out.append('</ol>')
    return '\n'.join(out).replace('\n\n','<br><br>').replace('\n','<br>')

def highlight_text(text, signals):
    result = md_to_html(text)
    hl_styles = {
        "verify":     "background:rgba(248,113,113,.22);border-bottom:2px solid #F87171;padding:1px 3px;border-radius:3px;cursor:help",
        "uncertain":  "background:rgba(245,200,66,.22);border-bottom:2px solid #F5C842;padding:1px 3px;border-radius:3px;cursor:help",
        "assumption": "background:rgba(183,148,244,.22);border-bottom:2px solid #B794F4;padding:1px 3px;border-radius:3px;cursor:help",
        "ok":         "background:rgba(61,214,140,.18);border-bottom:2px solid #3DD68C;padding:1px 3px;border-radius:3px;cursor:help",
    }
    for sig in signals:
        claim = sig.get("claim","").strip()
        if not claim or len(claim) < 4: continue
        style = hl_styles.get(sig.get("type","uncertain"), hl_styles["uncertain"])
        label = sig.get("label","").replace('"',"'")
        esc   = claim.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        try:
            result = re.sub(f'({re.escape(esc)})', f'<span style="{style}" title="{label}">\\1</span>', result, count=1, flags=re.IGNORECASE)
        except: pass
    return result

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:18px;font-weight:800;color:#E2EAF4;margin-bottom:4px">ChatGPT</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#6B8099;margin-bottom:16px">AI Output Evaluation · Beta</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature</div>', unsafe_allow_html=True)
    cl_on = st.toggle("Confidence Layer", value=True, key="cl_on")
    st.markdown('<div style="font-size:11px;color:#6B8099;margin-top:4px">Toggle off to see raw AI output</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-header">Session Stats</div>', unsafe_allow_html=True)
    score = min(100, 40 + st.session_state.eval_answered * 5 + st.session_state.msgs_sent * 3) if st.session_state.msgs_sent > 0 else 0
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#3DD68C">{score}</div><div class="metric-lbl">Calibration</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#4EA8F8">{st.session_state.signals_total}</div><div class="metric-lbl">Signals seen</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:#6B8099;margin-top:8px">Messages: {st.session_state.msgs_sent} · Eval answers: {st.session_state.eval_answered}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-header">How it works</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11.5px;color:#6B8099;line-height:1.8">1. You ask a question<br>2. AI answers it<br>3. ChatGPT analyzes the response<br>4. Signals, reasoning &amp; eval questions appear<br>5. You decide what to trust</div>', unsafe_allow_html=True)

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.signals_total = 0
        st.session_state.eval_answered = 0
        st.session_state.msgs_sent     = 0
        st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown('<div style="font-size:22px;font-weight:800;color:#E2EAF4;margin-bottom:2px">ChatGPT <span style="background:#0D2040;border:1px solid #1A3D6B;color:#4EA8F8;font-size:11px;font-weight:600;padding:3px 8px;border-radius:20px;vertical-align:middle;margin-left:8px">BETA</span></div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:13px;color:#6B8099;margin-bottom:16px">Ask anything. The ChatGPT analyzes every response in real time — surfacing signals, reasoning gaps, and evaluation questions.</div>', unsafe_allow_html=True)

STARTER_CHIPS = [
    "Should I quit my job for a startup offering 0.5% equity and 30% salary cut?",
    "Review my resume: 5 years PM at SaaS, drove 40% growth, led 3 product launches",
    "Best GTM strategy for a B2B SaaS tool targeting SMBs with $2k ACV?",
    "Is iterrows() bad for large pandas DataFrames? What should I use instead?",
    "What caused inflation to stay high through 2024?",
    "How do I negotiate salary for my first job offer as a software engineer?",
    "What are the risks of putting all savings into index funds?",
    "How should I structure a PRD for a new feature?",
]

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:30px 20px;color:#6B8099">
      <div style="font-size:36px;margin-bottom:12px;opacity:.4">🔍</div>
      <div style="font-size:16px;font-weight:700;color:#E2EAF4;margin-bottom:8px">Ask anything — get a real answer with real analysis</div>
      <div style="font-size:13px;max-width:480px;margin:0 auto;line-height:1.7">AI answers your question, then the Confidence Layer immediately analyzes the response — showing exactly what to trust, verify, and what assumptions were made.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:700;color:#6B8099;letter-spacing:.8px;text-transform:uppercase;margin-top:8px;margin-bottom:8px">TRY A STARTER</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, chip in enumerate(STARTER_CHIPS):
        with cols[i % 2]:
            if st.button(chip, key=f"chip_{i}", use_container_width=True):
                st.session_state["prefill"] = chip
                st.rerun()

# ── Render messages ───────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">👤 &nbsp;{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

    elif msg["role"] == "assistant":
        analysis = msg.get("analysis")
        signals  = analysis.get("signals", []) if analysis else []

        if cl_on and analysis and signals:
            rendered = highlight_text(msg["content"], signals)
        else:
            rendered = md_to_html(msg["content"])

        st.markdown(f'<div class="ai-bubble">🤖 &nbsp;{rendered}</div>', unsafe_allow_html=True)

        if cl_on and analysis:
            # Render small colored pills under the AI response so users see highlights instantly
            if signals:
                pill_map = [
                    ("verify","VERIFY","#F87171"),
                    ("uncertain","UNCERTAIN","#F5C842"),
                    ("assumption","ASSUMPTION","#B794F4"),
                    ("ok","RELIABLE","#3DD68C"),
                ]
                pills_html = ''
                for t, lbl, col in pill_map:
                    if any(sig.get('type') == t for sig in signals):
                        pills_html += f'<span style="display:inline-block;padding:4px 8px;border-radius:999px;background:{col};color:#091017;font-weight:700;font-size:11px;margin-right:8px">{lbl}</span>'
                if pills_html:
                    st.markdown(f'<div style="margin-top:8px;margin-bottom:8px">{pills_html}</div>', unsafe_allow_html=True)

            with st.expander(f"🔍 Confidence Layer · {len(signals)} signal{'s' if len(signals)!=1 else ''} detected", expanded=True):
                tab1, tab2, tab3 = st.tabs(["📊 Signals", "🧠 Reasoning", "✅ Evaluate"])

                with tab1:
                    # Color legend at the top of Signals tab
                    legend_html = (
                        '<div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">'
                        '<div style="display:flex;gap:8px;align-items:center;font-size:12px;color:#E2EAF4">'
                        '<span style="width:12px;height:12px;background:#F87171;display:inline-block;border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.2)"></span><span style="color:#E2EAF4;margin-right:8px">Verify (red)</span>'
                        '<span style="width:12px;height:12px;background:#F5C842;display:inline-block;border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.2)"></span><span style="color:#E2EAF4;margin-right:8px">Uncertain (amber)</span>'
                        '<span style="width:12px;height:12px;background:#B794F4;display:inline-block;border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.2)"></span><span style="color:#E2EAF4;margin-right:8px">Assumption (purple)</span>'
                        '<span style="width:12px;height:12px;background:#3DD68C;display:inline-block;border-radius:2px;box-shadow:0 0 0 1px rgba(0,0,0,.2)"></span><span style="color:#E2EAF4">Reliable (green)</span>'
                        '</div></div>'
                    )
                    st.markdown(legend_html, unsafe_allow_html=True)

                    if signals:
                        for sig in signals:
                            render_signal_card(sig)
                    else:
                        st.markdown('<div style="color:#3DD68C;font-size:12px">No major issues detected.</div>', unsafe_allow_html=True)

                with tab2:
                    steps = analysis.get("reasoning_steps", [])
                    if steps:
                        render_reasoning(steps)

                with tab3:
                    eval_qs = analysis.get("eval_questions", [])
                    if eval_qs:
                        # For each evaluation question, use a persistent record to avoid double-counting
                        all_answered = True
                        for qi, eq in enumerate(eval_qs):
                            st.markdown(f'<div style="font-size:12.5px;color:#E2EAF4;margin-bottom:7px;font-weight:600">{eq["question"]}</div>', unsafe_allow_html=True)
                            rkey = f"eval_{idx}_{qi}"
                            ans = st.radio(
                                label=eq["question"],
                                options=eq.get("options", []),
                                key=rkey,
                                label_visibility="collapsed",
                                horizontal=True,
                                index=None,
                            )
                            # Record first-time answers only
                            if ans and rkey not in st.session_state.eval_answers:
                                st.session_state.eval_answers[rkey] = ans
                                st.session_state.eval_answered += 1

                            # Visual confirmation below each question
                            if st.session_state.eval_answers.get(rkey):
                                st.markdown('<div style="color:#3DD68C;font-weight:700;margin-top:6px">✓ Answer recorded</div>', unsafe_allow_html=True)
                            else:
                                all_answered = False

                        # If all eval questions for this response are answered, show overall confirmation
                        if all_answered and len(eval_qs) > 0:
                            st.markdown('<div style="background:#08140E;border:1px solid #3DD68C;color:#3DD68C;padding:10px;border-radius:8px;margin-top:10px">🎯 All questions answered — calibration score updated in sidebar</div>', unsafe_allow_html=True)

            if analysis.get("confidence_breakdown") or analysis.get("domain_risk"):
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    bars = analysis.get("confidence_breakdown", [])
                    if bars:
                        st.markdown('<div class="section-header">Confidence by claim</div>', unsafe_allow_html=True)
                        render_conf_bars(bars)
                with col_b:
                    risk_text  = analysis.get("domain_risk", "")
                    risk_level = analysis.get("domain_risk_level", "medium")
                    if risk_text:
                        st.markdown('<div class="section-header">Domain risk</div>', unsafe_allow_html=True)
                        render_domain_risk(risk_text, risk_level)

        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask anything…", key="chat_input")
question   = prefill or user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.msgs_sent += 1

    client      = get_client()
    api_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.spinner("Thinking…"):
        try:
            main_response = call_ai(client, api_history, MAIN_SYSTEM)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    analysis = None
    if cl_on:
        with st.spinner("Confidence Layer analyzing…"):
            try:
                # ✅ FIXED: pass plain string prompt, not a messages list
                cl_raw   = call_ai_json(client, cl_prompt(question, main_response), CL_SYSTEM)
                analysis = parse_json(cl_raw)
                if analysis:
                    st.session_state.signals_total += len(analysis.get("signals", []))
            except Exception as e:
                analysis = None

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  main_response,
        "analysis": analysis,
    })
    st.rerun()

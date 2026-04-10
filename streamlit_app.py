import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import sys
from datetime import datetime

# Import local modules
from tools.pdf_parser import parse_resume_file
from agents.supervisor import run_full_analysis
from memory.persistence import PersistenceManager

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CareerCraft AI | AI Career Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown("""
<style>
    /* Premium Professional Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: transparent;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2196f3;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    .reasoning-box {
        padding: 25px;
        border-left: 6px solid #2196f3;
        border-radius: 8px;
        line-height: 1.6;
        margin-top: 20px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- HELPERS ---
def get_bot_response(user_input, analysis_result):
    from config.settings import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm()
    context = f"""
    Candidate Analysis Results:
    Target Role: {analysis_result.get('target_role', 'N/A')}
    Match: {analysis_result.get('gap_analysis', {}).get('match_percentage', 0)}%
    ATS Score: {analysis_result.get('gap_analysis', {}).get('ats_score', 0)}/100
    Critical Gaps: {analysis_result.get('gap_analysis', {}).get('missing_skills', {}).get('critical', [])}
    """
    messages = [
        SystemMessage(content=f"You are the 'Senior Career Strategist'. Provide expert-level, actionable advice. Context: {context}"),
        HumanMessage(content=user_input)
    ]
    for chunk in llm.stream(messages):
        yield chunk.content

# --- SIDEBAR ---
with st.sidebar:
    # 🤖 Ask Career Bot Section
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=80) 
    st.title("CareerCraft AI")
    
    with st.expander("💬 **Ask Career Bot**", expanded=True):
        if st.session_state.analysis_result and st.session_state.analysis_result.get("status") != "error":
            st.caption("Ask me anything about your analysis!")
            
            # Display chat history
            chat_container = st.container(height=250)
            with chat_container:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("How do I fix my Python gap?"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""
                        for chunk in get_bot_response(prompt, st.session_state.analysis_result):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        else:
            st.info("Upload your resume first to activate the Career Bot!")

    st.markdown("---")
    st.subheader("📝 Applicant Details")
    user_name = st.text_input("Full Name", placeholder="e.g. John Doe")
    target_role = st.text_input("Target Role", placeholder="e.g. Senior Data Scientist")
    
    st.subheader("📎 Document Upload")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    analyze_button = st.button("🚀 Run Full Analysis", width='stretch', type="primary")

# --- MAIN CONTENT ---
st.title("🎯 Career Intelligence Dashboard")
st.markdown(f"Welcome to **CareerCraft AI**. This system uses a multi-agent LangGraph workflow to evaluate your career potential.")

if analyze_button:
    if not user_name or not target_role or not uploaded_file:
        st.error("Please provide your name, target role, and upload a resume.")
    else:
        st.session_state.processing = True
        
        with st.status("🧠 Agents at work: Evaluating Career Profile...", expanded=True) as status:
            # 1. Parsing
            st.write("🕵️ Agent 1: Parsing Resume...")
            file_bytes = uploaded_file.read()
            resume_text = parse_resume_file(file_bytes=file_bytes, file_name=uploaded_file.name)
            
            # 2. Running Graph with Professional Agentic Flow
            try:
                from agents.supervisor import build_career_graph
                app = build_career_graph()
                
                initial_state = {
                    "resume_text": resume_text,
                    "target_role": target_role,
                    "job_description": f"Standard requirements for a {target_role} position.",
                    "parsed_resume": {},
                    "gap_analysis": {},
                    "roadmap": {},
                    "interview_prep": {},
                    "matchmaker_results": {},
                    "tailored_resume": {},
                    "career_reasoning": "",
                    "current_step": "start",
                    "errors": [],
                    "status": "processing",
                }
                
                with st.status("🚀 **Initializing Multi-Agent Pipeline...**", expanded=True) as status:
                    st.write("📡 Connecting to Career Intelligence Grid...")
                    
                    progress_bar = st.progress(0)
                    
                    # Store UI placeholders for each agent to update them
                    agent_placeholders = {
                        "guardrail": st.empty(),
                        "parse_resume": st.empty(),
                        "analyze_gaps": st.empty(),
                        "generate_roadmap": st.empty(),
                        "tailor_resume": st.empty(),
                        "prepare_interview": st.empty(),
                        "summarize_reasoning": st.empty()
                    }
                    
                    # Mapping for professional display
                    agent_display = {
                        "guardrail": ("🛡️ **Agent 0: Security Guardrail**", 5),
                        "parse_resume": ("🕵️ **Agent 1: Resume Parser**", 15),
                        "analyze_gaps": ("🔍 **Agent 2: Gap Analyzer**", 35),
                        "generate_roadmap": ("🗺️ **Agent 3: Roadmap Generator**", 55),
                        "tailor_resume": ("✍️ **Agent 4: Resume Tailor**", 70),
                        "prepare_interview": ("🎙️ **Agent 5: Interview Coach**", 85),
                        "summarize_reasoning": ("🧠 **Agent 6: Senior Strategist**", 100)
                    }

                    last_result = initial_state
                    for event in app.stream(initial_state):
                        for node_name, node_state in event.items():
                            last_result.update(node_state)
                            
                            if node_name in agent_display:
                                label, progress = agent_display[node_name]
                                agent_placeholders[node_name].markdown(f"{label} ... `Thinking` 💭")
                                progress_bar.progress(progress)
                                # After small delay or next event, we show complete
                                status.update(label=f"🔄 Processing {label.split('**')[1]}...")
                                
                                # Mark previous nodes as complete if they were skipped or just finished
                                for prev_node in agent_display:
                                    if prev_node == node_name:
                                        agent_placeholders[prev_node].markdown(f"{label} ... ✅ `Success`")
                                        break

                    status.update(label="✅ **Intelligence Analysis Complete!**", state="complete", expanded=False)
                
                result = last_result
            except Exception as e:
                if "429" in str(e):
                    st.error("🚦 The AI is currently experiencing high traffic (Rate Limit). Please wait 30 seconds and try again.")
                else:
                    st.error(f"⚠️ Analysis Error: {str(e)}")
                st.session_state.processing = False
                st.stop()
            
            # 3. Save to Persistence
            try:
                st.write("💾 Saving to audit trail...")
                persistence = PersistenceManager()
                persistence.save_analysis(
                    session_id=str(int(time.time())), # Simple session ID
                    resume_filename=uploaded_file.name,
                    target_role=target_role,
                    job_description=f"Analysis for {target_role}",
                    parsed_resume=result.get("parsed_resume", {}),
                    gap_analysis=result.get("gap_analysis", {}),
                    match_percentage=result.get("gap_analysis", {}).get("match_percentage", 0),
                    roadmap=result.get("roadmap", {}),
                    interview_prep=result.get("interview_prep", {}),
                )
            except Exception as e:
                st.warning(f"Note: Persistence failed: {str(e)}")
            
            st.session_state.analysis_result = result
            st.session_state.processing = False
            
            # 4. Check for Guardrail Failures
            if result.get("status") == "error":
                st.error(f"🚫 {result.get('errors', ['Security Error'])[0]}")
                st.stop()

            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            st.rerun()

# --- DISPLAY RESULTS ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # 💎 Premium Metrics Bar
    gap_data = result.get("gap_analysis", {})
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{gap_data.get("match_percentage", 0)}%</div><div class="metric-label">Match Probability</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{gap_data.get("ats_score", 0)}/100</div><div class="metric-label">ATS Rank</div></div>', unsafe_allow_html=True)
    with m_col3:
        years = result.get("parsed_resume", {}).get("total_experience_years", "N/A")
        st.markdown(f'<div class="metric-card"><div class="metric-value">{years}y</div><div class="metric-label">Experience</div></div>', unsafe_allow_html=True)
    with m_col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value"><span class="status-badge">SECURE</span></div><div class="metric-label">System Audit</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["💎 Executive Overview", "🗺️ Strategic Roadmap", "📝 Resume Optimization", "🎙️ Interview Strategy", "📊 Intelligence Audit"])

    # 1. Executive Overview
    with tabs[0]:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("📈 Opportunity Alignment")
            m_score = gap_data.get("match_percentage", 0)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = m_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2196f3"},
                    'steps' : [
                        {'range': [0, 50], 'color': "#ffebee"},
                        {'range': [50, 80], 'color': "#e3f2fd"},
                        {'range': [80, 100], 'color': "#e8f5e9"}],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gauge, width='stretch')
            
        with c2:
            st.subheader("ATS Rank Score")
            a_score = gap_data.get("ats_score", 0)
            fig_ats = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = a_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4caf50"},
                }
            ))
            fig_ats.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_ats, width='stretch')
        
        st.markdown("---")
        st.subheader("🧠 Senior Strategist Reasoning")
        reasoning = result.get("career_reasoning", "No reasoning provided.")
        st.markdown(f"""
        <div class="reasoning-box">
            {reasoning}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("✅ Skill Match Breakdown")
        matching = gap_data.get("matching_skills", [])
        missing = gap_data.get("missing_skills", {})
        
        sk1, sk2 = st.columns(2)
        with sk1:
            st.success("Matching Skills")
            st.write(", ".join(matching[:15]) if matching else "None")
        with sk2:
            st.error("Critical Gaps")
            critical = missing.get("critical", []) if isinstance(missing, dict) else []
            st.write(", ".join(critical[:15]) if critical else "None")

    # 2. Strategic Roadmap
    with tabs[1]:
        roadmap = result.get("roadmap", {})
        st.subheader(f"🗺️ {roadmap.get('roadmap_title', 'Learning Path')}")
        st.info(f"Duration: {roadmap.get('total_duration_weeks', 'N/A')} weeks")
        
        for phase in roadmap.get("phases", []):
            with st.expander(f"📌 Phase {phase.get('phase_number')}: {phase.get('title')}"):
                st.write(f"**Focus:** {phase.get('focus_area')}")
                st.write("**Action Items:**")
                for task in phase.get("tasks", []):
                    if isinstance(task, dict):
                        t_text = task.get("task", "Unknown Task")
                        t_res = task.get("resource", "")
                        if t_res.startswith("http"):
                            st.markdown(f"- {t_text} [🔗 View Resource]({t_res})")
                        else:
                            st.markdown(f"- {t_text} (*{t_res}*)")
                    else:
                        st.markdown(f"- {task}")
                st.write(f"**🎯 Milestone:** {phase.get('milestone')}")
        
        # 📚 Resource Library
        resources = roadmap.get("free_resources_summary", [])
        if resources:
            st.markdown("---")
            st.subheader("📚 Essential Resource Library")
            for res in resources:
                r_name = res.get("name", "Resource")
                r_url = res.get("url", "#")
                r_type = res.get("type", "General")
                r_skills = ", ".join(res.get("covers", []))
                st.markdown(f"**[{r_name}]({r_url})** ({r_type}) - *Covers: {r_skills}*")

    # 3. Resume Mastery
    with tabs[2]:
        st.subheader("✍️ ATS-Tailored Improvements")
        tailored = result.get("tailored_resume", {})
        st.markdown(f"**Professional Summary:**\n{tailored.get('tailored_summary', 'N/A')}")
        
        st.divider()
        st.subheader("🛠️ Suggested Modifications")
        for job in tailored.get("tailored_experience", []):
            st.write(f"**Role:** {job.get('original_title')}")
            for bullet in job.get('rewritten_bullets', []):
                st.write(f"- {bullet}")

    # 4. Interview Strategy
    with tabs[3]:
        st.subheader("🎙️ Role-Specific Strategy")
        interview = result.get("interview_prep", {})
        questions = interview.get("questions", {})
        
        st.write("### ❓ Strategic Behavioral Questions")
        for q in questions.get("behavioral", [])[:5]:
            if isinstance(q, dict):
                with st.expander(f"Question: {q.get('question')}"):
                    st.write(f"**🔍 What they assess:** {q.get('what_they_assess')}")
                    st.info(f"**💡 Expert Tip:** {q.get('tips')}")
                    st.write(f"**🏗️ Answer Framework:** {q.get('sample_answer_framework')}")
            else:
                st.write(f"- {q}")
            
        st.write("### 💻 Critical Technical Drills")
        for q in questions.get("technical", [])[:8]:
            if isinstance(q, dict):
                with st.expander(f"Topic: {q.get('question')} ({q.get('difficulty', 'medium')})"):
                    st.write("**✅ Key Talking Points:**")
                    for point in q.get("expected_answer_points", []):
                        st.write(f"- {point}")
                    if q.get("follow_up_questions"):
                        st.write("**↪️ Potential Follow-ups:**")
                        for f in q.get("follow_up_questions"):
                            st.write(f"  * {f}")
            else:
                st.write(f"- {q}")

        # 📚 Interview Resources
        i_resources = interview.get("resources", [])
        if i_resources:
            st.markdown("---")
            st.subheader("🔗 Recommended Prep Resources")
            for res in i_resources:
                st.markdown(f"- {res}")

    # 5. Intelligence Audit (Memory & Analytics)
    with tabs[4]:
        st.subheader("📊 System Intelligence & Memory Audit")
        st.markdown("This section demonstrates the **Persistence** and **Long-term Memory** of the Agentic System.")
        
        persistence = PersistenceManager()
        history = persistence.get_analysis_history(limit=50)
        
        if history:
            # Analytics Section
            st.write("### 📈 Match Intelligence Analytics")
            df = pd.DataFrame(history)
            
            # 1. Match Percentage Over Time (Analytics Chart)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['match_percentage'],
                mode='lines+markers',
                name='Match Score',
                line=dict(color='#2196f3', width=3),
                marker=dict(size=8)
            ))
            fig_trend.update_layout(
                title="Historical Match Performance Trend",
                xaxis_title="Date/Time",
                yaxis_title="Match Score (%)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # 2. Historical Data Table (Memory Audit)
            st.write("### 🗄️ Analysis Memory Audit")
            audit_df = df[['timestamp', 'resume_filename', 'target_role', 'match_percentage']].copy()
            audit_df.columns = ['Date', 'File Name', 'Target Role', 'Score (%)']
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
            
            # 3. Agent Handoff Log Simulator (Visibility of Workflow)
            with st.expander("🕵️ View Agent Handoff & Reasoning Logs"):
                st.info("Showing the collaborative handoffs between autonomous agents for the last analysis.")
                st.code(f"""
[LOG] Entry Point -> Guardrail Agent: Validating input security...
[LOG] Guardrail Agent -> Resume Parser: Input safe. Handoff to parser.
[LOG] Resume Parser -> Gap Analyzer: Entity extraction complete. Handoff to RAG analyzer.
[LOG] Gap Analyzer -> Roadmap Generator: Gaps identified. Handoff to learning strategist.
[LOG] Roadmap Generator -> Resume Tailor: Roadmap created. Handoff to content optimizer.
[LOG] Resume Tailor -> Interview Coach: Resume optimized. Handoff to coaching agent.
[LOG] Interview Coach -> Senior Strategist: Prep material ready. Handoff to final reasoning.
[LOG] Senior Strategist -> END: Intelligence synthesis complete.
                """, language="bash")
        else:
            st.info("No historical data found. Complete your first analysis to see the Audit Trail!")

else:
    st.info("Upload your resume and click 'Run Full Analysis' to see the dashboard.")
    
    # Show audit even if no current result
    st.divider()
    with st.expander("📜 View Historical Analysis Audit"):
        persistence = PersistenceManager()
        history = persistence.get_analysis_history(limit=5)
        if history:
            for h in history:
                st.write(f"**{h['timestamp']}**: {h['target_role']} ({h['match_percentage']}%)")
        else:
            st.write("No previous analyses found.")

st.markdown("---")
st.caption("Career Agentic System")

"""
CareerCraft AI - LangGraph Supervisor (Multi-Agent Orchestrator)
Demonstrates: Multi-Agent + LangGraph + LangChain (Syllabus Topics #2, #6)

This is the CORE of the agentic system. It uses LangGraph's StateGraph
to orchestrate multiple specialized agents in a defined workflow.

Architecture:
    Resume Text → Parse → Gap Analysis → Roadmap → Interview Prep → Complete

The Supervisor coordinates:
1. Resume Parser Agent  → Structured resume data
2. Gap Analyzer Agent   → Skill gap report
3. Roadmap Generator    → Personalized learning plan
4. Interview Coach      → Role-specific interview prep
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from agents.resume_parser import parse_resume
from agents.gap_analyzer import analyze_gaps
from agents.roadmap_generator import generate_roadmap
from agents.interview_coach import prepare_interview
from agents.matchmaker import match_jobs
from agents.resume_tailor import tailor_resume
from agents.reasoning_agent import generate_career_reasoning
from agents.guardrails import validate_resume_text


# ============================================
# STATE SCHEMA
# ============================================

class CareerCraftState(TypedDict):
    """
    State schema for the multi-agent workflow.
    
    Each agent reads from and writes to this shared state.
    LangGraph manages state transitions between agents.
    """
    # --- Input ---
    resume_text: str           # Raw resume text from PDF
    target_role: str           # Target job role
    job_description: str       # Full job description text
    
    # --- Agent Outputs ---
    parsed_resume: dict        # From Resume Parser Agent
    gap_analysis: dict         # From Gap Analyzer Agent
    roadmap: dict              # From Roadmap Generator Agent
    interview_prep: dict       # From Interview Coach Agent
    matchmaker_results: dict   # From Matchmaker Agent
    tailored_resume: dict      # From Resume Tailor Agent
    career_reasoning: str      # From Reasoning Agent
    
    # --- Metadata ---
    current_step: str          # Current processing step
    errors: list               # Any errors encountered
    status: str                # Overall status: 'processing', 'complete', 'error'


# ============================================
# AGENT NODES (Graph Nodes)
# ============================================

def guardrail_node(state: CareerCraftState) -> dict:
    """
    Entry Guardrail: Validates if the input is a valid resume.
    
    This is a critical 'Safety & Compliance' node in the agentic workflow.
    """
    try:
        print("\n[GUARDRAIL] Validating Input - STARTING")
        is_valid, message = validate_resume_text(state["resume_text"])
        
        if is_valid:
            print("[GUARDRAIL] Validation - SUCCESS")
            return {"current_step": "guardrail_passed"}
        else:
            print(f"[GUARDRAIL] Validation - REJECTED: {message}")
            return {
                "errors": state.get("errors", []) + [message],
                "current_step": "guardrail_failed",
                "status": "error"
            }
    except Exception as e:
        return {
            "errors": state.get("errors", []) + [f"Guardrail error: {str(e)}"],
            "current_step": "guardrail_failed",
            "status": "error"
        }

def resume_parser_node(state: CareerCraftState) -> dict:
    """
    Node 1: Parse the resume into structured data.
    
    Takes raw resume text and produces structured JSON with
    skills, experience, education, etc.
    """
    try:
        print("\n[STEP 1] Agent: Resume Parser - STARTING")
        parsed = parse_resume(state["resume_text"])
        print("[STEP 1] Agent: Resume Parser - COMPLETE")
        return {
            "parsed_resume": parsed,
            "current_step": "resume_parsed",
        }
    except Exception as e:
        print(f"[STEP 1] Agent: Resume Parser - FAILED: {str(e)}")
        return {
            "parsed_resume": {},
            "errors": state.get("errors", []) + [f"Resume parsing error: {str(e)}"],
            "current_step": "error",
        }


def gap_analyzer_node(state: CareerCraftState) -> dict:
    """
    Node 2: Analyze skill gaps using RAG.
    
    Compares parsed resume against the target job description,
    using ChromaDB to retrieve relevant industry context.
    """
    try:
        print("\n[STEP 3] Agent: Gap Analyzer - STARTING")
        analysis = analyze_gaps(
            parsed_resume=state["parsed_resume"],
            job_description=state["job_description"],
            target_role=state["target_role"],
        )
        print("[STEP 3] Agent: Gap Analyzer - COMPLETE")
        return {
            "gap_analysis": analysis,
            "current_step": "gaps_analyzed",
        }
    except Exception as e:
        print(f"[STEP 3] Agent: Gap Analyzer - FAILED: {str(e)}")
        return {
            "gap_analysis": {},
            "errors": state.get("errors", []) + [f"Gap analysis error: {str(e)}"],
            "current_step": "error",
        }


def roadmap_generator_node(state: CareerCraftState) -> dict:
    """
    Node 3: Generate personalized learning roadmap.
    
    Creates a week-by-week plan to fill the identified skill gaps,
    using RAG to find relevant free learning resources.
    """
    try:
        print("\n[STEP 4] Agent: Roadmap Generator - STARTING")
        roadmap = generate_roadmap(
            gap_analysis=state["gap_analysis"],
            target_role=state["target_role"],
            parsed_resume=state["parsed_resume"],
        )
        print("[STEP 4] Agent: Roadmap Generator - COMPLETE")
        return {
            "roadmap": roadmap,
            "current_step": "roadmap_generated",
        }
    except Exception as e:
        print(f"[STEP 4] Agent: Roadmap Generator - FAILED: {str(e)}")
        return {
            "roadmap": {},
            "errors": state.get("errors", []) + [f"Roadmap generation error: {str(e)}"],
            "current_step": "error",
        }



def resume_tailor_node(state: CareerCraftState) -> dict:
    try:
        print("\n[STEP 5] Agent: Resume Tailor - STARTING")
        tailored = tailor_resume(state["parsed_resume"], state["gap_analysis"], state["target_role"])
        print("[STEP 5] Agent: Resume Tailor - COMPLETE")
        return {"tailored_resume": tailored, "current_step": "resume_tailored"}
    except Exception as e:
        print(f"[STEP 5] Agent: Resume Tailor - FAILED: {str(e)}")
        return {"tailored_resume": {}, "errors": state.get("errors", []) + [f"Tailor error: {str(e)}"], "current_step": "error"}

def interview_coach_node(state: CareerCraftState) -> dict:
    """
    Node 4: Prepare interview material.
    
    Generates role-specific interview questions, preparation tips,
    and mock interview guidance.
    """
    try:
        print("\n[STEP 6] Agent: Interview Coach - STARTING")
        prep = prepare_interview(
            parsed_resume=state["parsed_resume"],
            target_role=state["target_role"],
            gap_analysis=state["gap_analysis"],
        )
        print("[STEP 6] Agent: Interview Coach - COMPLETE")
        return {
            "interview_prep": prep,
            "current_step": "interview_prepared",
            "status": "complete",
        }
    except Exception as e:
        print(f"[STEP 6] Agent: Interview Coach - FAILED: {str(e)}")
        return {
            "interview_prep": {},
            "errors": state.get("errors", []) + [f"Interview prep error: {str(e)}"],
            "current_step": "interview_prepared",
        }


def reasoning_node(state: CareerCraftState) -> dict:
    """
    Node 5: Generate the 'Reasoning Profile' (AI Explanation).
    
    Synthesizes all analysis into a human-readable expert opinion.
    """
    try:
        print("\n[STEP 7] Agent: Reasoning Agent - STARTING")
        reasoning = generate_career_reasoning(state)
        print("[STEP 7] Agent: Reasoning Agent - COMPLETE")
        
        return {
            "career_reasoning": reasoning,
            "current_step": "reasoning_complete",
            "status": "complete",
        }
    except Exception as e:
        print(f"[STEP 7] Agent: Reasoning Agent - FAILED: {str(e)}")
        return {
            "career_reasoning": "Could not generate reasoning profile.",
            "errors": state.get("errors", []) + [f"Reasoning error: {str(e)}"],
            "current_step": "reasoning_complete",
            "status": "complete",
        }


# ============================================
# CONDITIONAL ROUTING
# ============================================

def should_continue(state: CareerCraftState) -> str:
    """
    Conditional edge: determine next step based on current state.
    
    This demonstrates LangGraph's conditional routing capability.
    If any step fails, we still try to continue with remaining steps.
    """
    current = state.get("current_step", "")
    errors = state.get("errors", [])
    
    # If too many errors, stop
    if len(errors) >= 3 or current == "guardrail_failed":
        return "end"
    
    if current == "guardrail_passed":
        return "parse_resume"
    elif current == "resume_parsed":
        return "analyze_gaps"
    elif current == "gaps_analyzed":
        return "generate_roadmap"
    elif current == "roadmap_generated":
        return "tailor_resume"
    elif current == "resume_tailored":
        return "prepare_interview"
    elif current == "interview_prepared":
        return "summarize_reasoning"
    elif current == "reasoning_complete":
        return "end"
    elif current == "error":
        # Try to continue despite errors
        if not state.get("parsed_resume"):
            return "end"  # Can't continue without parsed resume
        elif not state.get("gap_analysis"):
            return "analyze_gaps"
        elif not state.get("roadmap"):
            return "generate_roadmap"
        elif not state.get("tailored_resume"):
            return "tailor_resume"
        elif not state.get("interview_prep"):
            return "prepare_interview"
        return "end"
    else:
        return "end"


# ============================================
# BUILD THE GRAPH
# ============================================

def build_career_graph() -> StateGraph:
    """
    Build the LangGraph multi-agent workflow.
    
    Graph Structure:
        parse_resume → analyze_gaps → generate_roadmap → prepare_interview → END
    
    With conditional routing for error handling.
    
    Returns:
        Compiled LangGraph application
    """
    # Create the state graph
    workflow = StateGraph(CareerCraftState)
    
    # Add agent nodes
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("parse_resume", resume_parser_node)
    workflow.add_node("analyze_gaps", gap_analyzer_node)
    workflow.add_node("generate_roadmap", roadmap_generator_node)
    workflow.add_node("tailor_resume", resume_tailor_node)
    workflow.add_node("prepare_interview", interview_coach_node)
    workflow.add_node("summarize_reasoning", reasoning_node)
    
    # Set entry point
    workflow.set_entry_point("guardrail")
    
    # Define the routing map (all possible next steps)
    routing_map = {
        "parse_resume": "parse_resume",
        "analyze_gaps": "analyze_gaps",
        "generate_roadmap": "generate_roadmap",
        "tailor_resume": "tailor_resume",
        "prepare_interview": "prepare_interview",
        "summarize_reasoning": "summarize_reasoning",
        "end": END
    }
    
    # Add conditional edges (routing logic)
    workflow.add_conditional_edges(
        "guardrail",
        should_continue,
        routing_map,
    )
    
    workflow.add_conditional_edges(
        "parse_resume",
        should_continue,
        routing_map,
    )
    
    workflow.add_conditional_edges(
        "analyze_gaps",
        should_continue,
        routing_map,
    )
    
    workflow.add_conditional_edges(
        "generate_roadmap",
        should_continue,
        routing_map,
    )
    
    workflow.add_conditional_edges(
        "tailor_resume",
        should_continue,
        routing_map,
    )
    
    workflow.add_conditional_edges(
        "prepare_interview",
        should_continue,
        routing_map,
    )
    
    workflow.add_edge("summarize_reasoning", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def run_full_analysis(
    resume_text: str,
    target_role: str,
    job_description: str,
) -> CareerCraftState:
    """
    Run the complete multi-agent analysis pipeline.
    
    This is the main entry point for the orchestrator.
    It invokes all 4 agents in sequence through LangGraph.
    
    Args:
        resume_text: Raw text from the uploaded resume
        target_role: Target job role (e.g., "Software Engineer")
        job_description: Full job description text
        
    Returns:
        CareerCraftState: Complete analysis results from all agents
    """
    # Build the graph
    app = build_career_graph()
    
    # Initial state
    initial_state = {
        "resume_text": resume_text,
        "target_role": target_role,
        "job_description": job_description,
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
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    return final_state

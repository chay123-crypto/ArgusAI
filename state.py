from typing import TypedDict

class AgentState(TypedDict):
    df:object
    filepath:str
    key_cols:list
    profiler:dict
    domain_info:dict
    raw_findings:dict
    report:str
    charts:list
    search_queries:list
    search_results:list
    interpreted_findings:dict
    causal_reasoning:dict
    key:object
    mapping_log:dict
    anonymized_df:object 
    out:list
    dashboard:str
    report_saved:bool
    report_approved:bool
    report_feedback:str
    dashboard_approved:bool
    dashboard_feedback:str
    error:str
    waiting_for:str
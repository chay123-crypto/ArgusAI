from IPython.display import FileLink, display
from state import AgentState
from langgraph.graph import END
import json
from langgraph.types import interrupt
import io
from helper import clean_report
import pandas as pd
from tools import run_profiler,domain_analyser,search_queries,web_search,causal_reasoning
from llm import llm
from visuals import deciding_plots,build_dashboard
from report import report_generator,report_maker
from stats import data_analyst,data_analysis
from crypto import anonymiser

def node_profiler(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        filepath=state['filepath']
        df,profiler=run_profiler(filepath)
        return {'df':df.to_json(orient='records'),'profiler':json.dumps(profiler,default=str)}
    except Exception as e:
        print(f'Pipeline aborted at node 1 due to {e}')
        return {'error':str(e)}

def node_analyser(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        profiler=json.loads(state['profiler'])
        domain,key_cols=domain_analyser(llm,profiler)
        return {'domain_info':json.dumps(domain,default=str),'key_cols':json.dumps(key_cols,default=str)}
    except Exception as e:
        print(f'Pipeline aborted at node 2 due to {e}')
        return {'error':str(e)}

def node_anonymiser(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        profiler=json.loads(state['profiler'])
        df=pd.read_json(io.StringIO(state['df']), orient='records')
        mapping_log,anonymized_df,key=anonymiser(llm,profiler,df)
        print("PII COLUMNS DETECTED:", list(mapping_log.keys()))
        return {'mapping_log':json.dumps(mapping_log, default=str),'anonymized_df':anonymized_df.to_json(orient='records'),'key':key}
    except Exception as e:
        print(f'Pipeline aborted at node 3 due to {e}')
        return {'error':str(e)}

def node_data_analyser(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        profiler=json.loads(state['profiler'])
        df=pd.read_json(io.StringIO(state['anonymized_df']), orient='records')
        key_cols=json.loads(state['key_cols'])
        domain_info=json.loads(state['domain_info'])
        finds=data_analysis(df, key_cols, domain_info)
        interpreted_finds=data_analyst(df,llm,key_cols,domain_info,profiler)
        return {'interpreted_findings':json.dumps(interpreted_finds,default=str),'raw_findings':json.dumps(finds,default=str)}
    except Exception as e:
        print(f'Pipeline aborted at node 4 due to {e}')
        return {'error':str(e)}

def node_query(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        domain=json.loads(state['domain_info'])
        internal_findings=json.loads(state['interpreted_findings'])
        queries=search_queries(llm,domain,internal_findings)
        return {'search_queries':json.dumps(queries, default=str)}
    except Exception as e:
        print(f'Pipeline aborted at node 5 due to {e}')
        return {'error':str(e)}

def node_search(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        queries=json.loads(state['search_queries'])
        results=web_search(queries)
        return {'search_results':json.dumps(results, default=str)}
    except Exception as e:
        print(f'Pipeline aborted at node 6 due to {e}')
        return {'error':str(e)}

def node_reasoner(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        domain_info=json.loads(state['domain_info'])
        interpreted_findings=json.loads(state['interpreted_findings'])
        cleaned_results=state['search_results']
        response=causal_reasoning(llm,domain_info,interpreted_findings,cleaned_results)
        return {'causal_reasoning':json.dumps(response, default=str)}
    except Exception as e:
        print(f'Pipeline aborted at node 7 due to {e}')
        return {'error':str(e)}

def node_reporter(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        domain=json.loads(state['domain_info'])
        reasoning=json.loads(state['causal_reasoning'])
        key=state['key']
        feedback=state.get('report_feedback',None)
        mapping_log=json.loads(state['mapping_log'])
        interpreted_findings=json.loads(state['interpreted_findings'])
        report=report_maker(llm,key,mapping_log,reasoning,interpreted_findings,domain,feedback)
        return {'report':clean_report(report)}
    except Exception as e:
        print(f'Pipeline aborted at node 8 due to {e}')
        return {'error':str(e)}
    
def node_plotdecider(state:AgentState):
    if state.get('error'):        
        return {} 
    try:
        key_cols=json.loads(state['key_cols'])
        domain_info=json.loads(state['domain_info'])
        df=pd.read_json(io.StringIO(state['anonymized_df']), orient='records')
        feedback=state.get('dashboard_feedback',None)
        columns=df.columns.to_list()
        findings=json.loads(state['raw_findings'])
        charts=deciding_plots(llm,df,findings,columns,key_cols,feedback)
        return {'charts':json.dumps(charts, default=str)}
    except Exception as e:
        print(f'Pipeline aborted at node 9 due to {e}')
        return {'error':str(e)}

def node_dashboard(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        key_cols=json.loads(state['key_cols'])
        domain=json.loads(state['domain_info'])
        df=pd.read_json(io.StringIO(state['anonymized_df']), orient='records')
        inputs=json.loads(state['charts'])
        key=state['key']
        mapping_log=json.loads(state['mapping_log'])
        print(f"[node_dashboard] Attempting to build dashboard with {len(inputs)} charts")
        dashboard=build_dashboard(df,key_cols,inputs,key,mapping_log,domain)
        print(f"[node_dashboard] Dashboard built successfully")
        return {'dashboard':dashboard} 
    except Exception as e:
        error_msg = f'Dashboard generation failed: {str(e)}'
        print(f'[node_dashboard] Pipeline aborted: {error_msg}')
        return {
            'dashboard': None,
            'dashboard_available': False,
            'dashboard_error': error_msg
        }

def node_reportgen(state:AgentState):
    if state.get('error'):        
        return {}
    try:
        report=state['report']
        report_generator(report,"outputs/report.pdf")
        display(FileLink("outputs/report.pdf"))
        return {'report_saved':True}
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        print(f'Pipeline aborted at node 11 due to {e}')
        return {'error':str(e)}
    
def human_inloop_report(state):
    if state.get('error'):
        return {}
    return {}   

def human_inloop_dashboard(state):
    if state.get('error'):
        return {}
    return {}   

def should_proceed_report(state):
    if state.get('report_approved')== True:
        return "node_9"
    return "revise_report"

def should_proceed_dashboard(state):
    if state.get('dashboard_approved')== True:
        return "node_10"
    return "revise_charts"
      
def check_error(state: AgentState):
    if state.get('error'):
        return 'end'
    else:
        return 'continue'

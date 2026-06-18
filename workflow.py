from langgraph.graph import StateGraph,END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import AgentState
from agent import node_profiler,node_analyser,node_anonymiser,node_data_analyser,node_query,node_search,node_reasoner,node_reporter,node_plotdecider,node_dashboard,node_reportgen,human_inloop_report,human_inloop_dashboard,should_proceed_dashboard,check_error,should_proceed_report

graph=StateGraph(AgentState)

graph.add_node("node_1",node_profiler)
graph.add_node("node_2",node_analyser)
graph.add_node("node_3",node_anonymiser)
graph.add_node("node_4",node_data_analyser)
graph.add_node("node_5",node_query)
graph.add_node("node_6",node_search)
graph.add_node("node_7",node_reasoner)
graph.add_node("node_8",node_reporter)
graph.add_node("node_9",node_plotdecider)
graph.add_node("node_10",node_dashboard)
graph.add_node("node_11",node_reportgen)
graph.add_node("check_report",human_inloop_report)
graph.add_node("check_dashboard",human_inloop_dashboard)

graph.set_entry_point("node_1")
graph.add_edge("node_2", "node_3")
graph.add_edge("node_3", "node_4")
graph.add_edge("node_4", "node_5")
graph.add_edge("node_5", "node_6")
graph.add_edge("node_6", "node_7")
graph.add_edge("node_7", "node_8")
graph.add_edge("node_8","check_report")
graph.add_edge("node_9", "check_dashboard")
graph.add_edge("node_10","node_11")
graph.add_edge("node_11", END)

graph.add_conditional_edges(
    "node_1",
    check_error,
    {'continue': 'node_2', 'end': END}
)
graph.add_conditional_edges(
    "check_report",
    should_proceed_report,
    {
        "node_9": "node_9",
        "revise_report": "node_8",
    }
)
graph.add_conditional_edges(
    "check_dashboard",
    should_proceed_dashboard,
    {
        "node_10": "node_10",
        "revise_charts": "node_9",
    }
)
conn = sqlite3.connect(
    "checkpoints.db",
    check_same_thread=False
)

memory = SqliteSaver(conn)

pipeline = graph.compile(
    checkpointer=memory,
    interrupt_before=["check_report", "check_dashboard"],
)
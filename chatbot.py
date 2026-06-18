from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from llm import chat_llm,llm
from config import MAX_HISTORY

def chatbot_followup(result,chat_llm,chat_history,user_message):
    chat_prompt=f"""You are an analyst assistant. Answer questions based ONLY on this analysis:

Domain: {result['domain_info']}
Key Metrics: {result['key_cols']}
Raw Findings:{result['raw_findings']}
Findings: {result['interpreted_findings']}
Causal Reasoning: {result['causal_reasoning']}
Report: {result['report']}
Charts : {result['charts']}
Rules:
- Only use numbers that exist in the context above
- If something isn't in the context, say so
- Be concise and specific"""
    if len(chat_history)==0:
        greet_prompt=f"""On startup, Greet the user with a warm tone,and provide 2 to 3 lines summary of of what data you analysed and how you can help the user.
        Be friendly and concise and ask if they have questions.Instruct the user to type 'exit' if they wan to leave the chat."""
        new_prompt=greet_prompt+chat_prompt
        greeting=llm.invoke([
            (SystemMessage(content=new_prompt)),(HumanMessage(content='hi'))
        ])
        print(f"\nArgusAI : {greeting.content}\n")
        chat_history.append(AIMessage(content=greeting.content))
        if user_message.strip() == "startup_greeting":
            return greeting.content
    EXIT_PHRASES = {"exit", "bye", "goodbye", "quit", "thanks", "thank you", "done"}
    if user_message.strip() in EXIT_PHRASES:
        return "Happy to help you with your analysis!"
    chat_history.append(HumanMessage(content=user_message))
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]
    messages=[SystemMessage(content=chat_prompt)]+chat_history
    bot_reply=chat_llm.invoke(messages)
    print(f"\nArgusAI : {bot_reply.content}\n")
    chat_history.append(AIMessage(content=bot_reply.content)) 
    return bot_reply.content
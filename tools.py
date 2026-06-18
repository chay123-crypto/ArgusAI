import pandas as pd
import ast
import json
import os
from tavily import TavilyClient
from langsmith import traceable
from helper import retry, safe_parse

@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def run_profiler(filepath):
    encodings=['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df=None
    for encoding in encodings:
        try:
            df=pd.read_csv(filepath, encoding=encoding)
            print(f"Successfully read with encoding:{encoding}")
            break
        except UnicodeDecodeError:
            continue 
    if df is None:
        raise ValueError("Could not read file with any encoding")
    likely_ids=[]
    likely_dates=[]
    likely_categorical=[]
    likely_numeric=[]

    for col in list(df.columns):
        if df[col].dtype==object:
            converted=pd.to_datetime(df[col],errors="coerce",infer_datetime_format=True)
            if converted.notna().mean()>0.6: 
                likely_dates.append(col)
                continue
            numeric_converted = pd.to_numeric(df[col], errors="coerce")
            success_rate = numeric_converted.notna().sum() / len(df)
            if success_rate > 0.8:
                likely_numeric.append(col)
                continue
            unique=df[col].nunique()
            if unique>=0.9*len(df):
                likely_ids.append(col)
                continue
            if unique < 100: 
                likely_categorical.append(col)
            numeric_converted=pd.to_numeric(df[col], errors="coerce")
        elif pd.api.types.is_numeric_dtype(df[col]):
            likely_numeric.append(col)

    profiler={
        'shape':df.shape,
        'dtypes':df.dtypes.astype(str).to_dict(),
        'columns':df.columns.tolist(),
        'null_counts':df.isnull().sum().to_dict(),
        "null_percent":(df.isnull().mean() * 100).round(2).to_dict(),
        "numeric_stats":{col: {str(k): v for k, v in stats.items()} for col, stats in df.describe().to_dict().items()},
        "cardinality":df.nunique().to_dict(),
        "duplicates":int(df.duplicated().sum()),
        "skewness":{k: round(v, 3) for k, v in df.skew(numeric_only=True).items()},
        "kurtosis":{k: round(v, 3) for k, v in df.kurt(numeric_only=True).items()},
        "likely_date_columns":likely_dates,
        "likely_categorical_columns": likely_categorical,
        "likely_numeric_columns":likely_numeric,
        "likely_id_columns":likely_ids,
        "sample_rows":df.head(1).to_dict()
    }
    for col in df.columns:
        if df[col].dtype=='object':
            df[col]=df[col].fillna('Unknown')
        elif pd.api.types.is_numeric_dtype(df[col]):
            if abs(df[col].skew())<=1:
                df[col]=df[col].fillna(df[col].mean())
            else:
                df[col]=df[col].fillna(df[col].median())
    return df,profiler

@traceable(name='domain_analyser')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def domain_analyser(llm,profile):
    dtypes=profile['dtypes']
    all_cols=profile['columns']

    imp_cols=[c for c in all_cols if dtypes.get(c) in ('int64','float64','int32','float32') and not any(x in c.lower() for x in ['key','code','num','index','date','time','year','month'])]
    imp_cols=imp_cols[:30] if len(imp_cols)>30 else imp_cols
    prompt=f"""You are a senior business data analyst. Analyze the CSV profile and return a single valid JSON object.

RULES:
- You must never leave the Key_Metrics_to_watch output empty.
- For finding Key_Metrics_to_watch,column names given to you should only be taken, do not take any other column names not in the profile.
- Return ONLY valid JSON. No markdown, no backticks, no explanation.
- Use exact key names as specified below.
- If columns are named V1, V2, ..., Vn alongside 'Amount' and 'Class' or 'Time', this is likely finance_transactions (PCA-transformed fraud detection data)
- Do not add any keys not listed below.

OUTPUT SCHEMA:
{{
  "Domain": "<one of: retail_sales | manufacturing_quality | manufacturing_production | manufacturing_maintenance | hr_attrition | hr_performance | finance_transactions | logistics | other>",
   "Subdomain": "<if Domain is 'other', give a 2-3 word label like 'clinical_outcomes' | 'student_performance' | 'crime_statistics' | 'energy_consumption' | 'environmental_monitoring' | 'sports_analytics' | 'other_unknown' etc.>",
  "Confidence_Score": "<one of: High | Medium | Low>",
  "Key_Metrics_to_watch": ["<col1>", "<col2>", "..."],
  "Metric_Direction": {{
    "<col1>": "<higher_is_better | lower_is_better>",
    "<col2>": "<higher_is_better | lower_is_better>"
  }},
  "Reason": "<exactly 2 sentences explaining your domain choice>"
}}

CONSTRAINTS:
- Key_Metrics_to_watch: only include numeric columns that are meaningful business metrics. Exclude ID columns, date columns, and free-text fields.
- Metric_Direction: must have one entry for every column in Key_Metrics_to_watch, no more, no less.

COLUMN NAMES: {imp_cols}

underscores not spaces. No variation allowed.
    """
    response=llm.invoke(prompt)
    response=safe_parse(response.content)
    key_cols=response['Key_Metrics_to_watch']
    return response,key_cols

def critical_info(llm,profiler):
    columns=profiler['columns']
    prompt=f"""You are a data privacy analyst.

You are given a list of column names from a business dataset: {columns}

Your task: identify which columns contain Personally Identifiable Information (PII) 
that must be anonymized before sending data to an external system.

═══════════════════════════════
FLAG THESE (PII):
═══════════════════════════════
- Person names (employee, operator, customer)
- Employee IDs or operator IDs
- Email addresses, phone numbers
- National IDs, passport numbers
- Batch or reference codes directly traceable to a specific individual

═══════════════════════════════
DO NOT FLAG THESE (not PII):
═══════════════════════════════
- Machine IDs, equipment codes, production line numbers
- Temperatures, pressures, or any sensor/operational metrics
- Timestamps, dates, shift codes
- Product names, SKUs, categories
- Numeric aggregates (counts, averages, totals)

═══════════════════════════════
OUTPUT
═══════════════════════════════
A valid Python list of column names that are PII. Nothing else.
If no PII columns exist, return an empty list: []

Example: ['EmployeeName', 'OperatorID', 'CustomerEmail']
"""
    response=llm.invoke(prompt)
    try:
        return ast.literal_eval(response.content.strip())
    except:
        return []

@traceable(name='search_queries')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def search_queries(llm,domain,internal_findings): 
    focus=internal_findings['investigation_focus']
    issues=internal_findings['critical_issues']
    issue_details = []
    for i, issue in enumerate(issues): 
        issue_details.append(f"""
  - Issue {i+1}: {issue.get('issue', 'Unknown')}
  - Severity: {issue.get('severity', 'Unknown')}
  - Evidence: {issue.get('evidence', 'Unknown')}
  - Why it matters: {issue.get('potential_harm', 'Unknown')}
""")
    prompt=f"""You are a business research analyst tasked with generating highly specific web search queries.
CONTEXT:
Your goal is to find ROOT CAUSES and PRACTICAL EXPLANATIONS for these specific data quality issues:
-DOMAIN: {domain['Domain']}

CRITICAL ISSUES TO RESEARCH:
{''.join(issue_details)}

INVESTIGATION FOCUS AREAS:
{json.dumps(focus[:3], indent=2)}

GOOD EXAMPLES FOR YOUR ISSUES:
For Outliers in Quantity:
"Why do promotional bulk orders create extreme outliers e-commerce inventory"
"Detecting real bulk orders vs data entry errors retail systems"
"Supply chain disruptions causing quantity outliers 2024"

OUTPUT REQUIREMENTS:
1. Each query MUST be a single line, enclosed in double quotes
2. NO bullet points, NO markdown, NO explanations
3. Return ONLY the queries, one per line

QUERY DESIGN RULES (MANDATORY):
1. SPECIFIC ROOT CAUSES, not symptoms
   ❌ BAD: "retail sales issues"
   ✅ GOOD: "Why do bulk orders cause high quantity outliers e-commerce"

2. Include mechanism/process, not just the problem
   ❌ BAD: "missing customer IDs"
   ✅ GOOD: "Guest checkout flows that bypass customer ID capture"

3. Target REAL-WORLD IMPLEMENTATIONS, not academic theory
   ❌ BAD: "customer data collection best practices"
   ✅ GOOD: "Shopify/WooCommerce guest checkout missing customer field 2024"

4. Include INDUSTRY CONTEXT (retail, e-commerce, data quality, etc.)
   ❌ BAD: "Why do systems fail"
   ✅ GOOD: "Root cause of duplicate invoice numbers in retail POS systems"

5. Include YEAR or RECENCY if applicable
   ❌ BAD: "data entry errors"
   ✅ GOOD: "Data validation failures retail 2024 industry case studies"

ANTI-PATTERNS (Avoid these):
- Generic statistics ("retail trends", "market analysis")
- Textbook definitions ("what is data quality")
- Generic best practices ("how to prevent errors")
- Vague terminology ("unusual patterns", "data issues")

GOOD EXAMPLES FOR YOUR ISSUES:
For Outliers in Quantity:
"Why do promotional bulk orders create extreme outliers e-commerce inventory"
"Detecting real bulk orders vs data entry errors retail systems"
"Supply chain disruptions causing quantity outliers 2024"

For Missing CustomerID:
"Why guest checkout abandons customer ID field collection e-commerce"
"Shopify/WooCommerce missing customer data rates implementation"
"How to enforce customer ID capture checkout validation rules"

Strictly output only a valid JSON array like this:
[
  "Query 1 here",
  "Query 2 here",
  "Query 3 here",
  "Query 4 here",
  "Query 5 here",
  "Query 6 here"
]
"Return ONE single flat JSON array containing ALL queries together. Do NOT group by issue. Do NOT return multiple arrays."
No other text. No markdown. No backticks."""

    response=llm.invoke(prompt)
    try:
        content=response.content 
        return safe_parse(content)  
    except Exception as e:
        print(f"Error: {e}")
        return []

def clean_results(search_results):
    extracted=[]
    for result in search_results:
        contents=[r['content'] for r in result['response']['results']]
        extracted.append(
            {
                'query':result['query'],
                'content':contents
            }
        )
    return extracted

@traceable(name='web_search')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def web_search(queries):
    results=[]
    search_client=TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    for query in queries:
        result=search_client.search(query=query.strip(),max_results=3,search_depth="basic")
        results.append({'query':query,'response':result})
    return clean_results(results)

@traceable(name='causal_reasoning')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def causal_reasoning(llm,domain_info,interpreted_findings,cleaned_results):
    internal={
        'critical_issues':interpreted_findings['critical_issues'],
        'root_cause_hypotheses':interpreted_findings['root_cause_hypotheses'],
        'executive_summary':interpreted_findings['executive_summary']
    }
    external_block=json.dumps(cleaned_results, indent=2) if cleaned_results else "NO EXTERNAL RESEARCH AVAILABLE"
    prompt=f"""You are a causal inference specialist. Evaluate whether the available evidence supports,
partially supports,contradicts or is insufficient to infer causality. Never state a causal relationship as fact unless both
internal evidence and external evidence support it.
INPUTS:
Domain: {domain_info}
Internal Findings: {json.dumps(internal, indent=2, default=str)}
External Research: {external_block}

FOR EACH critical issue, complete all 4 gates:

GATE 1 — FACT: State exact metric + value + unit + % of total. No vague words (high/notable/significant banned).

GATE 2 — MECHANISM: [External Factor] → [Specific Process] → [Internal Effect]. 
If the intermediate process is unknown,
describe the uncertainty explicitly and reduce confidence accordingly.
Do not invent missing process steps.

GATE 3 — VARIANCE: Only assign numerical percentages if the internal findings,
external evidence, or domain knowledge explicitly supports them.
Otherwise use:
"primary":"Calibration drift — dominant contributor (percentage unknown)"
"secondary":"Operator delay — secondary contributor (percentage unknown)"
"unexplained":"Requires maintenance logs"
Never invent percentages.

GATE 4 — FALSIFIABILITY: One specific data point that confirms. One that refutes.

DOMAIN CAPS (automatic confidence limits):
- Geographic concentration >70%: MEDIUM cap unless regional expansion data present
- Missing data >20%: MEDIUM cap until source audited  
- Outliers present: MEDIUM cap without transaction logs

CITATION RULE: For external_evidence, you MUST copy a verbatim phrase from the research above and name the query it came from.
Format: "Source query: '<query>' | Finding: '<copied phrase>'"
If external_block is empty or NO EXTERNAL RESEARCH AVAILABLE, set external_evidence=null and verdict=no_external_evidence for all issues.

RULES:
- Every percentage has a source tag: (industry benchmark) / (pilot data) / (estimate pending validation). Example: 82.37% → 65.90% (20% reduction) and 186.51 → 130.56 (30% reduction) is found in the report - these must be backed up with valid ecternal citations like
industry benchmarks or other target metrics found on external research. If a claim percentage has no source compulsorily omit the percentage claim instead say something like requires baseline period data to set target.
- external_evidence must quote a specific finding from research above, or set null
- All JSON string values single-line (no literal newlines inside strings)
-If internal evidence and external research disagree preserve both,explain why. Do NOT average them and set verdict="contradicted".
- If external research is unavailable, set verdict="no_external_evidence" and keep all fields except verdict null

OUTPUT:
{{
  "causal_connections": [
    {{
      "issue": "<label + key metric>",
      "internal_evidence": "<exact metric, value, unit, n>",
      "external_evidence":{{"source_query":"","quoted_finding":"","relevance":"Explain in one sentence how this research relates to the internal issue."}},
      "causal_mechanism": "<[Factor] → [Process] → [Effect] in [timeframe]>",
      "variance_decomposition": {{
        "primary": "<cause: X%> — <one-line justification>",
        "secondary": "<cause: Y%> — <justification>",
        "unexplained": "<Z%> — <what data resolves this>"
      }},
      "confidence": "<HIGH|MEDIUM|LOW>: <one sentence citing variance band>",
      "confirmation_test": "<specific data point>",
      "refutation_test": "<specific data point>",
      "verdict": "<supported|contradicted|no_external_evidence>"
    }}
  ],
  "overall_diagnosis": "<3 sentences max: dominant causal story, confidence levels, highest uncertainty link>",
  "meta_assessment": {{
    "binding_constraint": "<single data gap most limiting analysis>",
    "validation_sequence": ["<step 1>", "<step 2>", "<step 3>"]
  }},
  "evidence_strength": {{
    "internal": "<Strong/Moderate/Weak>: <one sentence justification>",
    "external": "<Strong/Moderate/Weak>: <one sentence justification>",
    "overall": "<Strong/Moderate/Weak>: <one sentence justification>"
  }}
}}"""
    response=llm.invoke(prompt)
    return safe_parse(response.content)


import numpy as np
from scipy import stats
import pandas as pd
from helper import sanitize_keys,safe_parse,retry
from langsmith import traceable
import json
from scipy import stats

def outlier_detection(df,key_cols):
    findings={}
    for col in key_cols:
        if col in df.select_dtypes(include=np.number).columns:
            if df[col].std()==0:
                continue
            z_score=abs(stats.zscore(df[col].dropna()))
           
            findings[col]={
                 'counts':int((z_score>3).sum()),
                  'max_zscore':round(float(z_score.max()),3)
            }
            if findings[col]['counts']>0:
                pass
            else:
                del findings[col]
    return findings

def trend_detection(df,key_cols):
    date_cols=[c for c in df.columns.to_list() if 'date' in c.lower()]
    if not date_cols:
        return {}
    new_df=df.copy()
    new_df[date_cols[0]]=pd.to_datetime(new_df[date_cols[0]])
    new_df=new_df.sort_values(date_cols[0])

    findings={}
    for col in key_cols:
        if col in new_df.select_dtypes(include=np.number).columns:
            y=new_df[col].dropna().values
            slope,_,r,_,_=stats.linregress(np.arange(len(y)),y)
            epsilon=1e-5
            if abs(slope)<=epsilon :
                    findings[col]={
                         'trend':'NIL',
                        'value':round(float(slope),3),
                          'Correlation':round(float((r)**2),3)
                    }
            elif slope<0 :
                    findings[col]={
                         'trend':'Negative',
                          'value':round(float(slope),3),
                          'Correlation':round(float((r)**2),3)
                    }
            elif slope>0 :
                    findings[col]={
                         'trend':'Positive',
                        'value':round(float(slope),3),
                          'Correlation':round(float((r)**2),3)
                    }
    return findings

def correlation_analysis(df):
    pos=[]
    corr=df[df.select_dtypes(include=np.number).columns].corr()
    for i in range(len(corr.columns)):
        for j in range(i+1,len(corr.columns)):
            value=corr.iloc[i,j]
            if abs(value)>=0.5:
                pos.append(
                    {
                        'col1':corr.columns[i],
                        'col2':corr.columns[j],
                        'corr':round(float(value),4),
                        'degree':'strong' if abs(value)>=0.7 else 'moderate',
                        'warning':'both columns might mean the same' if abs(value)>0.95 else None
                    }
                )
    return pos

def performer_rank(df,key_cols,domain_info):
    performer={}
    for cat_col in df.select_dtypes(include='object').columns:
        for metric in key_cols:
            if not metric in df.columns:
                continue
            if not np.issubdtype(df[metric].dtype, np.number):
                continue
            try:
                grouped=df.groupby(cat_col)[metric].agg(
                    ['mean','std','count'])
                direction=domain_info['Metric_Direction'].get(metric,"lower_is_better")
                if direction=="lower_is_better":
                    worst=grouped['mean'].idxmax()
                    best=grouped['mean'].idxmin()
                else:
                    worst=grouped['mean'].idxmin()
                    best=grouped['mean'].idxmax()
                performer[f"{cat_col}_{metric}"] = {
                    'worst':str(worst),
                    'worst_value': round(
                        grouped.loc[str(worst), 'mean'], 4
                    ),
                    'best':str(best),
                    'best_value':round(
                        grouped.loc[str(best),'mean'], 4
                    ),
                    'gap':round(
                        grouped['mean'].max()-grouped['mean'].min(),4)
                }
            except Exception as e:
                print(f"Skipping {cat_col}_{metric}: {e}")
                continue
    return performer
    

def percentage_growth(df,key_cols):
    growth={}
    date_cols=[c for c in df.columns.to_list() if 'date' in c.lower()]
    if not date_cols:
        return {}
    new_df=df.copy()
    new_df[date_cols[0]]=pd.to_datetime(new_df[date_cols[0]])
    new_df=new_df.sort_values(date_cols[0])
    for cat_col in new_df.select_dtypes(include='object').columns:
        for metric in key_cols:
            if not pd.api.types.is_numeric_dtype(new_df[metric]):
                continue
            if metric in new_df.columns:
                grouped=new_df.groupby(cat_col)[metric].mean()
                growth[f'{cat_col}-{metric}']={
                    'percent_growth':round(((grouped.iloc[-1]-grouped.iloc[0])/grouped.iloc[0]*100 if grouped.iloc[0]!=0 else 0),2)}
    return growth

def segmentation_analysis(df,key_cols):
    findings={}
    cat_cols=df.select_dtypes(include='object').columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    for cat_col in cat_cols:
        findings[cat_col]={}
        for metric in key_cols:
            if metric in df.columns:
                if np.issubdtype(df[metric].dtype, np.number):
                    grouped=df.groupby(cat_col)[metric].mean()
    
                    findings[cat_col][metric]={
                        'best':str(grouped.idxmax()),
                        'worst':str(grouped.idxmin())
                    }
    return findings

def distribution_analysis(df,key_cols):
    num_data=df[df.select_dtypes(include=np.number).columns]
    findings={}
    for col in num_data.columns:
        if col in key_cols:
            data=num_data[col].dropna()
            sample=data.sample(min(len(data),5000),random_state=42)
            _,pvalue=stats.shapiro(sample)
            skew=float(data.skew())
            kurt=float(data.kurt()) 
            if pvalue>0.05:
               label='normal'
            elif kurt > 3:
               label='heavy_tailed'
            elif skew>1:
               label='right_skewed'
            elif abs(skew)<0.5:
               label='symmetric'
            elif skew<-1:
               label = 'left_skewed'
            elif skew>0:
               label='slightly_right_skewed'
            else:
               label='slightly_left_skewed'
            findings[col]={
                'skew':skew,
                'distribution':label,
                'p_value':round(float(pvalue),2),
                'n':len(data)
            }
    return findings

def concentration_analysis(df,key_cols):
    findings={}
    cat_cols=df.select_dtypes(include='object').columns
    
    for cat_col in cat_cols:
        for metric in key_cols:
            if metric not in df.columns:
                continue
            if not np.issubdtype(df[metric].dtype, np.number):
                continue 
                
            grouped=df.groupby(cat_col)[metric].sum()
            total=grouped.sum()
            if total==0:
                continue
            
            top_contributor=grouped.idxmax()
            top_pct=round(float(grouped.max()/total*100),2)
            
            findings[f"{cat_col}_{metric}"]={
                'top_contributor':str(top_contributor),
                'contribution_pct':top_pct,
                'is_concentrated':top_pct>50}
    
    return findings
    
def pareto_summary(concentration_findings):
    critical={
        k:v for k, v in concentration_findings.items()
        if v['contribution_pct']>80}
    return critical

def data_analysis(df,key_cols,domain_info):
    findings={}
    findings['outlier_detection']=outlier_detection(df,key_cols)
    findings['segmentation_analysis']=segmentation_analysis(df,key_cols)
    findings['percentage_growth']=percentage_growth(df,key_cols)
    #findings['performer_rank']=performer_rank(df,key_cols,domain_info)
    findings['correlation_analysis']=correlation_analysis(df)
    findings['trend_detection']=trend_detection(df,key_cols)
    findings['concentration_analysis']=concentration_analysis(df,key_cols)
    findings['distribution_analysis']=distribution_analysis(df,key_cols)
    findings['pareto_summary']=pareto_summary(findings['concentration_analysis'])
    findings=sanitize_keys(findings)
    return findings

@traceable(name='data_analyst')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def data_analyst(df,llm,key_cols,domain_info,profiler):
    slim={
    'shape': profiler['shape'],
    'columns': profiler['columns'],
    'null_percent': profiler['null_percent'],
    'likely_date_columns': profiler['likely_date_columns'],
    'likely_categorical_columns': profiler['likely_categorical_columns'],
    'likely_id_columns': profiler['likely_id_columns'],
    'sample_rows': profiler['sample_rows'],
    }
    finds=data_analysis(df,key_cols,domain_info)
    prompt=f"""You are a senior data analyst at a large enterprise company.
You are given two inputs:
1. A dataset profile (schema, stats, nulls, etc.)
2. Pre-computed statistical findings from analysis functions

Your job is to INTERPRET these findings in clear business language for a non-technical manager.

═══════════════════════════════
STRICT RULES — READ BEFORE ANYTHING ELSE
═══════════════════════════════
- ONLY reference numbers that exist in the findings JSON below. 
  Do NOT invent, estimate, round, or extrapolate any value.
- If a field has no supporting data, output null. Never fill gaps with guesses.
- Do NOT use: "approximately", "likely around", "seems to suggest", "may indicate"
  unless you are directly quoting a finding.
- Do NOT reference columns, metrics, or entities not present in the inputs.
- Your job is interpretation, NOT recalculation. Trust every number as-is.

═══════════════════════════════
INPUTS
═══════════════════════════════
Domain: {domain_info['Domain']}
Key Metrics: {json.dumps(key_cols, default=str)}
Dataset Profile: {json.dumps(slim, default=str)}
Statistical Findings: {json.dumps(finds, default=str)}

═══════════════════════════════
OUTPUT — strict JSON, exact keys below
═══════════════════════════════
{{
  "executive_summary": "3-4 sentences. State the domain, the 2-3 biggest issues with exact numbers from findings, and one immediate action. No vague language.",

  "critical_issues": [
    {{
      "issue": "Short title",
      "evidence": "Exact value from findings — e.g. z-score of X in column Y",
      "cause": "Specific mechanism, not a generic statement",
      "severity": "High | Medium | Low",
      "potential_harm": "Concrete business consequence"
    }}
    // exactly 5 issues, ranked High to Low
  ],

  "root_cause_hypotheses": {{
    "issue_title": "One specific hypothesis per issue grounded in the data"
    // one key per critical issue above
  }},

  "recommended_actions": [
    {{
      "action": "Specific action title",
      "target_issue": "Which critical issue this addresses",
      "steps": ["Step 1", "Step 2"],
      "success_metric": "Measurable outcome with a number"
    }}
    // 1-2 actions per issue max
  ],

  "investigation_focus": [
    "Specific question for external research — must name a column or metric from findings",
    "Specific question 2",
    "Specific question 3"
  ]
}}
"cause": "Name the specific system/process failure that produced this — format: '[System/Process] failed to [action] because [reason]'. NOT a restatement of the issue.",

"root_cause_hypotheses": {{
  "<issue_title>": "One level deeper than cause. Format: 'If [specific condition] exists in the [system/process], it would produce [observed effect] because [mechanism]'"
}}
"investigation_focus": [
  "Must reference a specific column name AND a specific value from findings.",
  "Must be a causal question, not a descriptive one.",
  "❌ 'What causes missing data?' ✅ 'Why do 24.93% of CustomerID values go uncaptured in retail POS transactions under £10?'"
]

Return only valid JSON. No markdown, no backticks, no explanation outside the JSON.
"""
    response=llm.invoke(prompt)
    return safe_parse(response.content)
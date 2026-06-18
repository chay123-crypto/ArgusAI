import numpy as np
import markdown
import pdfkit
import json
from langsmith import traceable
from helper import retry,clean_report
from crypto import decryption

@traceable(name='report_maker')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def report_maker(llm, key, mapping_log, reasoning, internal_finding, domain, feedback=None):
    decrypt_map = decryption(key, mapping_log)
    domain_label=domain['Domain']
    if domain_label=='other' and domain.get('Subdomain',"NIL"):
        domain_label=f"other ({domain['Subdomain']})"
    
    feedback_section=f"\nREVISION FEEDBACK: {feedback}\nAddress this feedback specifically in the revised report.\n" if feedback else ""

    prompt=f"""You are a BI analyst writing a data quality report for C-suite executives.

INPUTS:
Domain Label:{domain_label}
Domain: {domain}
Internal Findings: {json.dumps(internal_finding,indent=2,default=str)}
Causal Reasoning: {json.dumps(reasoning, indent=2,default=str)}
{feedback_section}

MANDATORY: The Causal Reasoning input contains external_evidence fields with real research citations.
You MUST use these in the Root Cause Analysis section.
For each issue, cite the external_evidence exactly like this:
"[External Research: <source query> — <finding>]"
If you do not cite external evidence for at least 3 issues, your report FAILS quality gates.

RULES (apply to every sentence):
1. Every claim cites an exact number from findings. No vague words (high/notable/significant/improve/analyze).
2. Every percentage has a source tag: (industry benchmark) / (pilot data) / (estimate pending validation). Example: 82.37% → 65.90% (20% reduction) and 186.51 → 130.56 (30% reduction) is found in the report - these must be backed up with valid ecternal citations like
industry benchmarks or other target metrics found on external research. If a claim percentage has no source compulsorily omit the percentage claim instead say something like requires baseline period data to set target.
3. Every action has: Owner, Timeline, Phases, Success Metric with current→target numbers.
4. Root causes explain WHY via mechanism, not just WHAT. Link: Issue → Cause → Impact → Action → Outcome.
5. Severity: HIGH=blocks analytics/breaks systems | MEDIUM=reduces accuracy | LOW=optimization.
6. No invented numbers. Only use values present in findings.
7. Be decisive: "Primary driver is X" not "may be caused by X".
8. Word count: MINIMUM 1500 words. Count before submitting.
9.When computing target from current→target, use: target = current × (1 - reduction%). Do not approximate.
10.Integrate external research findings as natural prose, never as raw citation brackets.

OUTPUT — return ONLY this markdown structure, no preamble, no code blocks:

# Executive Summary
[250-300 words. Top 5 issues with exact evidence + 1 key action with owner/timeline/target.]

# Critical Issues
## [Issue Name]
- Evidence: [exact metric, value, unit, n]
- Root Cause: [specific mechanism with supporting data]
- Impact: [quantified business consequence]
- Severity: [HIGH/MEDIUM/LOW—one sentence justification]

[Repeat for each issue]

# Root Cause Analysis
[Per issue: mechanism→evidence ruling out alternatives → business impact link]

# Recommended Actions
[Max 5 actions, ordered by priority]
**Action N: [Title]** (Owner: [Team], Timeline: [X days], Priority: HIGH/MEDIUM)
- Objective: [problem solved + current→target metric]
- Phase 1 ([X days]): [specific step]
- Phase 2 ([X days]): [specific step]
- Phase 3 ([X days]): [specific step]
- Success Metric: [exact number current→target + source]
- Resource Needs: [roles + hours]

# Data Gaps and Next Steps
- [Data needed] (Owner: [Team],[X days]):Impact—[what decision this unlocks]

EXAMPLES:

BAD(invented baseline not in findings):
"Success Metric: 10% improvement in model accuracy (from 80% to 88%)"
— WRONG because 80% accuracy appears nowhere in findings. 88% was invented by calculating 80 × 1.10 = 88, but 80 was never a real number.

GOOD(uses only numbers from findings, shows calculation):
"Success Metric: Reduce Amount outliers by 50% (from 4,076 to 2,038)"
— CORRECT. 4,076 comes from outlier_detection findings. Target = 4,076 × (1 - 0.50) = 2,038.

ANOTHER GOOD EXAMPLE:
"Success Metric: Reduce missing CustomerID from 24.93% to 12.47% (50% reduction)"
— CORRECT. 24.93% comes from profiler null_percent. Target = 24.93 × (1 - 0.50) = 12.465 ≈ 12.47%.

RULE: 
- The starting number MUST exist verbatim in findings or profiler.
- The target MUST be calculated as: target = current × (1 - reduction%).
- Always show the calculation explicitly: "current × (1 - X%) = target"
- If the current baseline is unknown, write:
  "Success Metric: Reduce [metric] to within acceptable threshold — baseline to be established in Phase 1."

ANTI-HALLUCINATION (CRITICAL):
- Impact quantification: ONLY use numbers present in findings. If no impact number exists, write "Impact: Unquantified — requires [specific data] to estimate" — DO NOT invent percentages.
- Root cause ruling-out: ONLY cite evidence present in findings. If no counter-evidence exists, omit the ruling-out section entirely.
- A z-score is NOT a count. Never use a z-score as a quantity to reduce. For outlier actions, success metric must be "% of transactions with |z-score| > 3".
- If a metric has no clear business interpretation (e.g. skewness of 186.5, negative % growth in InvoiceNo), flag it as "Requires domain validation before reporting to C-suite" — do not invent a business narrative.
- Do not cite statistics from external research unless they appear verbatim in the causal_reasoning input. Never invent study findings.
EXAMPLE: 

## 30 Days
## 60 Days  
## 90 Days"""
    max_attempts=3
    for attempt in range(max_attempts):
        report=llm.invoke(prompt).content
        word_count = len(report.split())
        print(f"Attempt {attempt+1}: {word_count} words")
        
        if word_count>=1500:
            break
        
        if attempt<max_attempts-1:
            print(f"Report too short ({word_count} words). Regenerating with expansion prompt...")
            prompt+=f"\n\nPREVIOUS ATTEMPT WAS {word_count} WORDS — TOO SHORT. You must write at least 1500 words. Expand root cause analysis and action phases with more specific detail. Do not add filler — add substance."
    else:
        print(f"Warning: Could not reach 1500 words after {max_attempts} attempts. Using last generated report ({word_count} words).")

    for display,real in decrypt_map.items():
        report=report.replace(display,real)  
    return report

@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def report_generator(report,output_path="outputs/report.pdf"):
    report=clean_report(report).replace('\\n', '\n')
    html_body = markdown.markdown(report, extensions=["tables", "fenced_code", "nl2br"])
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
    @page {{
        size: A4;
        margin: 2.5cm 2.2cm;
        @bottom-right {{
          content: "Page " counter(page) " of " counter(pages);
          font-size: 9px; color: #888;
        }}
      }}
      body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; line-height: 1.7; color: #1a1a2e; }}
      h1 {{ font-size: 28px; font-weight: 700; color: #0f1117; margin: 0 0 10px; padding-bottom: 8px; border-bottom: 2px solid #4f46e5; }}
      h2 {{ font-size: 20px; font-weight: 600; color: #1a1d27; margin: 28px 0 8px; border-left: 3px solid #4f46e5; padding-left: 10px; }}
      h3 {{ font-size: 18px; font-weight: 600; color: #333; margin: 16px 0 6px; }}
      p  {{ margin: 0 0 13px; }}
      ul, ol {{ margin: 0 0 10px 20px; }}
      li {{ margin-bottom: 5px; }}
      strong {{ color: #0f1117; }}
      table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 13px; }}
      th {{ background: #1a1d27; color: #fff; font-weight: 600; padding: 8px 12px; text-align: left; }}
      td {{ padding: 7px 12px; border-bottom: 0.5px solid #e0e0e0; }}
      tr:nth-child(even) td {{ background: #f9f9fb; }}
      blockquote {{ border-left: 3px solid #4f46e5; margin: 12px 0; padding: 8px 14px; background: #f0f0ff; color: #444; font-style: italic; }}
      hr {{ border: none; border-top: 1px solid #e0e0e0; margin: 20px 0; }}
    </style>
    </head>
    <body>{html_body}</body>
    </html>"""
    
    options ={
        'page-size': 'A4',
        'margin-top': '2.5cm',
        'margin-right': '2.2cm',
        'margin-bottom': '2.5cm',
        'margin-left': '2.2cm',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }
    config = pdfkit.configuration(wkhtmltopdf='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')
    pdfkit.from_string(html, output_path, options=options, configuration=config)
    return output_path
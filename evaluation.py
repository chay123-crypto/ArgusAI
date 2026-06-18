import re
import json
from cryptography.fernet import Fernet
import pandas as pd

def eval_1(output):
    allowed=['retail_sales','manufacturing_quality','manufacturing_production',
               'manufacturing_maintenance','hr_attrition','hr_performance',
               'finance_transactions','logistics','other']
    scores = {
        'valid_domain':    output['Domain'] in allowed,
        'has_confidence':  output['Confidence_Score'] in ['High','Medium','Low'],

                'has_key_metrics': len(output['Key_Metrics_to_watch']) > 0,
        'has_directions':  len(output['Metric_Direction']) > 0,
        'keys_match':      all(k in output['Metric_Direction'] 
                               for k in output['Key_Metrics_to_watch'])
    }
    scores['total'] = sum(scores.values()) / len(scores)
    return scores

def eval_2(report, profiler, findings):
    ground_truth_text = str(profiler) + str(findings)
    raw_numbers = set(re.findall(r'\d+\.?\d*', ground_truth_text))
    report_normalized=report.replace('\u202f', ' ').replace('\u2011', '-').replace('\u2013', '-')
    expanded = set(raw_numbers)
    for n in raw_numbers:
        f = float(n)
        if f < 1:
            expanded.add(str(int(round(f * 100))))
            expanded.add(str(round(f * 100, 2)))
        elif f > 1:
            expanded.add(str(round(f / 100, 4)))
    numbers_in_findings = expanded

    skip_patterns = [
        r'page\s+\d+',
        r'over[-\s]?\d',
        r'top\s+\d+',
        r'ranked.{0,10}\d+',
        r'section\s+\d+',
        r'\d+\s*%\s*confidence',
        r'\d+\.\s+\*{0,2}\w',
        r'phase\s*[\d\-]',
        r'days?\b',
        r'hrs?\b',
        r'timeline',
        r'\d{4}',
        r'priority',
        r'estimate',
        r'approximately',
        r'roughly',
        r'≈',
        r'~\s*\d',
        r'projected?',
        r'variance',         
        r'account\s+for',   
        r'attribut',         
        r'responsible\s+for',
        r'\$[\d\.]+\s*[mk]',
        r'[mk]\b',          
        r'replacement\s+cost',
        r'hidden',
        r'lost\s+product',
    ]

    invented = []
    for match in re.finditer(r'(\d[\d,]*\.?\d*)',report):
        raw_match=match.group(1)
        num=raw_match.replace(',','')  
    
        try:
            fnum = float(num)
        except ValueError:
            continue

        start = max(0, match.start() - 30)
        end   = min(len(report), match.end() + 30)
        context = report[start:end].lower()
        start_context = report[max(0, match.start()-5):match.start()].lower()

        if any(op in start_context for op in ['= ', '× ', 'to ', '→']):
            continue
        if any(re.search(p, context) for p in skip_patterns):
            continue
        if fnum <= 20 and re.search(r'\d+\.\s+\*{0,2}\w', context):
            continue
        if fnum <= 20 and any(
            w in context for w in ['issue', 'rank', 'step', 'point', 'item', '#']
        ):
            continue
        stat_patterns = [r'mean', r'average', r'avg', r'gap', r'delta', r'\bvs\b',
                         r'compared', r'higher', r'lower', r'unit\s+gap', r'disparity']
        if any(re.search(p, context) for p in stat_patterns):
            continue

        if num not in numbers_in_findings:
            invented.append({'number': num, 'context': context.strip()})

    return {
        'invented': invented,
        'invention_count': len(invented),
        'pass': len(invented)<=5
    }

def eval_3(charts):
    violations=[]
    seen_pairs=set()

    for c in charts:
        ct = c['chart_type']
        x=tuple(c['x']) if isinstance(c['x'], list) else c['x']
        y=tuple(c.get('y',[])) if isinstance(c.get('y'), list) else c.get('y')
        pair=(x,y)

        if ct=='stacked_bar' and c.get('color') == c.get('x'):
            violations.append(f"stacked_bar: color==x on {c['x']}")
        if ct=='pivot_heatmap' and not c.get('z'):
            violations.append(f"pivot_heatmap missing z")
        if ct in ['bar','line','scatter'] and not c.get('x'):
            violations.append(f"{ct} missing x")
        if pair in seen_pairs:
            violations.append(f"Duplicate x/y pair: {pair}")
        seen_pairs.add(pair)

    return {
        'violations': violations,
        'violation_count': len(violations),
        'pass':len(violations)==0
    }

def eval_4(report):
    required_sections=[
        '# Executive Summary',
        '# Critical Issues',
        '# Root Cause',
        '# Recommended Actions',
        '# Data Gaps']
    import re
    actions = re.findall(r'^#{1,3} .*action', report, re.MULTILINE | re.IGNORECASE)   
    return {
        'has_all_sections':all(s in report for s in required_sections),
        'missing_sections':[s for s in required_sections if s not in report],
        'action_count': len(actions),
        'actions_under_5':len(actions) <= 5,
        'length':True if len(report)>=1500 else False
    }

def eval_5(causal_reasoning):
    issues=[]
    connections=causal_reasoning.get('causal_connections', [])
    
    if len(connections)==0:
        return {'pass': False, 'issues':['no causal_connections found']}
    
    for c in connections:
        label = c.get('issue','unknown')
        if not c.get('internal_evidence'):
            issues.append(f'{label}: missing internal_evidence')
        if not c.get('causal_mechanism'):
            issues.append(f'{label}: missing causal_mechanism')
        if c.get('verdict') not in ('supported', 'contradicted', 'no_external_evidence'):
            issues.append(f'{label}: invalid verdict value')
        vd = c.get('variance_decomposition', {})
        try:
            nums = re.findall(r'(\d+)%', str(vd))
            total = sum(int(n) for n in nums[:3])
            if not (85 <= total <= 115):
                issues.append(f'{label}: variance decomposition sums to {total},not ~100')
        except:
            issues.append(f'{label}: could not parse variance decomposition')
    
    return {
        'issues': issues,
        'issue_count': len(issues),
        'connection_count': len(connections),
        'pass': len(issues) == 0
    }

def eval_6(anonymized_df, original_df, key, mapping_log):
    f=Fernet(key.encode())
    issues=[]
    
    for col in mapping_log.keys():
        if col not in anonymized_df.columns:
            issues.append(f"{col} missing from anonymized_df")
            continue
        original_vals = set(original_df[col].astype(str).unique())
        anon_vals = set(anonymized_df[col].astype(str).unique())
        leaked = original_vals & anon_vals
        if leaked:
            issues.append(f"{col} still has original values: {leaked}")
        if not all(str(v).startswith('ID-') for v in anon_vals):
            issues.append(f"{col} has non-anonymized values")
    
    return {
        'issues':issues,
        'issue_count':len(issues),
        'pass':len(issues) == 0
    }
    

def evaluation(state):
    score={}
    score['domain']=eval_1(json.loads(state['domain_info']))
    score['report']=eval_4(state['report'])
    score['charts']=eval_3(json.loads(state['charts']))
    score['faithfulness']=eval_2(state['report'],json.loads(state['profiler']),json.loads(state['interpreted_findings']))
    score['causal']=eval_5(json.loads(state['causal_reasoning']))
    score['anonymisation']=eval_6(pd.read_json(state['anonymized_df']),pd.read_json(state['df']),state['key'],json.loads(state['mapping_log']))
    print("====SCORECARD====")
    total_score=0
    max_score=0
    for i,node in score.items():
        bool_items ={k: v for k, v in node.items() if isinstance(v, bool)}
        passed=sum(bool_items.values())
        total=len(bool_items)
        node_score=round((passed/total)*100) if total >0 else 0
        
        total_score+=passed
        max_score+=total
        print(f'Node Score:{node_score}')
    overall=round((total_score/max_score)*100) if max_score > 0 else 0
    print(f"\n{'='*35}")
    print(f"OVERALL SCORE:  {overall}/100")
    print(f"CHECKS PASSED:  {total_score}/{max_score}")
    print(f"{'='*35}")  
    score['overall'] = overall
    return score
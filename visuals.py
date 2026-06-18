import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import plotly.offline as pyo
from IPython.display import display
import json
import numpy as np
from crypto import decryption
from helper import safe_parse,derive_accents,random_template,retry
from langsmith import traceable
pyo.init_notebook_mode(connected=True)

def render_charts(inputs,df):
    figs=[]
    inputs=validate_charts(inputs,df)
    for c in inputs:
        try:
            chart_type=c['chart_type'] 
            x=c.get('x')
            y=c.get('y')
            z=c.get('z') 
            color=c.get('color') 
            priority=c.get('priority', 1)
            
            if chart_type=='stacked_bar':
                if x not in df.columns or y not in df.columns or color not in df.columns:
                    print(f"Skipping stacked_bar: missing columns {x}, {y}, {color}")
                    continue
                grouped=df.groupby([x,color])[y].sum().reset_index()
                if grouped.empty:
                    print(f"Skipping stacked_bar: no data after grouping")
                    continue
                fig1=px.bar(grouped,x=x,y=y,color=color,barmode='stack',title=f'{x} by {y} and {color}',template='plotly_white')
                figs.append(fig1)

            elif chart_type=='bar':
                if x not in df.columns or y not in df.columns:
                    print(f"Skipping bar: missing columns {x}, {y}")
                    continue
                grouped=df.groupby(x)[y].sum().reset_index()
                if grouped.empty:
                    print(f"Skipping bar: no data after grouping")
                    continue
                fig2=px.bar(grouped,x=x,y=y,title=f'{x} vs {y}',template='plotly_white')
                figs.append(fig2)

            elif chart_type=='histogram':
                col=x if x else y
                if col not in df.columns:
                    print(f"Skipping histogram: column {col} not found")
                    continue
                fig3=px.histogram(df,x=col,nbins=20,marginal='box',title=f'Distribution of {col}',template='plotly_white')
                figs.append(fig3)

            elif chart_type=='treemap':
                if x not in df.columns or y not in df.columns:
                    print(f"Skipping treemap: missing columns {x}, {y}")
                    continue
                grouped=df.groupby(x)[y].sum().reset_index()
                grouped=grouped[grouped[y] > 0]
                if grouped.empty:
                    print(f"Skipping treemap: no positive data after grouping")
                    continue
                fig4=px.treemap(grouped,path=[px.Constant('All'),x],values=y,title=f'{x} vs {y}',color=y,color_continuous_scale='RdYlGn_r',template='plotly_white')
                figs.append(fig4)

            elif chart_type=='pareto':
                if x not in df.columns or y not in df.columns:
                    print(f"Skipping pareto: missing columns {x}, {y}")
                    continue
                grouped =df.groupby(x)[y].mean().reset_index()
                grouped =grouped.sort_values(y, ascending=False)
                if grouped.empty:
                    print(f"Skipping pareto: no data after grouping")
                    continue
                grouped['cumulative_pct']=(
                    grouped[y].cumsum()/grouped[y].sum()*100
                )
                
                fig5= go.Figure()
                fig5.add_trace(go.Bar(x=grouped[x],y=grouped[y],name=y))
                fig5.add_trace(go.Scatter(
                    x=grouped[x], 
                    y=grouped['cumulative_pct'],
                    name='cumulative %',
                    yaxis='y2',
                    mode='lines+markers'
                ))
                fig5.update_layout(
                    yaxis2=dict(overlaying='y',side='right',range=[0,100]))
                figs.append(fig5)

            elif chart_type=='correlation_heatmap':
                numeric_df = df.select_dtypes(include='number')
                if numeric_df.shape[1] < 2:
                    print(f"Skipping correlation_heatmap: not enough numeric columns")
                    continue
                corr = numeric_df.corr()
                fig6=px.imshow(corr,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,title=f'Correlation Heatmap',text_auto=True,template='plotly_white')
                figs.append(fig6)

            elif chart_type=='scatter':
                if x not in df.columns or y not in df.columns:
                    print(f"Skipping scatter: missing columns {x}, {y}")
                    continue
                fig7=px.scatter(df,x=x,y=y,color=color if color else None,trendline='ols',title=f'{x} vs {y}',template='plotly_white')
                figs.append(fig7)

            elif chart_type=='line':
                if x not in df.columns or y not in df.columns:
                    print(f"Skipping line: missing columns {x}, {y}")
                    continue
                df_sorted=df.copy()
                try:
                    df_sorted[x]=pd.to_datetime(df_sorted[x], errors='coerce')
                    df_sorted=df_sorted.dropna(subset=[x])
                    df_sorted=df_sorted.sort_values(x)
                    grouped=df_sorted.groupby(x)[y].mean().reset_index()
                    if grouped.empty:
                        print(f"Skipping line: no data after grouping")
                        continue
                    fig8=px.line(grouped,x=x,y=y,markers=True,title=f'{y} over time',template='plotly_white')
                    figs.append(fig8)
                except Exception as e:
                    print(f"Skipping line chart: {e}")
                    continue

            elif chart_type=='pivot_heatmap':
               if z not in df.columns or x not in df.columns or y not in df.columns:
                    print(f"Skipping pivot_heatmap: missing columns {x}, {y}, {z}")
                    continue
               if not np.issubdtype(df[z].dtype, np.number):
                    print(f"Skipping pivot_heatmap: '{z}' is not numeric")
                    continue
               pivot=df.pivot_table(values=z,index=x,columns=y,aggfunc='mean')
               if pivot.empty:
                    print(f"Skipping pivot_heatmap: pivot table is empty")
                    continue
               fig9=px.imshow(pivot,color_continuous_scale='RdYlGn_r',text_auto=True,template='plotly_white',title=f'{z} by {x} and {y}')
               figs.append(fig9)

            else:
                print(f"Unknown chart type: {chart_type}")
        except Exception as e:
            print(f"[ERROR] Failed to render {c.get('chart_type')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return figs

def validate_charts(chart_recs,df):
    valid=[]
    for chart in chart_recs:
        ct=chart.get('chart_type')
        x=chart.get('x')
        y=chart.get('y')
        z=chart.get('z')
        color=chart.get('color')
        if not ct:
            print(f"Skipping chart: missing chart_type")
            continue
        if isinstance(x, list):
            if ct != 'correlation_heatmap':
                print(f"Skipping {ct}: x is a list {x}, not a string")
                continue
        if isinstance(y, list):
            print(f"Skipping {ct}: y is a list {y}, not a string")
            continue
        if isinstance(z, list):
            print(f"Skipping {ct}: z is a list {z}, not a string")
            continue
        if isinstance(color, list):
            print(f"Skipping {ct}: color is a list {color}, not a string")
            continue
        if x is None and ct not in ['correlation_heatmap']:
            print(f"Skipping {ct}: x column is None")
            continue
        if y is None and ct not in ['histogram', 'correlation_heatmap']:
            print(f"Skipping {ct}: y column is None")
            continue
        is_valid = True
        for col in [x, y, z, color]:
            if isinstance(col, list):
                for c in col:
                    if c not in df.columns:
                        print(f"Skipping {ct}: column '{c}' not found in dataframe")
                        is_valid = False
                        break
            elif col and col not in df.columns:
                print(f"Skipping {ct}: column '{col}' not found in dataframe")
                is_valid = False
                break
        if not is_valid:
            continue
            
        if ct in ['scatter','histogram'] and x and isinstance(x, str) and x in df.columns:
            if not pd.api.types.is_numeric_dtype(df[x]):
                print(f"Skipping {ct}: '{x}' is not numeric")
                continue
        if ct in ['scatter','bar'] and y and isinstance(y, str) and y in df.columns:
            if not pd.api.types.is_numeric_dtype(df[y]):
                print(f"Skipping {ct}: '{y}' is not numeric")
                continue
        valid.append(chart)
    
    print(f"Validated {len(valid)}/{len(chart_recs)} charts")
    return valid  

@traceable(name='deciding_plots')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def deciding_plots(llm,df,findings, columns, key_cols, feedback=None):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols =[c for c in df.columns if df[c].nunique() < 50 and c not in numeric_cols]
    feedback_section = f"""
REVISION FEEDBACK: {feedback}
Address this feedback specifically in your chart selection.""" if feedback else ""
    
    prompt=f"""You are a senior data analyst. Select exactly 6 charts for a BI dashboard.

REQUIRED (4 charts):
1. stacked_bar: x=category, y=metric, color=DIFFERENT_category
2. pivot_heatmap: x=category, y=category, z=numeric_metric
3. correlation_heatmap: 3. correlation_heatmap: x=null, y=null, z=null (auto-uses all numeric columns)
4. histogram: x=numeric_metric (distribution)

OPTIONAL (2 charts, choose from):
- bar, line, scatter, treemap, pareto

STRICT RULES:
1. Each chart must have unique x,y combination (no duplicates)
2. stacked_bar: x and color MUST be different columns
3. pivot_heatmap: z (values) must be numeric
4. correlation_heatmap: include only numeric columns
5. Avoid: ID columns, high-cardinality columns (>100 unique values)
6. Order by business impact: Revenue > Quantity > Price > Distribution
7. pareto: ONLY if concentration_analysis shows is_concentrated=true
8. treemap: ONLY if any category >50% of total

EXAMPLES (good):
- stacked_bar: x=Country, y=Revenue, color=ProductCategory ✓
- bar: x=Country, y=Quantity (simple relationship) ✓
- Bad: stacked_bar: x=Country, y=Revenue, color=Country ✗ (duplicate)

CAUTION:
1. NEVER recommend scatter/line with non-numeric columns
2. NEVER use columns: {[c for c in columns if c not in numeric_cols + categorical_cols]}
3. For any chart using {categorical_cols}, verify it's categorical first
4. If a column is not in NUMERIC list, do NOT use it for x/y in scatter/line

Allowed columns: {columns}
Key metrics: {key_cols}

Data findings: {json.dumps(findings, indent=2, default=str)}

{feedback_section}

CRITICAL VALIDATION RULES:
- EVERY chart MUST have ALL required columns filled (NO null values for x, y, color fields)
- x: MUST be a valid column name from allowed columns
- y: MUST be a valid column name from allowed columns  
- z: MUST be a valid column name from allowed columns (for pivot_heatmap, correlation_heatmap, or treemap)
- color: MUST be a valid column name from allowed columns (for stacked_bar only)
- NEVER use null for required fields. If a required field cannot be filled, SKIP that chart type

RESPOND ONLY WITH VALID JSON (no markdown, no explanation):
[
  {{"chart_type": "stacked_bar", "x": "product_line", "y": "sales", "z": null, "color": "region", "priority": 1, "reason": "..."}},
  {{"chart_type": "histogram", "x": "revenue", "y": null, "z": null, "color": null, "priority": 2, "reason": "..."}},
  ...
]"""
    response = llm.invoke(prompt)
    charts = safe_parse(response.content)
    return charts

@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def build_dashboard(df,key_cols,inputs,key,mapping_log,domain_info):
    inputs=validate_charts(inputs,df)
    inputs=inputs[:6]  

    if not inputs:
        raise ValueError("No valid charts after validation")

    decrypt_map=decryption(key, mapping_log)
    
    temp,bs,dp=random_template()
    
    try:
        figs=render_charts(inputs, df)
    except Exception as e:
        print(f"[ERROR] render_charts failed: {e}")
        raise

    chart_htmls_top=[]
    chart_htmls_bot=[]
    for i, fig in enumerate(figs):
        try:
            fig.update_layout(
                template='plotly_white',
                margin=dict(t=40, b=20, l=16, r=16),
                height=300 if i < 2 else 240,
                paper_bgcolor=dp["card"],
                plot_bgcolor=dp["card"],
                font=dict(family="Inter, Segoe UI, sans-serif", size=11, color=dp["text"]),
                title_font=dict(size=12, color=dp["text"], family="Inter, Segoe UI, sans-serif"),
                legend=dict(font=dict(size=10, color=dp["muted"]), bgcolor="rgba(0,0,0,0)"),
            )
        except Exception as e:
            print(f"[WARNING] Failed to update layout for chart {i}: {e}, using default template")
            fig.update_layout(
                margin=dict(t=40, b=20, l=16, r=16),
                height=300 if i < 2 else 240,
                font=dict(family="Inter, Segoe UI, sans-serif", size=11),
                title_font=dict(size=12, family="Inter, Segoe UI, sans-serif"),
            )
        
        try:
            html_chunk=pio.to_html(fig,full_html=False,include_plotlyjs='cdn')
            if i<2:
                chart_htmls_top.append(f'<div class="chart-card">{html_chunk}</div>')
            else:
                chart_htmls_bot.append(f'<div class="chart-card-sm">{html_chunk}</div>')
        except Exception as e:
            print(f"[ERROR] Failed to convert chart {i} to HTML: {e}")
            continue

    top_row="".join(chart_htmls_top)
    bot_row="".join(chart_htmls_bot)
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        df = df.copy()
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        df = df.sort_values(date_cols[0])
    metric_direction=domain_info.get("Metric_Direction", {})
    kpis = []
    for metric in key_cols:
        if metric not in df.columns:
            continue
        if not np.issubdtype(df[metric].dtype, np.number):
            continue
        mid=len(df)//2
        base=df[metric].iloc[:mid].mean()
        delta=((df[metric].iloc[mid:].mean()-base) /base*100) if base != 0 else 0
        val=float(df[metric].mean())
        val_str=f"{val/1000:.2f}K" if val >= 1000 else str(round(val, 2))
        kpis.append({
            "label":     metric.replace("_", " ").title(),
            "value":     val_str,
            "direction": metric_direction.get(metric, "higher_is_better"),
            "delta":     round(delta, 2),
        })
    kpis=kpis[:6]

    accents=[dp["accent"],dp["accent2"],dp["muted"],
               dp["accent"],dp["accent2"],dp["muted"]]

    def kpi_card(k, accent):
        sign="▲" if k["delta"] >= 0 else "▼"
        is_good=(k["delta"] >= 0 and k["direction"] == "higher_is_better") or \
                  (k["delta"] <  0 and k["direction"] == "lower_is_better")
        dcolor="#16a34a" if is_good else "#dc2626"
        return f"""
        <div class="kpi-card" style="border-top:3px solid {accent}">
            <div class="kpi-label">{k['label']}</div>
            <div class="kpi-value">{k['value']}</div>
            <div class="kpi-delta" style="color:{dcolor}">{sign} {abs(k['delta'])}% vs prior</div>
        </div>"""

    kpi_html="".join(kpi_card(k, accents[i]) for i, k in enumerate(kpis))

    cat_cols  = list(df.select_dtypes(include="object").columns[:7])
    btn_colors = derive_accents(dp["accent"], max(len(cat_cols), 4))
    sidebar_items = ""
    for j, col in enumerate(cat_cols):
        color = btn_colors[j % len(btn_colors)]
        sidebar_items += f"""
        <div class="nav-btn" style="background:{color}">
            {col.replace("_"," ").title()}
        </div>"""

    filter_col=cat_cols[0] if cat_cols else None
    filter_vals=list(df[filter_col].dropna().unique()[:5]) if filter_col else []
    filter_pills=""
    if filter_vals:
        for v in filter_vals:
            filter_pills+=f'<div class="filter-pill">{v}</div>'
        filter_section=f"""
        <div class="filter-group">
            <span class="filter-label">{filter_col.replace("_"," ").title() if filter_col else ""}</span>
            {filter_pills}
        </div>"""
    else:
        filter_section=""

    title=domain_info.get("Domain","Analytics").replace("_", " ").title()
    subdomain=domain_info.get("Subdomain", "")
    confidence=domain_info.get("Confidence_Score", "")

    html=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter','Segoe UI',sans-serif;background:{dp['bg']};color:{dp['text']};min-height:100vh;display:flex;flex-direction:column}}

/* ── HEADER ── */
.header{{
    background:{dp['sidebar']};
    padding:0 24px;
    display:flex;align-items:center;justify-content:space-between;
    flex-shrink:0;
}}
.header-left{{display:flex;flex-direction:column;justify-content:center;padding:10px 0}}
.header-title{{
    font-size:22px;font-weight:800;
    color:{dp['header_text']};
    letter-spacing:1px;text-transform:uppercase;
    line-height:1.1;
}}
.header-sub{{
    font-size:11px;font-weight:400;
    color:{dp['header_text']}88;margin-top:2px;
}}
.header-right{{display:flex;align-items:center;gap:8px;flex-direction:column;align-items:flex-end;padding:8px 0}}
.pills-row{{display:flex;gap:6px;align-items:center}}
.filter-group{{display:flex;align-items:center;gap:6px}}
.filter-label{{font-size:10px;font-weight:600;color:{dp['header_text']}88;text-transform:uppercase;letter-spacing:0.6px}}
.filter-pill{{
    font-size:10px;font-weight:600;padding:4px 10px;
    border-radius:4px;border:1px solid {dp['header_text']}44;
    color:{dp['header_text']};background:{dp['header_text']}15;
    cursor:pointer;transition:background 0.15s;
}}
.filter-pill:hover{{background:{dp['header_text']}30}}
.theme-pill{{
    font-size:10px;font-weight:600;padding:4px 10px;border-radius:4px;
    border:1px solid {dp['accent']}66;color:{dp['accent']};background:{dp['accent']}18;
    text-transform:uppercase;letter-spacing:0.4px;
}}
.confidence-pill{{
    font-size:10px;font-weight:600;padding:4px 10px;border-radius:4px;
    background:#16a34a22;color:#16a34a;border:1px solid #16a34a55;
    text-transform:uppercase;letter-spacing:0.4px;
}}

/* ── LAYOUT ── */
.body-wrap{{display:flex;flex:1;overflow:hidden}}

/* ── SIDEBAR ── */
.sidebar{{
    width:150px;background:{dp['sidebar']};
    flex-shrink:0;padding:16px 10px;
    display:flex;flex-direction:column;gap:8px;
}}
.sidebar-label{{
    font-size:9px;font-weight:700;
    color:{dp['header_text']}44;letter-spacing:1.2px;
    text-transform:uppercase;margin-bottom:4px;
}}
.nav-btn{{
    font-size:11px;font-weight:600;
    color:#ffffff;
    padding:9px 12px;
    border-radius:5px;
    cursor:pointer;
    text-align:center;
    opacity:0.92;
    transition:opacity 0.15s,transform 0.1s;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}}
.nav-btn:hover{{opacity:1;transform:translateX(2px)}}

/* ── MAIN ── */
.main{{flex:1;padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:14px}}

/* ── KPI ROW ── */
.kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}
.kpi-card{{
    background:{dp['card']};
    border:1px solid {dp['border']};
    border-radius:6px;
    padding:14px 14px 12px;
    display:flex;flex-direction:column;gap:3px;
}}
.kpi-label{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:{dp['muted']}}}
.kpi-value{{font-size:28px;font-weight:800;line-height:1.1;letter-spacing:-0.5px;color:{dp['text']}}}
.kpi-delta{{font-size:10px;font-weight:500;margin-top:2px}}

/* ── TOP CHARTS (2 col) ── */
.charts-top{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.chart-card{{
    background:{dp['card']};border:1px solid {dp['border']};
    border-radius:6px;padding:14px;overflow:hidden;
}}

/* ── BOTTOM CHARTS (up to 4 col) ── */
.charts-bot{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.chart-card-sm{{
    background:{dp['card']};border:1px solid {dp['border']};
    border-radius:6px;padding:12px;overflow:hidden;
}}

/* ── SECTION LABEL ── */
.section-label{{
    font-size:9px;font-weight:700;text-transform:uppercase;
    letter-spacing:1px;color:{dp['muted']};margin-bottom:6px;
}}

/* ── FOOTER ── */
.footer{{
    background:{dp['sidebar']};
    padding:8px 24px;
    display:flex;justify-content:space-between;align-items:center;
    font-size:10px;color:{dp['header_text']}66;flex-shrink:0;
}}
.footer-brand{{font-weight:700;color:{dp['accent']};letter-spacing:0.3px}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div class="header-left">
        <div class="header-title">{title} Analytics Dashboard</div>
        {f'<div class="header-sub">/ {subdomain.replace("_"," ").title()}</div>' if subdomain and subdomain != "other_unknown" else ""}
    </div>
    <div class="header-right">
        {filter_section}
        <div class="pills-row">
            <span class="theme-pill">Theme: {bs}</span>
            {f'<span class="confidence-pill">Confidence: {confidence}</span>' if confidence else ""}
        </div>
    </div>
</div>
<div class="body-wrap">
    <!-- SIDEBAR -->
    <div class="sidebar">
        <div class="sidebar-label">Dimensions</div>
        {sidebar_items}
    </div>
    <!-- MAIN -->
    <div class="main">
        <!-- KPIs -->
        <div>
            <div class="section-label">Key Performance Indicators</div>
            <div class="kpi-grid">{kpi_html}</div>
        </div>
        <!-- TOP CHARTS -->
        {'<div><div class="section-label">Charts &amp; Analysis</div><div class="charts-top">' + top_row + '</div></div>' if top_row else ''}
        <!-- BOTTOM CHARTS -->
        {'<div><div class="charts-bot">' + bot_row + '</div></div>' if bot_row else ''}
    </div>
</div>
<!-- FOOTER -->
<div class="footer">
    <span>Last refreshed: <span id="ts"></span></span>
    <span class="footer-brand">ArgusAI &nbsp;·&nbsp; Automated Insight Engine</span>
</div>
<script>document.getElementById('ts').textContent = new Date().toLocaleString();</script>
</body>
</html>"""
    for display,real in decrypt_map.items():
        html=html.replace(display,real)
    with open("outputs/dashboard.html","w",encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved. Theme: {bs}")
    return "outputs/dashboard.html"


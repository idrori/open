#!/usr/bin/env python3
"""Generate the main webapp with embedded problem data."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

with open(os.path.join(ROOT, "kdd_classification.json")) as f:
    kdd = json.load(f)

problems = []
for track, probs in kdd.items():
    if isinstance(probs, list):
        for p in probs:
            if isinstance(p, dict) and "cat" in p and "slug" in p:
                base = os.path.join(ROOT, "problems", p["cat"], p["slug"])
                problems.append({
                    "cat": p["cat"],
                    "slug": p["slug"],
                    "title": p.get("title", p["slug"].replace("-", " ").title()),
                    "track": track,
                    "has_pdf": os.path.isfile(os.path.join(base, "paper", "main.pdf")),
                    "has_review": os.path.isfile(os.path.join(base, "review.txt")),
                    "has_revision": os.path.isfile(os.path.join(base, "revised.pdf")),
                    "has_webapp": os.path.isfile(os.path.join(base, "webapp", "index.html")),
                })

problems_json = json.dumps(problems)

# Generate pre-rendered problem rows as static HTML
problem_rows_html = []
for p in problems:
    track_class = "track-" + p["track"]
    track_labels = {"research": "Research", "ai4sciences": "AI4Sci", "datasets": "Datasets", "ads": "ADS"}
    cat_names = {
        "LG": "Machine Learning", "AI": "Artificial Intelligence", "CL": "Computation & Language",
        "CV": "Computer Vision", "PH": "Physics", "PS": "Programming & Software",
        "ML": "Machine Learning (stat)", "EP": "Earth & Planetary", "RO": "Robotics",
        "FL": "Fluid Dynamics", "ST": "Statistics Theory", "IR": "Information Retrieval",
        "CC": "Computational Complexity", "CR": "Cryptography", "BI": "Bioinformatics",
        "HC": "Human-Computer Interaction", "GN": "General & Nuclear Physics",
        "SR": "Signal Processing", "TH": "Theoretical Physics", "DL": "Digital Libraries",
        "NC": "Neural Computing", "CY": "Computers & Society", "SO": "Software Engineering",
        "SD": "Sound", "NI": "Networking", "MT": "Math Theory", "QM": "Quantum Mechanics",
        "AO": "Astrophysics", "HE": "High Energy Physics", "GE": "Geophysics",
        "NA": "Numerical Analysis", "IV": "Imaging & Video", "CP": "Computational Physics",
        "IT": "Information Theory", "SY": "Systems & Control", "AR": "Architecture",
        "ME": "Medical Imaging", "CO": "Combinatorics", "PE": "Performance",
        "GA": "Genetic Algorithms", "PL": "Programming Languages", "SU": "Superconductivity",
        "MN": "Materials & Nanostructures"
    }

    title_escaped = p["title"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    cat_full = cat_names.get(p["cat"], p["cat"])
    prob_base = "problems/" + p["cat"] + "/" + p["slug"] + "/"
    paper_path = prob_base + "paper/main.pdf"
    infographic_path = prob_base + "paper/infographic.jpg"
    slides_path = prob_base + "paper/slides.pdf"
    app_path = prob_base + "webapp/index.html"

    links = ""
    if p["has_pdf"]:
        links += '<a href="' + paper_path + '" target="_blank" rel="noopener" class="paper-link">Paper</a>'
    links += '<a href="' + infographic_path + '" target="_blank" rel="noopener" class="review-link">Infographic</a>'
    links += '<a href="' + slides_path + '" target="_blank" rel="noopener" class="revision-link">Slides</a>'
    if p["has_webapp"]:
        links += '<a href="' + app_path + '" target="_blank" rel="noopener" class="app-link">App</a>'

    row = (
        '<div class="problem-row" '
        'data-search="' + (p["cat"] + " " + cat_full + " " + title_escaped + " " + p["slug"]).lower().replace('"', '') + '">'
        '<span class="problem-cat" title="' + cat_full + '">' + p["cat"] + '</span>'
        '<span class="problem-title">' + title_escaped + '</span>'
        '<span class="problem-links">' + links + '</span>'
        '</div>'
    )
    problem_rows_html.append(row)

all_rows = "\n".join(problem_rows_html)

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reproducible Automated Scientific Research on Open Problems at Scale</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,500&family=JetBrains+Mono:wght@300;400;500&family=Source+Sans+3:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0b0f;--bg2:#12131a;--bg3:#1a1b25;--bg4:#222330;
  --fg:#e8e6e1;--fg2:#b8b5ad;--fg3:#807d75;
  --accent:#c9a54e;--accent2:#e8c86a;--accent-dim:#6b5a2a;
  --track-research:#5b8fd4;--track-ai4sci:#4ec994;--track-datasets:#e8a44e;--track-ads:#d45b7a;
  --serif:'Cormorant Garamond',Georgia,serif;
  --sans:'Source Sans 3','Source Sans Pro',sans-serif;
  --mono:'JetBrains Mono','Fira Code',monospace;
  --max-w:1200px
}}
html{{scroll-behavior:smooth;font-size:16px}}
body{{font-family:var(--sans);background:var(--bg);color:var(--fg);line-height:1.6;-webkit-font-smoothing:antialiased}}
.hero{{position:relative;min-height:70vh;display:flex;align-items:center;justify-content:center;overflow:hidden;border-bottom:1px solid var(--accent-dim)}}
.hero::before{{content:'';position:absolute;inset:0;background:
  radial-gradient(ellipse 60% 50% at 20% 30%,rgba(201,165,78,0.06) 0%,transparent 70%),
  radial-gradient(ellipse 40% 60% at 80% 70%,rgba(78,201,148,0.04) 0%,transparent 70%),
  linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%)}}
.hero-grid{{position:absolute;inset:0;opacity:0.03;background-image:
  linear-gradient(rgba(201,165,78,0.3) 1px,transparent 1px),
  linear-gradient(90deg,rgba(201,165,78,0.3) 1px,transparent 1px);
  background-size:60px 60px}}
.hero-inner{{position:relative;text-align:center;max-width:900px;padding:2rem;animation:fadeUp 1s ease-out}}
.hero-label{{font-family:var(--mono);font-size:0.7rem;letter-spacing:0.3em;text-transform:uppercase;color:var(--accent);margin-bottom:1.5rem;opacity:0.8}}
.hero h1{{font-family:var(--serif);font-size:clamp(2rem,5vw,3.5rem);font-weight:300;line-height:1.15;color:var(--fg);margin-bottom:1.5rem;letter-spacing:-0.01em}}
.hero h1 em{{font-style:italic;color:var(--accent2);font-weight:400}}
.hero-sub{{font-size:1rem;color:var(--fg2);max-width:700px;margin:0 auto 2rem;line-height:1.7;font-weight:300}}
.hero-authors{{font-size:0.85rem;color:var(--fg3);margin-bottom:2rem}}
.hero-authors strong{{color:var(--fg2);font-weight:500}}
.hero-stats{{display:flex;gap:3rem;justify-content:center;flex-wrap:wrap;margin-top:2rem}}
.stat{{text-align:center}}
.stat-num{{font-family:var(--serif);font-size:2.5rem;font-weight:600;color:var(--accent2);line-height:1}}
.stat-label{{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--fg3);margin-top:0.3rem}}
.nav{{position:sticky;top:0;z-index:100;background:rgba(10,11,15,0.92);backdrop-filter:blur(12px);border-bottom:1px solid rgba(201,165,78,0.15);padding:0.6rem 0}}
.nav-inner{{max-width:var(--max-w);margin:0 auto;display:flex;align-items:center;gap:1.5rem;padding:0 2rem;overflow-x:auto}}
.nav a{{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--fg3);text-decoration:none;white-space:nowrap;padding:0.4rem 0;border-bottom:1px solid transparent;transition:all 0.2s}}
.nav a:hover{{color:var(--accent);border-bottom-color:var(--accent)}}
.nav-title{{font-family:var(--serif);font-size:0.9rem;color:var(--fg);font-weight:500;margin-right:auto;white-space:nowrap}}
section{{max-width:var(--max-w);margin:0 auto;padding:4rem 2rem}}
.section-label{{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.3em;text-transform:uppercase;color:var(--accent);margin-bottom:0.5rem}}
.section-title{{font-family:var(--serif);font-size:clamp(1.5rem,3vw,2.2rem);font-weight:400;color:var(--fg);margin-bottom:1rem;line-height:1.2}}
.section-desc{{color:var(--fg2);font-weight:300;max-width:700px;margin-bottom:2.5rem;line-height:1.7}}
.overview-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.5rem;margin-top:2rem}}
.overview-card{{background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-radius:2px;padding:1.5rem;transition:all 0.3s}}
.overview-card:hover{{border-color:var(--accent-dim);transform:translateY(-2px)}}
.overview-card h3{{font-family:var(--serif);font-size:1.1rem;color:var(--fg);margin-bottom:0.5rem;font-weight:500}}
.overview-card p{{font-size:0.85rem;color:var(--fg3);line-height:1.6}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin-top:2rem}}
.chart-box{{background:var(--bg2);border:1px solid rgba(255,255,255,0.05);border-radius:2px;padding:1.5rem}}
.chart-box h4{{font-family:var(--mono);font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--fg3);margin-bottom:1rem}}
@media(max-width:768px){{.charts-row{{grid-template-columns:1fr}}}}
.controls{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem;align-items:center}}
.search-box{{flex:1;min-width:200px;background:var(--bg2);border:1px solid rgba(255,255,255,0.08);border-radius:2px;padding:0.6rem 1rem;color:var(--fg);font-family:var(--sans);font-size:0.85rem;outline:none;transition:border-color 0.2s}}
.search-box:focus{{border-color:var(--accent-dim)}}
.search-box::placeholder{{color:var(--fg3)}}
.filter-btn{{background:var(--bg3);border:1px solid rgba(255,255,255,0.06);border-radius:2px;padding:0.5rem 1rem;color:var(--fg3);font-family:var(--mono);font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;cursor:pointer;transition:all 0.2s}}
.filter-btn:hover,.filter-btn.active{{background:var(--accent-dim);color:var(--accent2);border-color:var(--accent-dim)}}
.cat-select{{background:var(--bg2);border:1px solid rgba(255,255,255,0.08);border-radius:2px;padding:0.5rem 0.8rem;color:var(--fg2);font-family:var(--mono);font-size:0.7rem;outline:none;cursor:pointer;transition:border-color 0.2s;-webkit-appearance:none;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23807d75'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 0.7rem center;padding-right:2rem}}
.cat-select:focus{{border-color:var(--accent-dim)}}
.cat-select option{{background:var(--bg2);color:var(--fg2)}}
.results-count{{font-family:var(--mono);font-size:0.65rem;color:var(--fg3);letter-spacing:0.1em}}
.problem-list{{display:flex;flex-direction:column;gap:1px}}
.problem-row{{display:grid;grid-template-columns:50px 1fr auto;gap:1rem;align-items:center;padding:0.75rem 1rem;background:var(--bg2);border-left:2px solid transparent;transition:all 0.2s}}
.problem-row:hover{{background:var(--bg3);border-left-color:var(--accent)}}
.problem-row.header{{background:var(--bg);position:sticky;top:52px;z-index:10;font-family:var(--mono);font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--fg3);border-left:2px solid transparent}}
.problem-row.hidden{{display:none}}
.problem-cat{{font-family:var(--mono);font-size:0.7rem;color:var(--accent);font-weight:500}}
.problem-title{{font-size:0.85rem;color:var(--fg);font-weight:400}}
.problem-track{{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;padding:0.2rem 0.5rem;border-radius:1px;text-align:center}}
.track-research{{color:var(--track-research);background:rgba(91,143,212,0.1);border:1px solid rgba(91,143,212,0.2)}}
.track-ai4sciences{{color:var(--track-ai4sci);background:rgba(78,201,148,0.1);border:1px solid rgba(78,201,148,0.2)}}
.track-datasets{{color:var(--track-datasets);background:rgba(232,164,78,0.1);border:1px solid rgba(232,164,78,0.2)}}
.track-ads{{color:var(--track-ads);background:rgba(212,91,122,0.1);border:1px solid rgba(212,91,122,0.2)}}
.problem-links{{display:flex;gap:0.5rem;justify-content:flex-end}}
.problem-links a{{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--fg3);text-decoration:none;padding:0.25rem 0.5rem;border:1px solid rgba(255,255,255,0.08);border-radius:1px;transition:all 0.2s}}
.problem-links a:hover{{color:var(--accent2);border-color:var(--accent-dim)}}
.problem-links a.paper-link:hover{{color:#5b8fd4;border-color:rgba(91,143,212,0.4)}}
.problem-links a.review-link:hover{{color:#e8a44e;border-color:rgba(232,164,78,0.4)}}
.problem-links a.revision-link:hover{{color:#d45b7a;border-color:rgba(212,91,122,0.4)}}
.problem-links a.app-link:hover{{color:#4ec994;border-color:rgba(78,201,148,0.4)}}
.pipeline-steps{{display:flex;gap:0;margin:2rem 0;overflow-x:auto}}
.pipe-step{{flex:1;min-width:90px;padding:1.2rem 0.6rem;background:var(--bg2);border-right:1px solid var(--bg);text-align:center;position:relative;transition:background 0.2s}}
.pipe-step:hover{{background:var(--bg3)}}
.pipe-step:first-child{{border-radius:2px 0 0 2px}}
.pipe-step:last-child{{border-radius:0 2px 2px 0;border-right:none}}
.pipe-num{{font-family:var(--serif);font-size:1.5rem;color:var(--accent);font-weight:600;line-height:1}}
.pipe-label{{font-family:var(--mono);font-size:0.55rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--fg3);margin-top:0.4rem;line-height:1.3}}
.pipe-step::after{{content:'\\203A';position:absolute;right:-8px;top:50%;transform:translateY(-50%);color:var(--accent-dim);font-size:1.2rem;z-index:1}}
.pipe-step:last-child::after{{display:none}}
footer{{border-top:1px solid rgba(201,165,78,0.15);padding:3rem 2rem;text-align:center}}
footer p{{font-size:0.75rem;color:var(--fg3);font-family:var(--mono);letter-spacing:0.05em}}
footer a{{color:var(--accent);text-decoration:none}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
.fade-in{{opacity:0;transform:translateY(15px);transition:all 0.6s ease-out}}
.fade-in.visible{{opacity:1;transform:translateY(0)}}
::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--bg4);border-radius:3px}}
@media(max-width:600px){{
  .problem-row{{grid-template-columns:40px 1fr;gap:0.5rem;padding:0.6rem 0.8rem}}
  .problem-links,.problem-row.header .problem-links{{display:none}}
  .hero-stats{{gap:1.5rem}}
  .pipeline-steps{{flex-wrap:wrap}}
  .pipe-step{{min-width:80px;flex:none;width:calc(25% - 1px)}}
  .pipe-step::after{{display:none}}
}}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-grid"></div>
  <div class="hero-inner">
    <div class="hero-label">KDD 2026 &mdash; AI for Sciences Track</div>
    <h1>Reproducible Automated Scientific Research on <em>Open Problems</em> at Scale</h1>
    <p class="hero-sub">An AI-driven pipeline that autonomously conducts scientific research on 317 open problems across 49 disciplines, producing verified, reproducible research packages.</p>
    <p class="hero-authors">
      <strong>Iddo Drori</strong> &middot;
      <strong>Alexy Skoutnev</strong> &middot;
      <strong>Kirill Acharya</strong> &middot;
      <strong>Gaston Longhitano</strong> &middot;
      <strong>Avi Shporer</strong> &middot;
      <strong>Madeleine Udell</strong> &middot;
      <strong>Dov Te'eni</strong>
    </p>
    <div class="hero-stats">
      <div class="stat"><div class="stat-num">317</div><div class="stat-label">Open Problems</div></div>
      <div class="stat"><div class="stat-num">49</div><div class="stat-label">Disciplines</div></div>
      <div class="stat"><div class="stat-num">317</div><div class="stat-label">Papers</div></div>
      <div class="stat"><div class="stat-num">317</div><div class="stat-label">Web Apps</div></div>
      <div class="stat"><div class="stat-num">100%</div><div class="stat-label">Completion</div></div>
    </div>
  </div>
</header>

<nav class="nav">
  <div class="nav-inner">
    <span class="nav-title">Open Problems at Scale</span>
    <a href="#overview">Overview</a>
    <a href="#pipeline">Pipeline</a>
    <a href="#charts">Analytics</a>
    <a href="#problems">All 317 Problems</a>
    <a href="main/main.pdf" target="_blank" rel="noopener">Paper PDF</a>
  </div>
</nav>

<section id="overview" class="fade-in">
  <div class="section-label">Overview</div>
  <div class="section-title">AI-Driven Scientific Discovery</div>
  <p class="section-desc">For each of 317 open scientific problems curated from recent arXiv publications, our pipeline produces a complete research package: computational solution, deterministic experiments, structured data, a full-length paper, and an interactive web application.</p>
  <div class="overview-grid">
    <div class="overview-card">
      <h3>49 Scientific Disciplines</h3>
      <p>From quantum mechanics and astrophysics to fluid dynamics, biology, and machine learning. The pipeline handles theoretical, computational, and experimental problems across the full spectrum of science.</p>
    </div>
    <div class="overview-card">
      <h3>Three-Layer Verification</h3>
      <p>Every numerical claim is cross-referenced against source data. All experiments are rerun for reproducibility. Mathematical proofs are formally verified.</p>
    </div>
    <div class="overview-card">
      <h3>Full Reproducibility</h3>
      <p>Deterministic seeding (seed 42), pinned dependencies (NumPy 1.26.4, SciPy 1.12.0), and automated experiment reruns ensure bitwise-identical results across runs.</p>
    </div>
    <div class="overview-card">
      <h3>Interactive Exploration</h3>
      <p>Every problem comes with a self-contained HTML web application featuring Chart.js interactive visualizations, data tables, and responsive design.</p>
    </div>
  </div>
</section>

<section id="pipeline" class="fade-in">
  <div class="section-label">Architecture</div>
  <div class="section-title">Seven-Stage Research Pipeline</div>
  <p class="section-desc">Each open problem enters the pipeline and progresses through seven sequential stages, producing a complete, verified research package.</p>
  <div class="pipeline-steps">
    <div class="pipe-step"><div class="pipe-num">1</div><div class="pipe-label">Problem<br>Analysis</div></div>
    <div class="pipe-step"><div class="pipe-num">2</div><div class="pipe-label">Data</div></div>
    <div class="pipe-step"><div class="pipe-num">3</div><div class="pipe-label">Code</div></div>
    <div class="pipe-step"><div class="pipe-num">4</div><div class="pipe-label">Paper</div></div>
    <div class="pipe-step"><div class="pipe-num">5</div><div class="pipe-label">Infographic</div></div>
    <div class="pipe-step"><div class="pipe-num">6</div><div class="pipe-label">Slides</div></div>
    <div class="pipe-step"><div class="pipe-num">7</div><div class="pipe-label">App</div></div>
  </div>
</section>

<section id="charts" class="fade-in">
  <div class="section-label">Analytics</div>
  <div class="section-title">Corpus Statistics</div>
  <div class="charts-row">
    <div class="chart-box">
      <h4>Problems by Track</h4>
      <canvas id="trackChart" height="250"></canvas>
    </div>
    <div class="chart-box">
      <h4>Top 15 Categories</h4>
      <canvas id="catChart" height="250"></canvas>
    </div>
  </div>
</section>

<section id="problems">
  <div class="section-label">Corpus</div>
  <div class="section-title">All 317 Research Packages</div>
  <p class="section-desc">Each problem has a compiled PDF paper and an interactive web application. Use the search and filters below to explore.</p>

  <div class="controls">
    <input type="text" class="search-box" id="searchBox" placeholder="Search by title, category, slug...">
    <select class="cat-select" id="catSelect">
      <option value="all">All Categories</option>
    </select>
    <span class="results-count" id="resultsCount">317 problems</span>
  </div>

  <div class="problem-row header">
    <span>Cat</span>
    <span>Problem Title</span>
    <span style="text-align:right">Links</span>
  </div>
  <div id="problemList" class="problem-list">
{all_rows}
  </div>
</section>

<footer>
  <p>Reproducible Automated Scientific Research on Open Problems at Scale<br>
  KDD 2026 &middot; AI for Sciences Track &middot;
  <a href="main/main.pdf" target="_blank" rel="noopener">Paper PDF</a></p>
</footer>

<script>
// Filter and search using CSS class toggling (no innerHTML)
var currentCat = 'all';
var searchTerm = '';
var rows = document.querySelectorAll('#problemList .problem-row');
var catSelect = document.getElementById('catSelect');

// Populate category dropdown from data
(function() {{
  var cats = {{}};
  for (var i = 0; i < rows.length; i++) {{
    var catEl = rows[i].querySelector('.problem-cat');
    if (catEl) {{
      var code = catEl.textContent.trim();
      var title = catEl.getAttribute('title') || code;
      if (!cats[code]) cats[code] = {{code: code, title: title, count: 0}};
      cats[code].count++;
    }}
  }}
  var sorted = Object.values(cats).sort(function(a, b) {{ return b.count - a.count; }});
  for (var j = 0; j < sorted.length; j++) {{
    var opt = document.createElement('option');
    opt.value = sorted[j].code;
    opt.textContent = sorted[j].code + ' \u2014 ' + sorted[j].title + ' (' + sorted[j].count + ')';
    catSelect.appendChild(opt);
  }}
}})();

function filterProblems() {{
  var count = 0;
  for (var i = 0; i < rows.length; i++) {{
    var row = rows[i];
    var searchMatch = !searchTerm || row.getAttribute('data-search').indexOf(searchTerm.toLowerCase()) !== -1;
    var catEl = row.querySelector('.problem-cat');
    var catCode = catEl ? catEl.textContent.trim() : '';
    var catMatch = currentCat === 'all' || catCode === currentCat;
    if (searchMatch && catMatch) {{
      row.classList.remove('hidden');
      count++;
    }} else {{
      row.classList.add('hidden');
    }}
  }}
  document.getElementById('resultsCount').textContent = count + ' problem' + (count !== 1 ? 's' : '');
}}

document.getElementById('searchBox').addEventListener('input', function(e) {{
  searchTerm = e.target.value;
  filterProblems();
}});

catSelect.addEventListener('change', function() {{
  currentCat = catSelect.value;
  filterProblems();
}});

// Charts
Chart.defaults.color = '#807d75';
Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.font.family = "'Source Sans 3', sans-serif";

new Chart(document.getElementById('trackChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Research (203)','AI for Sciences (90)','Datasets & Benchmarks (22)','Applied Data Science (2)'],
    datasets: [{{ data: [203, 90, 22, 2], backgroundColor: ['#5b8fd4','#4ec994','#e8a44e','#d45b7a'], borderWidth: 0 }}]
  }},
  options: {{ responsive:true, plugins:{{legend:{{position:'bottom',labels:{{padding:12,font:{{size:11}}}}}}}} }}
}});

new Chart(document.getElementById('catChart'), {{
  type: 'bar',
  data: {{
    labels: ['LG','AI','CL','CV','PH','PS','ML','EP','RO','FL','ST','IR','CC','CR','BI'],
    datasets: [{{ data: [50,40,38,32,23,13,12,11,9,8,7,6,6,5,5],
      backgroundColor: ['#c9a54e','#5b8fd4','#4ec994','#e8a44e','#d45b7a','#9b7fd4','#4ec9c9','#d4a05b','#7fd49b','#c94e7a','#d4c95b','#5bd4a0','#7f9bd4','#c9784e','#4e8bc9'],
      borderWidth: 0, borderRadius: 2 }}]
  }},
  options: {{ indexAxis:'y', responsive:true, plugins:{{legend:{{display:false}}}}, scales:{{x:{{grid:{{display:false}}}},y:{{grid:{{display:false}}}}}} }}
}});


// Scroll animations
var observer = new IntersectionObserver(function(entries) {{
  entries.forEach(function(e) {{ if(e.isIntersecting) e.target.classList.add('visible'); }});
}}, {{threshold: 0.1}});
document.querySelectorAll('.fade-in').forEach(function(el) {{ observer.observe(el); }});

// Animate stat numbers
document.querySelectorAll('.stat-num').forEach(function(el) {{
  var val = el.textContent;
  var isPercent = val.indexOf('%') !== -1;
  var num = parseInt(val);
  if (isNaN(num)) return;
  el.textContent = '0';
  var start = performance.now();
  function update(now) {{
    var p = Math.min((now - start) / 1500, 1);
    var ease = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(num * ease) + (isPercent ? '%' : '');
    if (p < 1) requestAnimationFrame(update);
  }}
  setTimeout(function() {{ requestAnimationFrame(update); }}, 500);
}});
</script>
</body>
</html>'''

os.makedirs(os.path.join(BASE, "webapp"), exist_ok=True)
with open(os.path.join(BASE, "webapp", "index.html"), "w") as f:
    f.write(html)

print("Generated webapp/index.html (%d bytes, %d problem rows)" % (len(html), len(problems)))

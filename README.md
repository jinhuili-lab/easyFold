# easyFold

**easyFold** is an interactive, structure-aware platform for **AlphaFold3 job management, visualization, and domain-level interpretation**.  
It integrates confidence metrics (pLDDT, PAE), contact maps, and automated domain segmentation to enable **deep, interpretable analysis of predicted protein structures**.

![img](https://raw.githubusercontent.com/jinhuili-lab/personal_image_bed/master/forMD/202512141825186.png)


## ✨ Key Features

### 🔹 AlphaFold3 Job Management
- Docker-based AlphaFold3 execution
- User-level job submission, tracking, and result management
- Administrator dashboard for global job monitoring and control
- Unique job IDs to avoid conflicts across users

### 🔹 Interactive Structure Visualization
- **3D structure visualization** using Mol\*
- Supports PDB / mmCIF outputs
- Optional domain-colored structures (via B-factor encoding)

### 🔹 Confidence & Contact Analysis
- **pLDDT per-residue curve**
- **PAE heatmap** visualization
- **Contact map** (CA–CA, configurable cutoff)
- Mouse-based region selection on PAE and contact maps

### 🔹 Domain-Aware Interpretation (Optional)
- Integrated **Merizo** for automatic domain segmentation
- Domain boundary visualization:
  - Sequence domain bar
  - Domain overlays on PAE and contact maps
  - Structure highlighting by domain
- Quantitative **intra- / inter-domain contact density** analysis

### 🔹 Linked, Interpretable Views
- Domain bar ↔ PAE ↔ contact map ↔ 3D structure are fully linked
- Selecting a region or domain highlights corresponding residues across views
- Designed for **structure-informed domain interpretation**, not just visualization

---

## 🧠 Design Philosophy

easyFold is **not** just a wrapper for AlphaFold.

It is designed to support:
- Domain-level reasoning
- Structure-aware interpretation
- Confidence-guided analysis
- Exploration of inter-domain coupling and organization

This makes easyFold suitable for:
- Multi-domain proteins
- Large bacterial proteins
- Toxin systems
- Structure-based functional annotation studies

---

## 🏗 Architecture Overview

- **Backend**: FastAPI + SQLAlchemy + SQLite
- **Frontend**: Jinja2 templates + Tailwind-style layout
- **Visualization**:
  - Mol\* (3D structure)
  - Chart.js (pLDDT, comparisons)
  - Canvas-based heatmaps (PAE, contact maps)
- **Execution**:
  - Docker-based AlphaFold3
  - Configurable host paths via admin profile
- **Domain segmentation**:
  - Merizo (optional, on-demand)

---

## 📦 Installation

### 1. Clone repository
```bash
git clone https://github.com/your-org/easyFold.git
cd easyFold
```
### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 3. Configure Docker
1. Docker installed and running
2. AlphaFold3 image available (e.g. cford38/alphafold3)
3. GPU support recommended (--gpus all)
### 4. Start server
```
python -m uvicorn app:app --reload
```
Then open:
👉 http://127.0.0.1:8000

## 👤 User Roles
### Regular User
Submit AlphaFold3 jobs

Check jobs before submission

View and download results

Explore structure, confidence, contact maps, and domains

Administrator
Configure host paths (input/output/models/AFDB)

Monitor CPU / GPU / memory usage

View and manage all users’ jobs

Stop or delete jobs

Configure execution limits (single-node mode)

## 📊 Result Dashboard
Each job provides a multi-tab dashboard:

Overview: job metadata and artifacts

Structure: interactive 3D visualization

Confidence: pLDDT, PAE, contact map

Domains: Merizo-based domain segmentation and statistics

Compare: multi-model / seed comparison

Logs: real-time execution logs

## 🧪 Domain & Contact Metrics
easyFold automatically computes:

Intra-domain contact density

Inter-domain contact density

Observed vs. possible contact ratios

These metrics support:

Domain validity assessment

Structural coupling analysis

Domain-level functional hypotheses

## 🚀 Roadmap
Planned or optional extensions:

Stable Mol* residue selection API integration

RMSD-based model comparison

Batch export of publication-ready figures

Remote / cluster execution (PRO mode)

Public web-server deployment

## 📖 Citation
If you use easyFold in your research, please cite:

Li, J. et al.
easyFold: an interactive platform for structure-aware domain interpretation of AlphaFold predictions.
Manuscript in preparation.

## 📬 Contact
Maintained by: Jinhui Li
For issues, suggestions, or collaboration, please open an issue or contact the author.

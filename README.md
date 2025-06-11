# 🧠 SepsisPredict - Early Sepsis Detection Using MIMIC-IV

**SepsisPredict** is a machine learning-based pipeline designed to detect early signs of **sepsis** using structured clinical data from the **MIMIC-IV** dataset. This project aims to provide a reproducible baseline for clinical deterioration prediction using Random Forests and data visualization tools.

---

## 📁 Dataset Sources

We use a subset of the [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) dataset, specifically the following 5 CSV files:

- `ADMISSIONS.csv`
- `PATIENTS.csv`
- `D_LABITEMS.csv`
- `LABEVENTS.csv`
- `structured_medical_records.csv`

These files are located in the `data/` directory and were used to build features and perform exploratory analysis.

---

## 🚀 Project Highlights

- 🔍 Exploratory Data Analysis with Seaborn and Matplotlib  
- 🧼 Preprocessing of patient vitals, labs, and demographics  
- 🧠 Random Forest Classifier for Sepsis Prediction  
- 📊 Visualizations: Heatmaps, Age Distributions, Feature Importances  
- 📁 Outputs saved in `data/output_visuals.png` and `data/results.txt`

---

## 🛠️ How to Run

### 1. Clone this Repository
```bash
git clone https://github.com/your-username/SepsisPredict_MIMIC_Demo.git
cd SepsisPredict_MIMIC_Demo

### 2. Setup Python Environment
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

3. Run the Main Script
bash
Copy
Edit
python main.py
✅ Outputs
📈 Visuals saved to: data/output_visuals.png
📄 Evaluation report saved to: data/results.txt

📊 Sample Visuals


💡 Why This Matters
Sepsis is a life-threatening condition that requires timely diagnosis and intervention. Leveraging machine learning on clinical records can significantly reduce mortality rates by enabling early detection and automated risk scoring.

📜 License
MIT License © 2025
Use freely with attribution. For research and educational purposes.

🙌 Acknowledgments
MIMIC-IV Database by PhysioNet

PhysioNet Challenge & MIT-LCP

Sepsis-3 Clinical Guidelines

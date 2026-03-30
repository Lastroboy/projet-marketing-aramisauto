# 🚗 Aramisauto — Analyse des Profils Clients

Application Streamlit interactive pour explorer les profils clients d'Aramisauto, tester des hypothèses marketing et générer des recommandations de véhicules personnalisées.

---

## 📁 Structure du projet

```
projet/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── df_master.csv
```

---

## 🚀 Lancement rapide

**1. Cloner le dépôt et se placer dans le dossier**
```bash
cd aramisauto
```

**2. Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**4. Lancer l'application**
```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501).

---

## 📊 Contenu de l'application

| Onglet | Description |
|---|---|
| 🏠 Présentation | KPIs globaux et synthèse des insights marketing |
| 📊 Analyse Exploratoire | Profils clients et profils véhicules |
| 🔬 Analyse & Hypothèses | Tests statistiques : Pearson, Chi-deux, ACM |
| 🎯 Recommandation | Moteur de recommandation KNN personnalisé |

---

## ⚙️ Prérequis

- Python **3.9+**
- Le fichier **`data/df_master.csv`** doit être présent avant de lancer l'application

---

## 👥 Auteurs

Noa Delaporte, Sarah Scalart et Yuna Cann — Étudiant(e)s en M1 MAS 2025/2026 — Université de Rennes 2

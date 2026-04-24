
# =========================================================
# app.py  —  Système Intelligent de Rotation des Cultures
# Compatible Streamlit Cloud (aucun input(), chemins relatifs)
# =========================================================

import os, json, joblib, time
import numpy as np
import pandas as pd
import streamlit as st

# ── pymoo ────────────────────────────────────────────────
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CropOpt — Rotation Intelligente",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS minimaliste ──────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f2016; }
  [data-testid="stSidebar"] * { color: #d4f5c4 !important; }
  .metric-box {
      background: #1a3020; border-radius: 10px;
      padding: 1rem; text-align: center; color: #d4f5c4;
  }
  .metric-box h2 { margin: 0; font-size: 1.8rem; color: #6ee37a; }
  .metric-box p  { margin: 0; font-size: 0.85rem; opacity: .7; }
  .stDataFrame { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CHEMINS (relatifs → Streamlit Cloud)
# =========================================================
#BASE = os.path.dirname(os.path.abspath(__file__))
BASE = os.getcwd()
DATA_DIR  = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "model")

# =========================================================
# CHARGEMENT DONNÉES & MODÈLE  (mis en cache)
# =========================================================
@st.cache_resource(show_spinner="Chargement du modèle ML…")
def load_all():
    df   = pd.read_csv(os.path.join(DATA_DIR,  "dataset_cleaned.csv"))
    data = pd.read_csv(os.path.join(DATA_DIR,  "dataset_final_with_soil.csv"))
    df   = df.reset_index(drop=True)
    data = data.reset_index(drop=True)

    with open(os.path.join(MODEL_DIR, "features.json")) as f:
        features = json.load(f)

    automl = joblib.load(os.path.join(MODEL_DIR, "automl.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    df_ml        = df.reindex(columns=features, fill_value=0)
    df_ml_scaled = scaler.transform(df_ml)
    yield_pred   = automl.predict(df_ml_scaled)

    return data, yield_pred

data, yield_pred = load_all()

# =========================================================
# PRÉTRAITEMENT  (mis en cache)
# =========================================================
@st.cache_data(show_spinner=False)
def preprocess(data_hash):
    crops   = data["Crop"].astype(str).unique().tolist()
    wilayas = data["Wilaya"].astype(str).unique().tolist()

    # ── Rescaling rendement ──────────────────────────────
    YIELD_KG_MIN, YIELD_KG_MAX = 500, 60_000

    yield_cache: dict = {}
    for i in range(min(len(data), len(yield_pred))):
        crop   = str(data.loc[i, "Crop"])
        wilaya = str(data.loc[i, "Wilaya"])
        v      = float(yield_pred[i])
        if (crop, wilaya) in yield_cache:
            yield_cache[(crop, wilaya)] = (yield_cache[(crop, wilaya)] + v) / 2
        else:
            yield_cache[(crop, wilaya)] = v

    def _rescale(v):
        return (YIELD_KG_MIN + v * (YIELD_KG_MAX - YIELD_KG_MIN)) / 1_000

    yield_cache = {k: _rescale(v) for k, v in yield_cache.items()}
    yield_by_crop = {}
    for (crop, _), y in yield_cache.items():
        yield_by_crop.setdefault(crop, []).append(y)
    yield_crop_mean = {c: float(np.mean(vs)) for c, vs in yield_by_crop.items()}
    global_yield_mean = float(np.mean(list(yield_crop_mean.values()))) if yield_crop_mean else 5.0

    # ── Prix ─────────────────────────────────────────────
    PRICE_MIN, PRICE_MAX = 35_000, 250_000
    _raw_price = data.groupby("Crop")["Price"].mean()
    _pmin, _pmax = _raw_price.min(), _raw_price.max()
    Price = {}
    for crop, raw_p in _raw_price.items():
        norm = 0.5 if _pmax == _pmin else (raw_p - _pmin) / (_pmax - _pmin)
        Price[crop] = PRICE_MIN + norm * (PRICE_MAX - PRICE_MIN)

    # ── Coûts ────────────────────────────────────────────
    COST_MIN, COST_MAX = 15_000, 250_000
    Water = data.groupby("Crop")["WATER REQUIREMENT"].mean().to_dict()
    N_req = data.groupby("Crop")["N"].mean().to_dict()
    P_req = data.groupby("Crop")["P"].mean().to_dict()
    K_req = data.groupby("Crop")["K"].mean().to_dict()

    _raw_int = {c: Water.get(c,0)+N_req.get(c,0)+P_req.get(c,0)+K_req.get(c,0) for c in crops}
    _imin, _imax = min(_raw_int.values()), max(_raw_int.values())
    Cost = {}
    for c in crops:
        norm = (_raw_int[c]-_imin)/(_imax-_imin) if _imax>_imin else 0.5
        Cost[c] = COST_MIN + norm*(COST_MAX-COST_MIN)

    return (crops, wilayas, yield_cache, yield_crop_mean, global_yield_mean,
            Price, Cost, Water)

(crops, wilayas,
 yield_cache, yield_crop_mean, global_yield_mean,
 Price, Cost, Water) = preprocess(len(data))

C = len(crops)
W = len(wilayas)
crop_to_idx = {c: i for i, c in enumerate(crops)}
idx_to_crop = {i: c for c, i in crop_to_idx.items()}

# =========================================================
# CLASSIFICATION AGRO
# =========================================================
PERENNIAL_KW = [
    "olive","grape","apple","pear","cherry","apricot","peach","nectarine",
    "plum","fig","almond","orange","lemon","lime","citrus","pomelo",
    "grapefruit","date","locust bean","carob","quince","other stone fruit",
    "other citrus","other tropical","other fruits n.e.c.","artichoke",
]
FAMILIES = {
    "Solanaceae":    ["tomato","eggplant","aubergine","pepper","chilli","capsicum","potato"],
    "Cucurbitaceae": ["cucumber","gherkin","pumpkin","squash","gourd","watermelon","melon"],
    "Leguminosae":   ["pea","bean","lentil","chick","vetch","broad bean","horse bean","soy"],
    "Brassicaceae":  ["cauliflower","broccoli","rape","colza","cabbage","turnip"],
    "Apiaceae":      ["carrot","fennel","celery"],
    "Asteraceae":    ["artichoke","sunflower"],
    "Poaceae":       ["wheat","barley","oat","maize","sorghum","triticale","rice"],
    "Alliaceae":     ["onion","shallot","garlic","leek"],
}
MIN_RETURN_GAP = 3

def _is_perennial(c):  return any(k in c.lower() for k in PERENNIAL_KW)
def _get_family(c):
    cl = c.lower()
    for fam, kws in FAMILIES.items():
        if any(k in cl for k in kws): return fam
    return None
def _is_legume(c): return _get_family(c) == "Leguminosae"
def _is_cereal(c): return _get_family(c) == "Poaceae"

crop_is_perennial = {c: _is_perennial(c) for c in crops}
crop_family       = {c: _get_family(c)   for c in crops}
crop_is_legume    = {c: _is_legume(c)    for c in crops}
crop_is_cereal    = {c: _is_cereal(c)    for c in crops}

def predict_yield(crop, wilaya):
    if (crop, wilaya) in yield_cache:    return yield_cache[(crop, wilaya)]
    if crop in yield_crop_mean:          return yield_crop_mean[crop]
    return global_yield_mean

# =========================================================
# PROBLÈME NSGA-II
# =========================================================
class CropRotationProblem(ElementwiseProblem):
    def __init__(self, budget, mb_min, T):
        super().__init__(n_var=W*T, n_obj=4, n_constr=6, xl=0, xu=C-1, type_var=int)
        self.budget = budget
        self.mb_min = mb_min
        self.T = T

    def _evaluate(self, x, out, *args, **kwargs):
        T = self.T
        x = np.clip(np.array(x, dtype=int), 0, C-1).reshape(W, T)

        total_yield = total_profit = total_cost = total_water = total_mb = 0.0
        for w in range(W):
            for t in range(T):
                crop   = idx_to_crop[x[w,t]]
                wilaya = wilayas[w]
                y      = predict_yield(crop, wilaya)
                cost   = Cost[crop]
                mb_wt  = y * Price.get(crop, 1) - cost
                total_yield  += y
                total_profit += mb_wt
                total_cost   += cost
                total_water  += Water.get(crop, 0)
                total_mb     += mb_wt

        rotation_score = 0
        for w in range(W):
            for t in range(1, T):
                if crop_is_cereal[idx_to_crop[x[w,t-1]]] and crop_is_legume[idx_to_crop[x[w,t]]]:
                    rotation_score += 1
        rotation_score /= (W*(T-1)+1e-9)

        out["F"] = [-total_yield, -total_profit, total_water, -rotation_score]

        # g0 budget
        g0 = (total_cost - self.budget) / (self.budget + 1e-9)

        # g1 pas de même culture consécutive
        consec_viol = sum(x[w,t]==x[w,t-1] for w in range(W) for t in range(1,T))
        g1 = consec_viol / (W*(T-1)+1e-9)

        # g2 délai de retour
        ret_viol = 0
        for w in range(W):
            for t in range(T):
                if crop_is_perennial[idx_to_crop[x[w,t]]]: continue
                for lag in range(1, min(MIN_RETURN_GAP, t+1)):
                    if x[w,t]==x[w,t-lag]: ret_viol+=1; break
        g2 = ret_viol / (W*T+1e-9)

        # g3 pérennes
        per_viol = 0
        for w in range(W):
            fc = idx_to_crop[x[w,0]]
            if crop_is_perennial[fc]:
                per_viol += sum(x[w,t]!=x[w,0] for t in range(1,T))
            else:
                per_viol += sum(crop_is_perennial[idx_to_crop[x[w,t]]] for t in range(1,T))
        g3 = per_viol / (W*T+1e-9)

        # g4 même famille
        fam_viol = sum(
            1 for w in range(W) for t in range(1,T)
            if crop_family[idx_to_crop[x[w,t-1]]] is not None
            and crop_family[idx_to_crop[x[w,t-1]]] == crop_family[idx_to_crop[x[w,t]]]
        )
        g4 = fam_viol / (W*(T-1)+1e-9)

        # g5 marge brute min
        g5 = (self.mb_min-total_mb)/(abs(self.mb_min)+1e-9) if self.mb_min>0 else -1.0

        out["G"] = [g0, g1, g2, g3, g4, g5]

# =========================================================
# SIDEBAR — NAVIGATION
# =========================================================
st.sidebar.image("https://em-content.zobj.net/source/apple/354/seedling_1f331.png", width=60)
st.sidebar.title("CropOpt")
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Navigation",
    ["🔮 Prédiction du Rendement", "⚙️ Optimisation NSGA-II", "📊 Tableau de bord"]
)
st.sidebar.markdown("---")
st.sidebar.caption(f"🌾 {C} cultures  •  🗺️ {W} wilayas")

# =========================================================
# MODE 1 : PRÉDICTION
# =========================================================
if mode == "🔮 Prédiction du Rendement":
    st.header("🔮 Prédiction du Rendement")
    st.markdown("Estimez le rendement d'une culture dans une wilaya donnée.")

    col1, col2 = st.columns(2)
    with col1:
        crop   = st.selectbox("🌱 Culture", sorted(crops))
    with col2:
        wilaya = st.selectbox("📍 Wilaya",  sorted(wilayas))

    if st.button("📊 Prédire le rendement", type="primary"):
        y    = predict_yield(crop, wilaya)
        cost = Cost.get(crop, 0)
        price= Price.get(crop, 0)
        mb   = y * price - cost

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, unit in [
            (c1, "Rendement",    f"{y:.2f}",          "t/ha"),
            (c2, "Prix marché",  f"{price:,.0f}",     "DA/t"),
            (c3, "Coût prod.",   f"{cost:,.0f}",      "DA/ha"),
            (c4, "Marge brute",  f"{mb:,.0f}",        "DA/ha"),
        ]:
            col.markdown(f"""
            <div class="metric-box">
              <p>{label}</p>
              <h2>{val}</h2>
              <p>{unit}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        if mb > 0:
            st.success(f"✅ Culture rentable dans la wilaya de **{wilaya}** avec une MB de **{mb:,.0f} DA/ha**.")
        else:
            st.warning("⚠️ Marge brute négative — envisagez une autre culture ou réduisez les coûts.")

# =========================================================
# MODE 2 : OPTIMISATION
# =========================================================
elif mode == "⚙️ Optimisation NSGA-II":
    st.header("⚙️ Optimisation Multi-Objectifs NSGA-II")
    st.markdown("Trouvez le plan de rotation optimal pour toutes les wilayas.")

    # ── Paramètres ──────────────────────────────────────
    with st.expander("⚙️ Paramètres de l'optimisation", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            T = st.slider("🗓️ Horizon (années)", 2, 10, 5)
            pop_size = st.slider("👥 Taille population", 50, 500, 200, step=50)
        with col2:
            n_gen    = st.slider("🔁 Générations", 20, 300, 100, step=10)
            mb_ratio = st.slider("📈 Seuil marge brute (%)", 0, 80, 30) / 100

    # ── Budget auto ──────────────────────────────────────
    avg_cost = float(np.mean(list(Cost.values())))
    min_bgt  = avg_cost * W * T
    max_bgt  = max(Cost.values()) * W * T

    st.markdown(f"""
    **💡 Budget recommandé :** entre `{min_bgt:,.0f}` DA et `{int(min_bgt*1.5):,.0f}` DA  
    *(basé sur {W} wilayas × {T} ans × coût moyen {avg_cost:,.0f} DA/ha)*
    """)

    budget = st.number_input(
        "💰 Budget total (DA)",
        min_value=int(min_bgt * 0.3),
        max_value=int(max_bgt * 3),
        value=int(min_bgt * 1.2),
        step=100_000,
        format="%d"
    )

    if budget < min_bgt * 0.5:
        st.warning("⚠️ Budget très serré — risque d'infaisabilité élevé.")

    # ── Lancement ────────────────────────────────────────
    if st.button("🚀 Lancer l'optimisation", type="primary"):

        # Calcul du seuil MB
        avg_price_val = float(np.mean(list(Price.values())))
        avg_mb_slot   = global_yield_mean * avg_price_val - avg_cost
        mb_total_theo = avg_mb_slot * W * T
        mb_min = mb_ratio * mb_total_theo if avg_mb_slot > 0 else 0.0

        problem = CropRotationProblem(budget=budget, mb_min=mb_min, T=T)
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        progress_bar = st.progress(0, text="Initialisation…")
        status_text  = st.empty()
        t0 = time.time()

        class ProgressCallback:
            def __call__(self, algorithm):
                gen = algorithm.n_gen
                pct = int(gen / n_gen * 100)
                elapsed = time.time() - t0
                progress_bar.progress(pct, text=f"Génération {gen}/{n_gen} — {elapsed:.0f}s écoulées")
                status_text.caption(f"🔬 Pop. courante : {len(algorithm.pop)} individus")

        cb = ProgressCallback()

        with st.spinner("Optimisation en cours…"):
            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                callback=cb,
                verbose=False,
                seed=42,
            )

        progress_bar.empty()
        status_text.empty()
        elapsed = time.time() - t0

        # ── Résultats ────────────────────────────────────
        if res.X is None:
            st.error("❌ Aucune solution faisable trouvée. Augmentez le budget ou réduisez le seuil MB.")
        else:
            n_sol = len(res.X)
            st.success(f"✅ {n_sol} solution(s) Pareto trouvée(s) en {elapsed:.1f}s.")

            # ── Tableau Pareto (toutes solutions) ──────────
            st.subheader("📋 Front de Pareto")
            pareto_data = []
            for i in range(n_sol):
                F = res.F[i]
                pareto_data.append({
                    "Solution": i+1,
                    "Rendement total (t)": round(-F[0], 2),
                    "Profit total (DA)":   int(-F[1]),
                    "Eau totale (u)":      round(F[2], 1),
                    "Score rotation":      round(-F[3], 4),
                })
            df_pareto = pd.DataFrame(pareto_data)
            st.dataframe(df_pareto, use_container_width=True)

            # ── Sélection solution à afficher ──────────────
            sol_idx = st.number_input("Afficher la solution n°", 1, n_sol, 1) - 1
            best_x  = np.array(res.X[sol_idx], dtype=int).reshape(W, T)
            best_F  = res.F[sol_idx]

            # ── Métriques clés ─────────────────────────────
            st.subheader(f"📊 Métriques — Solution {sol_idx+1}")
            total_slots = W * T
            c1,c2,c3,c4 = st.columns(4)
            metrics = [
                (c1, "Rendement total",  f"{-best_F[0]:,.1f}", "t"),
                (c2, "Profit total",     f"{-best_F[1]:,.0f}", "DA"),
                (c3, "Eau totale",       f"{best_F[2]:,.0f}",  "u"),
                (c4, "Score rotation",   f"{-best_F[3]:.4f}",  ""),
            ]
            for col, label, val, unit in metrics:
                col.markdown(f"""<div class="metric-box">
                  <p>{label}</p><h2>{val}</h2><p>{unit}</p>
                </div>""", unsafe_allow_html=True)

            # ── Plan de rotation ───────────────────────────
            st.subheader("🌱 Plan de rotation optimal")
            rotation_df = pd.DataFrame(
                {f"Année {t+1}": [idx_to_crop[best_x[w,t]] for w in range(W)] for t in range(T)},
                index=wilayas
            )
            rotation_df.index.name = "Wilaya"
            st.dataframe(rotation_df, use_container_width=True)

            # ── Vérification post-hoc ──────────────────────
            st.subheader("🔍 Vérification des contraintes")
            consec_ok = all(best_x[w,t]!=best_x[w,t-1] for w in range(W) for t in range(1,T))
            ret_ok = True
            for w in range(W):
                for t in range(T):
                    if crop_is_perennial[idx_to_crop[best_x[w,t]]]: continue
                    for lag in range(1, min(MIN_RETURN_GAP, t+1)):
                        if best_x[w,t]==best_x[w,t-lag]: ret_ok=False
            per_ok = True
            for w in range(W):
                fc = idx_to_crop[best_x[w,0]]
                if crop_is_perennial[fc]:
                    if any(best_x[w,t]!=best_x[w,0] for t in range(1,T)): per_ok=False
                else:
                    if any(crop_is_perennial[idx_to_crop[best_x[w,t]]] for t in range(1,T)): per_ok=False
            fam_ok = all(
                crop_family[idx_to_crop[best_x[w,t-1]]] is None
                or crop_family[idx_to_crop[best_x[w,t-1]]]!=crop_family[idx_to_crop[best_x[w,t]]]
                for w in range(W) for t in range(1,T)
            )
            checks = [
                ("Anti-monoculture consécutive", consec_ok),
                (f"Délai de retour ≥ {MIN_RETURN_GAP} ans (annuelles)", ret_ok),
                ("Exclusion mutuelle pérennes",   per_ok),
                ("Succession même famille",        fam_ok),
            ]
            for label, ok in checks:
                if ok: st.success(f"✅ {label}")
                else:  st.error(f"❌ {label} — violation détectée")

            # ── Export CSV ────────────────────────────────
            csv = rotation_df.to_csv(encoding="utf-8-sig")
            st.download_button(
                "⬇️ Télécharger le plan (CSV)",
                data=csv,
                file_name=f"rotation_solution_{sol_idx+1}.csv",
                mime="text/csv",
            )

# =========================================================
# MODE 3 : TABLEAU DE BORD
# =========================================================
else:
    st.header("📊 Tableau de bord — Aperçu des données")

    tab1, tab2, tab3 = st.tabs(["🌾 Cultures", "🗺️ Wilayas", "💰 Économie"])

    with tab1:
        st.subheader("Rendement moyen par culture (t/ha)")
        df_yield = pd.DataFrame([
            {"Culture": c, "Rendement moyen (t/ha)": round(v, 2)}
            for c, v in sorted(yield_crop_mean.items(), key=lambda x: -x[1])
        ])
        st.dataframe(df_yield, use_container_width=True, height=400)

    with tab2:
        st.subheader("Distribution des cultures par wilaya")
        counts = data.groupby("Wilaya")["Crop"].nunique().reset_index()
        counts.columns = ["Wilaya", "Nb cultures distinctes"]
        st.dataframe(counts.sort_values("Nb cultures distinctes", ascending=False),
                     use_container_width=True)

    with tab3:
        st.subheader("Prix & Coûts estimés par culture")
        econ = pd.DataFrame([{
            "Culture":       c,
            "Prix (DA/t)":   int(Price.get(c, 0)),
            "Coût (DA/ha)":  int(Cost.get(c, 0)),
            "Rdt moyen (t/ha)": round(yield_crop_mean.get(c, global_yield_mean), 2),
            "MB estimée (DA/ha)": int(
                yield_crop_mean.get(c, global_yield_mean) * Price.get(c, 0) - Cost.get(c, 0)
            ),
        } for c in sorted(crops)])
        econ = econ.sort_values("MB estimée (DA/ha)", ascending=False)
        st.dataframe(econ, use_container_width=True, height=400)

        pos = (econ["MB estimée (DA/ha)"] > 0).sum()
        st.info(f"✅ {pos}/{len(crops)} cultures ont une marge brute estimée positive.")
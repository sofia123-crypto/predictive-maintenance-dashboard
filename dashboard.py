
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ─── Chargement des données ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("predictions.csv")
    sensors = pd.read_csv("sensor_data.csv")
    return df, sensors

df, sensors = load_data()

y_test = df["real_rul"].values
y_pred = df["predicted_rul"].values

def health_status(rul):
    if rul <= 30:   return "CRITICAL", "#e74c3c", "🔴"
    elif rul <= 60: return "WARNING",  "#f39c12", "🟡"
    else:           return "NORMAL",   "#2ecc71", "🟢"

# ─── En-tête ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1F3864, #2E75B6);
            padding: 25px; border-radius: 12px; margin-bottom: 25px;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2em;">
        🔧 Smart Predictive Maintenance Dashboard
    </h1>
    <p style="color: #BDC3C7; text-align: center; margin: 8px 0 0 0; font-size: 1em;">
        NASA CMAPSS FD004 · LSTM Deep Learning · Raspberry Pi Simulation
    </p>
</div>
""", unsafe_allow_html=True)

# ─── KPIs globaux ────────────────────────────────────────────────────────
n_normal   = sum(1 for p in y_pred if p > 60)
n_warning  = sum(1 for p in y_pred if 30 < p <= 60)
n_critical = sum(1 for p in y_pred if p <= 30)
total      = len(y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("🔢 Total",    total)
c2.metric("🟢 Normal",   n_normal)
c3.metric("🟡 Warning",  n_warning)
c4.metric("🔴 Critical", n_critical)
c5.metric("📊 RMSE",     f"{rmse:.2f}")
c6.metric("📐 MAE",      f"{mae:.2f}")

st.markdown("---")

# ─── Layout principal ────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("🔍 Sélection du moteur")

    engine_id = st.selectbox(
        "Choisir un moteur :",
        df["engine_id"].values,
        format_func=lambda x: f"Moteur {x}"
    )

    row = df[df["engine_id"] == engine_id].iloc[0]
    rul_pred = int(row["predicted_rul"])
    rul_real = int(row["real_rul"])
    status, color, emoji = health_status(rul_pred)

    # Carte de statut
    st.markdown(f"""
    <div style="background-color:{color}22; border-left:5px solid {color};
                padding:20px; border-radius:10px; margin:15px 0;">
        <h2 style="color:{color}; margin:0;">{emoji} {status}</h2>
        <hr style="border-color:{color}44; margin:10px 0;">
        <p style="font-size:20px; margin:5px 0;">
            <b>RUL Prédit :</b> {rul_pred} cycles
        </p>
        <p style="font-size:16px; color:gray; margin:0;">
            RUL Réel : {rul_real} cycles &nbsp;|&nbsp;
            Erreur : {abs(rul_pred - rul_real)} cycles
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Recommandation
    st.subheader("📋 Recommandation")
    if status == "CRITICAL":
        st.error("""
        ⚠️ **INTERVENTION URGENTE requise !**
        
        - Arrêter la machine dès que possible
        - Contacter l'équipe de maintenance
        - Risque de panne dans < 30 cycles
        """)
    elif status == "WARNING":
        st.warning("""
        🔔 **Maintenance préventive à planifier**
        
        - Programmer l'intervention sous 2 semaines
        - Commander les pièces de rechange
        - Augmenter la fréquence de surveillance
        """)
    else:
        st.success("""
        ✅ **Moteur en bon état**
        
        - Aucune intervention requise
        - Surveillance continue recommandée
        - Prochain contrôle selon calendrier
        """)

    # Infos embedded
    st.subheader("🔌 Déploiement embarqué")
    st.info("""
    **Plateforme :** Raspberry Pi 4
    
    **Modèle :** LSTM FD004 (Keras)
    
    **Format embarqué :** TFLite
    
    **Taille originale :** 1693 KB
    
    **Taille TFLite :** 192 KB (-88%)
    
    **Latence :** ~126 ms/prédiction
    
    **RAM utilisée :** 25% du RPi 4
    
    **CPU :** 5.1%
    """)

with right_col:
    tab1, tab2, tab3 = st.tabs([
        "📈 Capteurs",
        "📊 Analyse globale",
        "🏥 Tableau de santé"
    ])

    with tab1:
        st.subheader(f"Évolution des capteurs — Moteur {engine_id}")

        # Récupérer les 30 derniers cycles du moteur sélectionné
        sensor_row = sensors[sensors["engine_id"] == engine_id].iloc[0]
        sensor_values = sensor_row.drop("engine_id").values.astype(float)
        sensor_matrix = sensor_values.reshape(30, 17)

        # Noms des capteurs actifs FD004
        sensor_names = ["s2","s3","s4","s6","s7","s8",
                        "s9","s11","s12","s13","s14","s15",
                        "s17","s19","s20","s21","s7"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        plot_sensors = [0, 1, 2, 7, 8, 11]  # s2,s3,s4,s11,s12,s15
        plot_names   = ["s2","s3","s4","s11","s12","s15"]
        plot_colors  = ["#2E75B6","#e74c3c","#2ecc71",
                        "#f39c12","#9b59b6","#1abc9c"]

        for i, (s_idx, s_name, c) in enumerate(
                zip(plot_sensors, plot_names, plot_colors)):
            axes[i].plot(sensor_matrix[:, s_idx],
                        color=c, linewidth=2.5, marker="o",
                        markersize=3)
            axes[i].fill_between(range(30), sensor_matrix[:, s_idx],
                                alpha=0.15, color=c)
            axes[i].set_title(f"Capteur {s_name}", fontweight="bold")
            axes[i].set_xlabel("Cycle (fenêtre 30)")
            axes[i].set_ylabel("Valeur normalisée")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)

        plt.suptitle(
            f"Moteur {engine_id} · RUL={rul_pred} cycles · {emoji} {status}",
            fontweight="bold", fontsize=13
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Analyse globale — 248 moteurs")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Scatter plot coloré par statut
        scatter_colors = [health_status(p)[1] for p in y_pred]
        axes[0].scatter(y_test, y_pred, c=scatter_colors,
                       alpha=0.7, edgecolors="white", s=60)
        axes[0].plot([0,125],[0,125],"r--",
                    label="Prédiction parfaite", linewidth=2)
        axes[0].set_xlabel("RUL Réel (cycles)")
        axes[0].set_ylabel("RUL Prédit (cycles)")
        axes[0].set_title(f"Réel vs Prédit\nRMSE={rmse:.2f} · MAE={mae:.2f}",
                         fontweight="bold")
        patches = [
            mpatches.Patch(color="#2ecc71", label=f"Normal ({n_normal})"),
            mpatches.Patch(color="#f39c12", label=f"Warning ({n_warning})"),
            mpatches.Patch(color="#e74c3c", label=f"Critical ({n_critical})"),
        ]
        axes[0].legend(handles=patches)
        axes[0].grid(True, alpha=0.3)

        # Pie chart
        sizes  = [n_normal, n_warning, n_critical]
        labels = [f"Normal\n{n_normal} ({n_normal/total*100:.0f}%)",
                  f"Warning\n{n_warning} ({n_warning/total*100:.0f}%)",
                  f"Critical\n{n_critical} ({n_critical/total*100:.0f}%)"]
        axes[1].pie(sizes, labels=labels,
                   colors=["#2ecc71","#f39c12","#e74c3c"],
                   autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize":11})
        axes[1].set_title("Distribution des statuts", fontweight="bold")

        # Histogramme des erreurs
        errors = y_pred - y_test
        axes[2].hist(errors, bins=30, color="#2E75B6",
                    edgecolor="white", alpha=0.85)
        axes[2].axvline(x=0, color="red", linestyle="--",
                       linewidth=2, label="Erreur = 0")
        axes[2].axvline(x=errors.mean(), color="orange",
                       linestyle="--", linewidth=2,
                       label=f"Moyenne = {errors.mean():.1f}")
        axes[2].set_xlabel("Erreur (Prédit - Réel)")
        axes[2].set_ylabel("Fréquence")
        axes[2].set_title("Distribution des erreurs", fontweight="bold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Métriques comparatives FD001 vs FD004
        st.subheader("📊 Comparaison FD001 vs FD004")
        comp_df = pd.DataFrame({
            "Dataset"   : ["FD001 (Simple)", "FD004 (Complexe)"],
            "Conditions": [1, 6],
            "Moteurs"   : [100, 249],
            "RMSE"      : [12.58, rmse],
            "MAE"       : [9.61, mae],
            "Difficulte": ["⭐ Facile", "⭐⭐⭐⭐ Très difficile"]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("🏥 Tableau de santé — Tous les moteurs")

        # Ajouter statut au dataframe
        df["statut"] = [health_status(p)[2] + " " + health_status(p)[0]
                       for p in y_pred]
        df["erreur_abs"] = abs(df["error"])

        # Filtres
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            status_filter = st.selectbox(
                "Filtrer par statut :",
                ["Tous", "🔴 CRITICAL", "🟡 WARNING", "🟢 NORMAL"]
            )
        with col_f2:
            sort_by = st.selectbox(
                "Trier par :",
                ["engine_id", "predicted_rul", "erreur_abs"]
            )

        display_df = df.copy()
        if status_filter != "Tous":
            display_df = display_df[
                display_df["statut"].str.contains(
                    status_filter.split()[1])]

        display_df = display_df.sort_values(sort_by)

        st.dataframe(
            display_df[["engine_id","real_rul",
                        "predicted_rul","error","statut"]].rename(columns={
                "engine_id"     : "Moteur",
                "real_rul"      : "RUL Réel",
                "predicted_rul" : "RUL Prédit",
                "error"         : "Erreur",
                "statut"        : "Statut"
            }),
            use_container_width=True,
            height=400,
            hide_index=True
        )
        st.caption(f"Affichage : {len(display_df)} moteurs")

# ─── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:12px; padding:10px;">
    🔧 Smart Embedded Predictive Maintenance System<br>
    NASA CMAPSS FD004 · LSTM Deep Learning · TensorFlow Lite · Raspberry Pi · 2024/2025
</div>
""", unsafe_allow_html=True)

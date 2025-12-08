import streamlit as st
import joblib
import pandas as pd
import altair as alt  

-- Model yükleme --
model = joblib.load("models/final_model.pkl")

st.title("🩺 Diyabet Risk Tahmin Uygulaması")
st.write("Aşağıdaki bilgileri doldurarak tahmini diyabet riskinizi ve dünya ortalamasıyla karşılaştırmasını görebilirsiniz.")

# YAŞ GRUPLARI
yas_map = {
    "18–24": 1,
    "25–34": 3,      # 25–29 (2) ve 30–34 (3) arası
    "35–44": 5,      # 35–39 (4) ve 40–44 (5)
    "45–54": 7,      # 45–49 (6) ve 50–54 (7)
    "55–64": 9,      # 55–59 (8) ve 60–64 (9)
    "65–74": 11,     # 65–69 (10) ve 70–74 (11)
    "75 ve üzeri": 13  # 75–79 (12) ve 80+ (13)
}

# EĞİTİM
egitim_map = {
    "İlkokul veya daha az": 1,
    "Ortaokul": 2,
    "Lise": 3,
    "Üniversite (devam ediyor)": 4,
    "Üniversite mezunu": 5,
    "Lisansüstü (Master/Doktora)": 6
}

# GELİR 
gelir_map = {
    "Asgari Ücret Altı (0–22.000 TL)": 1,
    "Asgari Ücret – 2× Asgari Ücret (22.000–44.000 TL)": 2,
    "2× – 3× Asgari Ücret (44.000–66.000 TL)": 3,
    "Orta Gelir (66.000–100.000 TL)": 4,
    "Üst-Orta Gelir (100.000–150.000 TL)": 5,
    "Yüksek Gelir (150.000–250.000 TL)": 6,
    "Çok Yüksek Gelir (250.000–400.000 TL)": 7,
    "Ultra Yüksek Gelir (400.000 TL üzeri)": 8
}

# --- FORM ---

st.subheader("Sağlık Bilgileri")

HighBP = st.selectbox("Yüksek Tansiyon", ["Hayır", "Evet"])
HighChol = st.selectbox("Yüksek Kolesterol", ["Hayır", "Evet"])
CholCheck = st.selectbox("Kolesterol Kontrolü Yapıldı mı?", ["Hayır", "Evet"])
BMI = st.number_input("Vücut Kitle İndeksi (BMI)", min_value=10.0, max_value=70.0, step=0.1)
Smoker = st.selectbox("Sigara Kullanımı", ["Hayır", "Evet"])
Stroke = st.selectbox("Geçirilmiş Felç", ["Hayır", "Evet"])
Heart = st.selectbox("Kalp Hastalığı veya Kalp Krizi", ["Hayır", "Evet"])
PhysActivity = st.selectbox("Düzenli Fiziksel Aktivite", ["Hayır", "Evet"])
Fruits = st.selectbox("Düzenli Meyve Tüketimi", ["Hayır", "Evet"])
Veggies = st.selectbox("Düzenli Sebze Tüketimi", ["Hayır", "Evet"])
Alcohol = st.selectbox("Aşırı Alkol Tüketimi", ["Hayır", "Evet"])
AnyHealthcare = st.selectbox("Herhangi Bir Sağlık Sigortası / Sağlık Hizmeti", ["Hayır", "Evet"])
NoDoc = st.selectbox("Ücret Nedeniyle Doktora Gidememe", ["Hayır", "Evet"])

st.subheader("Genel Sağlık Durumu")
GenHlth = st.slider("Genel Sağlık (1=Mükemmel, 5=Kötü)", 1, 5)
MentHlth = st.slider("Ruhsal Sağlık Problemi (son 30 günde, gün)", 0, 30)
PhysHlth = st.slider("Fiziksel Sağlık Problemi (son 30 günde, gün)", 0, 30)
DiffWalk = st.selectbox("Yürümede Zorluk", ["Hayır", "Evet"])

st.subheader("Demografik Bilgiler")
Sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
Age = st.selectbox("Yaş Grubu", list(yas_map.keys()))
Education = st.selectbox("Eğitim Düzeyi", list(egitim_map.keys()))
Income = st.selectbox("Gelir Düzeyi", list(gelir_map.keys()))

# DÜNYA ORTALAMASI (örnek ~%10 prevalans)
WORLD_AVG = 0.10

if st.button("💡 Diyabet Riskimi Hesapla"):
    # Modelin beklediği formatta input dataframe
    df_input = pd.DataFrame([[
        1 if HighBP=="Evet" else 0,
        1 if HighChol=="Evet" else 0,
        1 if CholCheck=="Evet" else 0,
        BMI,
        1 if Smoker=="Evet" else 0,
        1 if Stroke=="Evet" else 0,
        1 if Heart=="Evet" else 0,
        1 if PhysActivity=="Evet" else 0,
        1 if Fruits=="Evet" else 0,
        1 if Veggies=="Evet" else 0,
        1 if Alcohol=="Evet" else 0,
        1 if AnyHealthcare=="Evet" else 0,
        1 if NoDoc=="Evet" else 0,
        GenHlth,
        MentHlth,
        PhysHlth,
        1 if DiffWalk=="Evet" else 0,
        1 if Sex=="Erkek" else 0,
        yas_map[Age],
        egitim_map[Education],
        gelir_map[Income]
    ]], columns=[
        'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
        'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
        'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
        'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'
    ])

    # Tahmin
    prob = model.predict_proba(df_input)[0][1]
    risk_yuzde = round(prob * 100, 2)
    world_yuzde = WORLD_AVG * 100

    st.success(f"🎯 Tahmini diyabet riskiniz: **%{risk_yuzde}**")

    # Risk seviyesine göre mesaj
    if prob > 0.60:
        st.error("⚠️ Yüksek risk: Bir sağlık profesyoneline danışmanız önerilir.")
    elif prob > 0.30:
        st.warning("Orta risk: Yaşam tarzınızı ve risk faktörlerinizi gözden geçirmeniz faydalı olabilir.")
    else:
        st.info("Düşük risk: Mevcut durumunuz görece olarak düşük riskli görünüyor.")

    st.markdown("---")
    st.subheader("📊 Risk Karşılaştırması: Siz vs Dünya Ortalaması")

    # Grafik için dataframe
    chart_df = pd.DataFrame({
        "Kategori": ["Sizin Tahmini Riskiniz", "Dünya Ortalaması"],
        "Risk (%)": [risk_yuzde, world_yuzde]
    })

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Kategori", sort=None, title=""),
            y=alt.Y("Risk (%)", title="Risk (%)"),
            color="Kategori"
        )
    )

    st.altair_chart(chart, use_container_width=True)

    st.caption("ℹ️ Dünya ortalaması, genel diyabet prevalansı için yaklaşık %10 kabul edilerek gösterilmiştir.")

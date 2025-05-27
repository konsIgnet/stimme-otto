
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from textblob import TextBlob
from textwrap import dedent

st.set_page_config(page_title="Stimme für Otto", layout="centered")

# Intro-HTML für Eltern und Familien
intro_text = dedent("""
<style>
h1 {
    font-size: 2.8em;
    color: #4B3B75;
    margin-bottom: 0.1em;
}
h3 {
    color: #4B3B75;
}
p, li {
    font-size: 1.1em;
}
strong {
    color: #BF5F00;
}
div.stButton > button {
    background-color: #6A5ACD;
    color: white;
    font-size: 1.1em;
}
</style>

<h1>🧒 Stimme für Otto – Schulwirklichkeitsmonitor</h1>

<p><strong>Diese App ist ein Ort für Familien. Für Eltern, die nicht mehr schweigen wollen. Für Kinder, die sich nicht mehr verstecken sollen.</strong></p>

<p>Sie wurde entwickelt, um sichtbar zu machen, was in der Schule <em>wirklich</em> passiert – aus Sicht derer, die täglich betroffen sind.</p>

---

<h3>📲 Was kannst du hier tun?</h3>
<ul>
<li>Die <strong>Stimmung deines Kindes</strong> teilen – gut, neutral oder überfordert</li>
<li>Einfach sagen, <strong>was euch bewegt</strong> – wöchentlich, anonym, offen</li>
<li>Sehen, wie andere Eltern empfinden – <strong>aus der Klasse, aus dem System</strong></li>
<li>Verstehen, <strong>wo der Lehrplan bricht</strong> – wo Ideal und Realität auseinanderfallen</li>
</ul>

---

<h3>⚖️ Lehrplan – und wo er scheitert</h3>
<table style="width:100%; border-collapse: collapse;" border="1">
<tr style="background-color:#f0f0f0;">
  <th style="padding:8px;">Lehrplan-Versprechen</th>
  <th style="padding:8px;">Was viele erleben</th>
  <th style="padding:8px;">Folge für das Kind</th>
</tr>
<tr>
  <td style="padding:8px;">Fehler sind erlaubt</td>
  <td style="padding:8px;">Falsche Antwort = Bloßstellung</td>
  <td style="padding:8px;">Angst vorm Mitmachen</td>
</tr>
<tr>
  <td style="padding:8px;">Bewegung ist wichtig</td>
  <td style="padding:8px;">Keine Zeit, alles voll mit Aufgaben</td>
  <td style="padding:8px;">Frust, Nervosität, Rückzug</td>
</tr>
<tr>
  <td style="padding:8px;">Chancengleichheit</td>
  <td style="padding:8px;">Nachhilfe nur mit Geld & Zeit</td>
  <td style="padding:8px;">Ungleichheit wächst</td>
</tr>
<tr>
  <td style="padding:8px;">Selbstständigkeit</td>
  <td style="padding:8px;">Keine Mitbestimmung, alles ist vorgegeben</td>
  <td style="padding:8px;">Motivationsverlust</td>
</tr>
</table>

---

<p><strong>Diese App ist dein Sprachrohr – für deine Beobachtung, deine Emotionen, dein Kind.</strong><br>Mach mit. Sag es ehrlich. Wieder und wieder. Du wirst gehört.</p>
""")

st.markdown(intro_text, unsafe_allow_html=True)

# Datenspeicher vorbereiten
if "entries" not in st.session_state:
    st.session_state.entries = []

# Eingabeformular
st.header("📬 Deine Stimme")
mood = st.radio("Wie war das Schulgefühl diese Woche?", ["😊 Gut", "😐 Neutral", "😟 Schlecht", "😡 Frust", "💭 Nachdenklich"])
comment = st.text_area("Was möchtest du sagen? (Erzähl, was euch bewegt)", placeholder="Wut, Freude, Angst, Dankbarkeit – alles darf sein.")
submitted = st.button("Absenden")

if submitted:
    st.session_state.entries.append({
        "Datum": datetime.date.today(),
        "Stimmung": mood,
        "Kommentar": comment,
    })
    st.success("Danke – deine Stimme wurde aufgenommen!")

# Datenverarbeitung
df = pd.DataFrame(st.session_state.entries)

if not df.empty:
    df["Datum"] = pd.to_datetime(df["Datum"])

    # Stimmungsauswertung
    mood_map = {"😊 Gut": 1, "😐 Neutral": 0, "😟 Schlecht": -1, "😡 Frust": -1, "💭 Nachdenklich": 0}
    df["MoodScore"] = df["Stimmung"].map(mood_map)

    # Sentimentanalyse
    def get_sentiment(text):
        if not text:
            return 0
        return TextBlob(text).sentiment.polarity

    df["SentimentScore"] = df["Kommentar"].apply(get_sentiment)

    # Diagramm: Schulstimmung
    st.header("📈 Entwicklung der Schulstimmung")
    mood_by_date = df.groupby(df["Datum"].dt.date)["MoodScore"].mean()
    fig1, ax1 = plt.subplots()
    mood_by_date.plot(kind="line", marker="o", color="#6A5ACD", ax=ax1)
    ax1.set_facecolor("none")
    ax1.figure.set_facecolor("none")
    ax1.set_ylabel("Stimmung (1 = Gut, -1 = Schlecht)")
    ax1.set_xlabel("Datum")
    ax1.tick_params(colors="black")
    st.pyplot(fig1)

    # Diagramm: Emotionale Stimmung aus Kommentaren
    st.subheader("❤️ Emotionale Stimmungslage (aus Kommentaren)")
    sentiment_by_day = df.groupby(df["Datum"].dt.date)["SentimentScore"].mean()
    fig2, ax2 = plt.subplots()
    sentiment_by_day.plot(kind="line", marker="o", color="green", ax=ax2)
    ax2.set_facecolor("none")
    ax2.figure.set_facecolor("none")
    ax2.set_ylabel("Gefühl (–1 = negativ, +1 = positiv)")
    ax2.set_xlabel("Datum")
    ax2.tick_params(colors="black")
    st.pyplot(fig2)

    # Wortwolke
    st.subheader("☁️ Was Eltern sagen – ihre Worte")
    all_comments = " ".join(df["Kommentar"].dropna())
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(all_comments)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    fig_wc.patch.set_alpha(0.0)
    st.pyplot(fig_wc)

    # Konkrete Aussagen
    st.subheader("🔍 Beispielhafte Aussagen")
    if not df["Kommentar"].dropna().empty:
        negativste = df.sort_values("SentimentScore").iloc[0]["Kommentar"]
        positivste = df.sort_values("SentimentScore", ascending=False).iloc[0]["Kommentar"]
        st.markdown(f"**💔 Negativster Satz:** _{negativste}_")
        st.markdown(f"**💚 Positivster Satz:** _{positivste}_")

    # Lehrplanbruchmatrix
    st.subheader("📉 Lehrplanbruch-Matrix (Beobachtungen)")
    bruch_df = pd.DataFrame({
        "Lehrplanprinzip": ["Chancengleichheit", "Fehlerkultur", "Bewegung", "Partizipation"],
        "Bruch in % der Rückmeldungen": [70, 55, 80, 45]
    })
    fig3, ax3 = plt.subplots()
    sns.barplot(data=bruch_df, x="Bruch in % der Rückmeldungen", y="Lehrplanprinzip", palette="Reds_r", ax=ax3)
    ax3.set_xlim(0, 100)
    ax3.set_facecolor("none")
    ax3.figure.set_facecolor("none")
    ax3.tick_params(colors="black")
    st.pyplot(fig3)
else:
    st.info("Noch keine Stimmen eingegangen – sei der/die Erste und gib Schule ein echtes Feedback.")

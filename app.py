import base64
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from evaluation import (
    BENCHMARK_QUESTIONS,
    render_evaluation_panel,
    set_evaluation_candidate,
)

# OpenAI imports for the cloud AI module.
try:
    from openai_ai import ask_openai
except Exception:
    ask_openai = None

OPENAI_VECTOR_STORE_SECRET = "OPENAI_VECTOR_STORE_ID"


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Cyprus Solution Landscape Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# File paths
# ============================================================

DATA_FILE = Path("cyprus_master_dataset_v3.xlsx")
TRANSLATIONS_FILE = Path("translations.csv")

# Optional logo files.
# The app still runs if one or both are missing.
GSP_LOGO = Path("gsp_logo.png")
UCFS_LOGO = Path("ucfs_logo.png")


# ============================================================
# Responsive CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            padding-top: 1.8rem;
        }

        h1 {
            font-size: 1.55rem !important;
            line-height: 1.25 !important;
        }

        h2, h3 {
            font-size: 1.15rem !important;
        }

        .stAlert {
            font-size: 0.85rem;
        }

        [data-testid="stSidebar"] {
            min-width: 260px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load data
# ============================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        st.stop()

    df = pd.read_excel(DATA_FILE)

    required_columns = {
        "year",
        "community",
        "theme",
        "variable",
        "question_text",
        "response_category",
        "percent",
        "source_file",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        st.error(f"Missing required columns in data file: {sorted(missing)}")
        st.stop()

    return df


def load_translations() -> pd.DataFrame:
    """
    Do not cache translations.

    This avoids Streamlit keeping an old translations.csv after GitHub updates.
    """
    if not TRANSLATIONS_FILE.exists():
        st.error(f"Translation file not found: {TRANSLATIONS_FILE}")
        st.stop()

    return pd.read_csv(TRANSLATIONS_FILE)


df = load_data()
translations = load_translations()


# ============================================================
# Language / translation
# ============================================================

language = st.sidebar.selectbox(
    "Language / Γλώσσα / Dil",
    ["English", "Greek", "Turkish"],
)


def tr(key: str) -> str:
    """Translate a key into the selected interface language."""
    row = translations.loc[translations["key"] == key]
    if row.empty:
        return key

    value = row.iloc[0].get(language, key)
    return key if pd.isna(value) else str(value)


# ============================================================
# Logo helpers
# ============================================================

def image_to_data_uri(path: Path) -> str | None:
    """Convert a local PNG/JPG/SVG logo into a data URI for Plotly downloads."""
    if not path.exists():
        return None

    ext = path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    if ext == "svg":
        mime = "svg+xml"
    else:
        mime = ext

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return None


gsp_logo_uri = image_to_data_uri(GSP_LOGO)
ucfs_logo_uri = image_to_data_uri(UCFS_LOGO)


def show_logo_header() -> None:
    """
    Compact institutional logo header with safe top spacing.
    Prevents logos being clipped at the top of the Streamlit page.
    """
    if not GSP_LOGO.exists() and not UCFS_LOGO.exists():
        return

    # Safe top spacer to avoid clipping under Streamlit's top toolbar.
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    left, middle, right = st.columns([1.2, 3, 1.2])

    with left:
        if GSP_LOGO.exists():
            st.image(str(GSP_LOGO), width=130)

    with right:
        if UCFS_LOGO.exists():
            st.image(str(UCFS_LOGO), width=130)

    # Space between logos and title.
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


def add_logos_to_figure(fig):
    """
    Add compact logos to Plotly figures so downloaded PNGs include branding.

    Logos are intentionally small to avoid crowding chart titles or data.
    """
    if gsp_logo_uri:
        fig.add_layout_image(
            dict(
                source=gsp_logo_uri,
                xref="paper",
                yref="paper",
                x=0,
                y=1.10,
                sizex=0.12,
                sizey=0.12,
                xanchor="left",
                yanchor="top",
                layer="above",
            )
        )

    if ucfs_logo_uri:
        fig.add_layout_image(
            dict(
                source=ucfs_logo_uri,
                xref="paper",
                yref="paper",
                x=1,
                y=1.10,
                sizex=0.12,
                sizey=0.12,
                xanchor="right",
                yanchor="top",
                layer="above",
            )
        )

    top_margin = 80 if (gsp_logo_uri or ucfs_logo_uri) else 40
    fig.update_layout(margin=dict(l=20, r=20, t=top_margin, b=30))
    return fig


PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "cyprus_solution_landscape",
        "height": 700,
        "width": 1100,
        "scale": 3,
    },
}


# ============================================================
# Stable labels and colour mappings
# ============================================================

community_label = {
    "GC": tr("gc"),
    "TC": tr("tc"),
}
community_reverse = {v: k for k, v in community_label.items()}

solution_order = [
    "bbf_support",
    "unitary_state_support",
    "two_states_support",
    "status_quo_support",
]

solution_label = {key: tr(key) for key in solution_order}
solution_reverse = {v: k for k, v in solution_label.items()}

response_label = {
    "against": tr("against"),
    "tolerate": tr("tolerate"),
    "in_favor": tr("in_favor"),
}

binary_label = {
    "accepted": tr("accepted"),
    "rejected": tr("rejected"),
}

# Required colour scheme:
# In favor / Accepted = light green
# Tolerate = dark green
# Against / Rejected = red
response_colors = {
    tr("in_favor"): "lightgreen",
    tr("tolerate"): "darkgreen",
    tr("against"): "red",
}

binary_colors = {
    tr("accepted"): "lightgreen",
    tr("rejected"): "red",
}

# Required solution colours:
# BBF = green
# Unitary State = orange
# Two States = red
# Keeping Status Quo = pink
solution_colors = {
    tr("bbf_support"): "green",
    tr("unitary_state_support"): "orange",
    tr("two_states_support"): "red",
    tr("status_quo_support"): "pink",
}


# ============================================================
# Derived accepted / rejected data
# ============================================================

accepted = (
    df[df["response_category"].isin(["in_favor", "tolerate"])]
    .groupby(["year", "community", "variable"], as_index=False)["percent"]
    .sum()
    .rename(columns={"percent": "accepted"})
)

rejected = (
    df[df["response_category"] == "against"]
    [["year", "community", "variable", "percent"]]
    .rename(columns={"percent": "rejected"})
)

df_binary = pd.merge(
    accepted,
    rejected,
    on=["year", "community", "variable"],
    how="inner",
)


# ============================================================
# Header
# ============================================================

show_logo_header()

st.title(tr("app_title"))
st.caption(tr("app_subtitle"))

st.markdown(
    f"""
    **{tr("method_data_label")}:** {tr("method_data_text")}  
    **{tr("method_measure_label")}:** {tr("method_measure_text")}  
    **{tr("method_derived_label")}:** {tr("method_derived_text")}  
    **{tr("method_note_label")}:** {tr("method_note_text")}
    """
)


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.header(tr("controls"))

display_mode = st.sidebar.radio(
    tr("display"),
    [tr("desktop"), tr("mobile")],
    index=0,
    help=tr("display_help"),
)

community_options = [tr("both"), tr("gc"), tr("tc")]
selected_community_label = st.sidebar.selectbox(
    tr("community"),
    community_options,
    index=0,
)

solution_options = [solution_label[key] for key in solution_order]
selected_solution_label = st.sidebar.selectbox(
    tr("solution"),
    solution_options,
    index=0,
)

view_mode = st.sidebar.radio(
    tr("view_mode"),
    [tr("accepted_rejected"), tr("full_distribution")],
    index=0,
)

selected_variable = solution_reverse[selected_solution_label]

if selected_community_label == tr("both"):
    selected_community = "Both"
else:
    selected_community = community_reverse[selected_community_label]

mobile_mode = display_mode == tr("mobile")


# ============================================================
# Helper functions
# ============================================================

def chart_height() -> int:
    return 430 if mobile_mode else 520


def prepare_distribution_data(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["community_label"] = out["community"].map(community_label)
    out["response_category_label"] = out["response_category"].map(response_label)
    out["solution_label"] = out["variable"].map(solution_label)
    return out


def prepare_binary_data(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["community_label"] = out["community"].map(community_label)
    out["solution_label"] = out["variable"].map(solution_label)
    return out


def render_distribution_line_chart(data: pd.DataFrame) -> None:
    plot_data = prepare_distribution_data(data)

    fig = px.line(
        plot_data,
        x="year",
        y="percent",
        color="response_category_label",
        markers=True,
        color_discrete_map=response_colors,
        labels={
            "year": tr("year"),
            "percent": tr("percent"),
            "response_category_label": tr("category"),
        },
        title=None,
    )

    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        height=chart_height(),
        legend_title_text=tr("category"),
    )
    fig = add_logos_to_figure(fig)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_binary_line_chart(data: pd.DataFrame) -> None:
    plot_data = prepare_binary_data(data)

    melted = plot_data.melt(
        id_vars=["year", "community", "community_label", "variable", "solution_label"],
        value_vars=["accepted", "rejected"],
        var_name="category",
        value_name="percent",
    )
    melted["category_label"] = melted["category"].map(binary_label)

    fig = px.line(
        melted,
        x="year",
        y="percent",
        color="category_label",
        markers=True,
        color_discrete_map=binary_colors,
        labels={
            "year": tr("year"),
            "percent": tr("percent"),
            "category_label": tr("category"),
        },
        title=None,
    )

    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        height=chart_height(),
        legend_title_text=tr("category"),
    )
    fig = add_logos_to_figure(fig)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ============================================================
# Filter selected solution
# ============================================================

df_filt = df[df["variable"] == selected_variable].copy()
df_bin_filt = df_binary[df_binary["variable"] == selected_variable].copy()

if selected_community != "Both":
    df_filt = df_filt[df_filt["community"] == selected_community]
    df_bin_filt = df_bin_filt[df_bin_filt["community"] == selected_community]


# ============================================================
# Main visual
# ============================================================

st.subheader(selected_solution_label)

if view_mode == tr("accepted_rejected"):
    st.info(tr("accepted_info"))

if selected_community == "Both":
    if mobile_mode:
        tab_gc, tab_tc = st.tabs([tr("gc"), tr("tc")])

        with tab_gc:
            if view_mode == tr("full_distribution"):
                render_distribution_line_chart(df_filt[df_filt["community"] == "GC"])
            else:
                render_binary_line_chart(df_bin_filt[df_bin_filt["community"] == "GC"])

        with tab_tc:
            if view_mode == tr("full_distribution"):
                render_distribution_line_chart(df_filt[df_filt["community"] == "TC"])
            else:
                render_binary_line_chart(df_bin_filt[df_bin_filt["community"] == "TC"])

    else:
        if view_mode == tr("full_distribution"):
            plot_data = prepare_distribution_data(df_filt)

            fig = px.line(
                plot_data,
                x="year",
                y="percent",
                color="response_category_label",
                facet_col="community_label",
                markers=True,
                color_discrete_map=response_colors,
                labels={
                    "year": tr("year"),
                    "percent": tr("percent"),
                    "response_category_label": tr("category"),
                    "community_label": tr("community"),
                },
                title=None,
            )

            fig.update_yaxes(range=[0, 100])
            fig.update_layout(
                height=520,
                legend_title_text=tr("category"),
            )
            fig = add_logos_to_figure(fig)

            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        else:
            plot_data = prepare_binary_data(df_bin_filt)

            melted = plot_data.melt(
                id_vars=["year", "community", "community_label", "variable", "solution_label"],
                value_vars=["accepted", "rejected"],
                var_name="category",
                value_name="percent",
            )
            melted["category_label"] = melted["category"].map(binary_label)

            fig = px.line(
                melted,
                x="year",
                y="percent",
                color="category_label",
                facet_col="community_label",
                markers=True,
                color_discrete_map=binary_colors,
                labels={
                    "year": tr("year"),
                    "percent": tr("percent"),
                    "category_label": tr("category"),
                    "community_label": tr("community"),
                },
                title=None,
            )

            fig.update_yaxes(range=[0, 100])
            fig.update_layout(
                height=520,
                legend_title_text=tr("category"),
            )
            fig = add_logos_to_figure(fig)

            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

else:
    if view_mode == tr("full_distribution"):
        render_distribution_line_chart(df_filt)
    else:
        render_binary_line_chart(df_bin_filt)


# ============================================================
# Cross-solution comparison
# ============================================================

st.divider()
st.subheader(tr("compare_solutions"))

df_compare = df_binary.copy()

if selected_community != "Both":
    df_compare = df_compare[df_compare["community"] == selected_community]

df_compare = prepare_binary_data(df_compare)

if selected_community == "Both" and mobile_mode:
    tab_gc2, tab_tc2 = st.tabs([tr("gc"), tr("tc")])

    for tab, community_code in [(tab_gc2, "GC"), (tab_tc2, "TC")]:
        with tab:
            sub = df_compare[df_compare["community"] == community_code]

            fig2 = px.line(
                sub,
                x="year",
                y="accepted",
                color="solution_label",
                markers=True,
                color_discrete_map=solution_colors,
                labels={
                    "year": tr("year"),
                    "accepted": tr("accepted"),
                    "solution_label": tr("solution"),
                },
                title=None,
            )

            fig2.update_yaxes(range=[0, 100])
            fig2.update_layout(
                height=430,
                legend_title_text=tr("solution"),
            )
            fig2 = add_logos_to_figure(fig2)

            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

else:
    fig2 = px.line(
        df_compare,
        x="year",
        y="accepted",
        color="solution_label",
        markers=True,
        facet_col="community_label" if selected_community == "Both" else None,
        color_discrete_map=solution_colors,
        labels={
            "year": tr("year"),
            "accepted": tr("accepted"),
            "solution_label": tr("solution"),
            "community_label": tr("community"),
        },
        title=None,
    )

    fig2.update_yaxes(range=[0, 100])
    fig2.update_layout(
        height=chart_height(),
        legend_title_text=tr("solution"),
    )
    fig2 = add_logos_to_figure(fig2)

    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)


# ============================================================
# Joint acceptance
# ============================================================

st.divider()
st.subheader(tr("maximum_possible_agreement"))
st.caption(tr("joint_acceptance_note"))

joint = (
    df_binary
    .pivot_table(
        index=["year", "variable"],
        columns="community",
        values="accepted",
        aggfunc="first",
    )
    .reset_index()
)

if {"GC", "TC"}.issubset(joint.columns):
    joint["joint_acceptance"] = joint[["GC", "TC"]].min(axis=1)
    joint["solution_label"] = joint["variable"].map(solution_label)

    fig3 = px.line(
        joint,
        x="year",
        y="joint_acceptance",
        color="solution_label",
        markers=True,
        color_discrete_map=solution_colors,
        labels={
            "year": tr("year"),
            "joint_acceptance": tr("joint_acceptance"),
            "solution_label": tr("solution"),
        },
        title=None,
    )

    fig3.update_yaxes(range=[0, 100])
    fig3.update_layout(
        height=chart_height(),
        legend_title_text=tr("solution"),
    )
    fig3 = add_logos_to_figure(fig3)

    st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)



# ============================================================
# AI module: Ask the data and theory
# ============================================================

st.divider()

AI_LABELS = {
    "English": {
        "title": "Ask the data and theory",
        "intro": "Ask a question about the dashboard data and, where relevant, the theoretical interpretation from the indexed papers and Conflict and Change.",
        "mode": "Answer mode",
        "data_only": "Data only",
        "book_only": "Papers + book only",
        "data_book": "Data + papers + book interpretation",
        "question": "Your question",
        "placeholder": "Example: Which solution has the highest joint acceptance in 2025, and how can this be interpreted theoretically?",
        "button": "Ask OpenAI",
        "missing": "OpenAI is not configured yet. Add OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID to Streamlit secrets.",
        "no_question": "Please enter a question.",
        "answer": "Answer",
        "with_book": "Theoretical grounding uses the indexed published-paper corpus and Conflict and Change.",
        "package_missing": "The openai package is not installed. Add openai to requirements.txt.",
    },
    "Greek": {
        "title": "Ρώτησε τα δεδομένα και τη θεωρία",
        "intro": "Κάντε ερώτηση για τα δεδομένα του πίνακα και, όπου είναι σχετικό, για τη θεωρητική ερμηνεία από το ευρετηριασμένο σώμα δημοσιευμένων εργασιών του Charis Psaltis.",
        "mode": "Τύπος απάντησης",
        "data_only": "Μόνο δεδομένα",
        "book_only": "Μόνο δημοσιευμένες εργασίες",
        "data_book": "Δεδομένα + ερμηνεία από δημοσιευμένες εργασίες",
        "question": "Η ερώτησή σας",
        "placeholder": "Παράδειγμα: Ποια λύση έχει την υψηλότερη κοινή αποδοχή το 2025 και πώς ερμηνεύεται θεωρητικά;",
        "button": "Ρώτησε το OpenAI",
        "missing": "Το OpenAI δεν έχει ρυθμιστεί ακόμη. Προσθέστε OPENAI_API_KEY και OPENAI_VECTOR_STORE_ID στα μυστικά του Streamlit.",
        "no_question": "Παρακαλώ γράψτε μια ερώτηση.",
        "answer": "Απάντηση",
        "with_book": "Η θεωρητική τεκμηρίωση χρησιμοποιεί το ευρετηριασμένο σώμα δημοσιευμένων εργασιών.",
        "package_missing": "Το πακέτο openai δεν είναι εγκατεστημένο. Προσθέστε openai στο requirements.txt.",
    },
    "Turkish": {
        "title": "Veriye ve teoriye sor",
        "intro": "Panel verileri hakkında ve ilgili olduğunda Charis Psaltis'in indekslenmiş yayımlanmış makaleler bütününe dayalı teorik yorum hakkında soru sorun.",
        "mode": "Yanıt türü",
        "data_only": "Sadece veri",
        "book_only": "Sadece yayımlanmış makaleler",
        "data_book": "Veri + yayımlanmış makaleler yorumu",
        "question": "Sorunuz",
        "placeholder": "Örnek: 2025 yılında hangi çözüm en yüksek ortak kabule sahiptir ve bu teorik olarak nasıl yorumlanabilir?",
        "button": "OpenAI'ya sor",
        "missing": "OpenAI henüz yapılandırılmadı. Streamlit secrets içine OPENAI_API_KEY ve OPENAI_VECTOR_STORE_ID ekleyin.",
        "no_question": "Lütfen bir soru girin.",
        "answer": "Yanıt",
        "with_book": "Teorik temellendirme indekslenmiş yayımlanmış makaleler bütününü kullanır.",
        "package_missing": "openai paketi kurulu değil. requirements.txt dosyasına openai ekleyin.",
    },
}

L = AI_LABELS.get(language, AI_LABELS["English"])

LOCAL_PROVIDER_LABELS = {
    "English": {
        "provider": "AI provider",
        "openai": "OpenAI (cloud)",
        "local": "Local AI (Ollama)",
        "local_note": "Local mode keeps questions, dashboard data, and retrieved passages on this computer.",
        "button": "Ask Local AI",
        "spinner": "The local model is analysing the available sources...",
        "model_used": "Local model used",
        "sources": "Retrieved paper and book sources",
        "source_note": "Source numbers below map exactly to the SOURCE labels in the answer. Page numbers refer to pages in the indexed PDF and may differ from the publication's printed pagination.",
        "combined_note": "Combined mode includes dashboard tables and is slower. For a question about scholarship without survey figures, choose ‘Papers + book only’.",
        "failed": "Local AI request failed:",
    },
    "Greek": {
        "provider": "Πάροχος ΤΝ",
        "openai": "OpenAI (νέφος)",
        "local": "Τοπική ΤΝ (Ollama)",
        "local_note": "Στην τοπική λειτουργία, οι ερωτήσεις, τα δεδομένα και τα ανακτημένα αποσπάσματα παραμένουν σε αυτόν τον υπολογιστή.",
        "button": "Ρώτησε την Τοπική ΤΝ",
        "spinner": "Το τοπικό μοντέλο αναλύει τις διαθέσιμες πηγές...",
        "model_used": "Τοπικό μοντέλο",
        "sources": "Ανακτημένες πηγές από άρθρα και βιβλίο",
        "source_note": "Οι αριθμοί πηγών αντιστοιχούν ακριβώς στις ενδείξεις SOURCE της απάντησης. Οι αριθμοί σελίδων αφορούν το ευρετηριασμένο PDF και μπορεί να διαφέρουν από την έντυπη σελιδαρίθμηση.",
        "combined_note": "Η συνδυασμένη λειτουργία περιλαμβάνει τους πίνακες δεδομένων και είναι πιο αργή. Για ερώτηση μόνο για τη βιβλιογραφία, επιλέξτε «Μόνο δημοσιευμένες εργασίες».",
        "failed": "Αποτυχία αιτήματος προς την Τοπική ΤΝ:",
    },
    "Turkish": {
        "provider": "Yapay zekâ sağlayıcısı",
        "openai": "OpenAI (bulut)",
        "local": "Yerel Yapay Zekâ (Ollama)",
        "local_note": "Yerel modda sorular, panel verileri ve getirilen bölümler bu bilgisayarda kalır.",
        "button": "Yerel Yapay Zekâya Sor",
        "spinner": "Yerel model mevcut kaynakları analiz ediyor...",
        "model_used": "Kullanılan yerel model",
        "sources": "Getirilen makale ve kitap kaynakları",
        "source_note": "Aşağıdaki kaynak numaraları yanıttaki SOURCE etiketleriyle tam olarak eşleşir. Sayfa numaraları indekslenen PDF'ye aittir ve basılı yayının sayfalarından farklı olabilir.",
        "combined_note": "Birleşik mod veri tablolarını da içerdiği için daha yavaştır. Anket rakamları gerektirmeyen akademik sorular için yalnızca makale ve kitap modunu seçin.",
        "failed": "Yerel Yapay Zekâ isteği başarısız oldu:",
    },
}
P = LOCAL_PROVIDER_LABELS.get(language, LOCAL_PROVIDER_LABELS["English"])


def get_streamlit_secret(name: str, default: str = "") -> str:
    """Return a Streamlit secret without raising when no local secrets file exists."""
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


st.subheader(L["title"])
st.caption(L["intro"])

answer_provider = st.radio(
    P["provider"],
    [P["openai"], P["local"]],
    horizontal=True,
)

if answer_provider == P["local"]:
    st.caption(P["local_note"])


def build_ai_data_context(user_question: str = "") -> str:
    """Build a compact, question-aware and numerically authoritative data context."""
    raw_cols = ["year", "community", "variable", "response_category", "percent"]
    raw_data = df[raw_cols].sort_values(
        ["variable", "community", "year", "response_category"]
    )

    binary_data = df_binary.sort_values(["variable", "community", "year"]).copy()
    binary_data["solution_name"] = binary_data["variable"].map(solution_label)

    joint_context = joint.copy()
    if {"GC", "TC"}.issubset(joint_context.columns):
        cols = ["year", "variable", "GC", "TC", "joint_acceptance"]
        joint_context = joint_context[cols].sort_values(["variable", "year"])
        joint_context["solution_name"] = joint_context["variable"].map(solution_label)

    requested_years = {
        int(value)
        for value in re.findall(r"\b(?:19|20)\d{2}\b", user_question)
    }
    if requested_years:
        raw_data = raw_data[raw_data["year"].isin(requested_years)]
        binary_data = binary_data[binary_data["year"].isin(requested_years)]
        if "year" in joint_context.columns:
            joint_context = joint_context[joint_context["year"].isin(requested_years)]

    question_lower = user_question.casefold()
    joint_terms = [
        "joint", "bicommunal", "common acceptance",
        "κοινή αποδοχή", "κοινη αποδοχη", "δικοινοτική",
        "ortak kabul", "iki toplum",
    ]
    distribution_terms = [
        "against", "tolerate", "in favor", "full distribution",
        "εναντίον", "ανέχομαι", "υπέρ",
        "karşı", "tolere", "lehine",
    ]
    binary_terms = [
        "accepted", "rejected", "acceptance", "rejection",
        "αποδοχή", "απόρριψη", "αποδεκ",
        "kabul", "redd",
    ]

    wants_joint = any(term in question_lower for term in joint_terms)
    wants_distribution = any(term in question_lower for term in distribution_terms)
    wants_binary = any(term in question_lower for term in binary_terms)

    if wants_joint and not joint_context.empty:
        maxima = []
        for year_value, group in joint_context.groupby("year"):
            leader = group.sort_values("joint_acceptance", ascending=False).iloc[0]
            maxima.append(
                f"{int(year_value)}: {leader['solution_name']} "
                f"({leader['variable']}); GC={leader['GC']:.1f}%; "
                f"TC={leader['TC']:.1f}%; "
                f"joint=min(GC, TC)={leader['joint_acceptance']:.1f}%"
            )
        maxima_text = "\n".join(maxima)
        return f"""
AUTHORITATIVE PRECOMPUTED MAXIMUM JOINT ACCEPTANCE:
{maxima_text}

JOINT ACCEPTANCE DATA:
{joint_context.to_csv(index=False)}

RULE: joint_acceptance is calculated in Python as min(GC, TC).
Copy the displayed values exactly; do not recalculate or substitute values from another table.
"""

    if wants_distribution:
        return f"""
ORIGINAL THREE-CATEGORY DATA:
{raw_data.to_csv(index=False)}

RULE: in_favor, tolerate, and against are separate response categories.
Copy percentages exactly from this table.
"""

    if wants_binary:
        return f"""
DERIVED ACCEPTED / REJECTED DATA:
{binary_data.to_csv(index=False)}

RULE: accepted = in_favor + tolerate; rejected = against.
Copy percentages exactly from this table.
"""

    return f"""
DERIVED ACCEPTED / REJECTED DATA:
{binary_data.to_csv(index=False)}

JOINT ACCEPTANCE DATA:
{joint_context.to_csv(index=False)}

RULES:
- accepted = in_favor + tolerate
- rejected = against
- joint_acceptance = min(GC, TC)
Copy percentages exactly from the supplied tables.
"""

answer_mode = st.radio(
    L["mode"],
    [L["data_book"], L["data_only"], L["book_only"]],
    horizontal=True,
)

if answer_provider == P["local"] and answer_mode == L["data_book"]:
    st.info(P["combined_note"])

evaluation_enabled = st.checkbox(
    "Benchmark evaluation mode",
    help=(
        "Use a fixed research question, capture model and timing information, "
        "rate the answer, and export the comparison as CSV."
    ),
)

selected_benchmark = {
    "id": "MANUAL",
    "domain": "Manual question",
    "recommended_mode": "",
    "question": "",
}

if evaluation_enabled:
    selected_index = st.selectbox(
        "Benchmark question",
        options=range(len(BENCHMARK_QUESTIONS)),
        format_func=lambda index: (
            f"{BENCHMARK_QUESTIONS[index]['id']} — "
            f"{BENCHMARK_QUESTIONS[index]['domain']}"
        ),
    )
    selected_benchmark = BENCHMARK_QUESTIONS[selected_index]
    st.caption(
        f"Recommended answer mode: {selected_benchmark['recommended_mode']}"
    )
    question = st.text_area(
        L["question"],
        value=selected_benchmark["question"],
        height=160,
        key=f"benchmark_question_{selected_benchmark['id']}",
    )
else:
    question = st.text_area(
        L["question"],
        placeholder=L["placeholder"],
        height=120,
        key="manual_ai_question",
    )

benchmark_for_run = {**selected_benchmark, "question": question}

ask_button_label = L["button"] if answer_provider == P["openai"] else P["button"]

if st.button(ask_button_label):
    if not question.strip():
        st.warning(L["no_question"])
    elif answer_provider == P["local"]:
        if answer_mode == L["data_only"]:
            local_mode = "data_only"
            local_data_context = build_ai_data_context(question)
        elif answer_mode == L["book_only"]:
            local_mode = "theory_only"
            local_data_context = ""
        else:
            local_mode = "combined"
            local_data_context = build_ai_data_context(question)

        try:
            from local_ai import ask_local

            with st.spinner(P["spinner"]):
                local_result = ask_local(
                    question=question,
                    mode=local_mode,
                    data_context=local_data_context,
                )

            st.markdown(f"### {L['answer']}")
            st.markdown(local_result["answer"])
            st.caption(f"{P['model_used']}: {local_result['model']}")
            performance = local_result.get("performance", {})
            if performance.get("total_seconds") is not None:
                st.caption(
                    f"Response time: {performance['total_seconds']:.1f} seconds · "
                    f"Prompt tokens: {performance.get('prompt_tokens', 0)} · "
                    f"Output tokens: {performance.get('output_tokens', 0)}"
                )

            if local_result["sources"]:
                with st.expander(P["sources"]):
                    st.caption(P["source_note"])
                    for source in local_result["sources"]:
                        source_number = source["source_number"]
                        citation = source["citation"]
                        st.markdown(f"**SOURCE {source_number} — {citation}**")
                        excerpt = source.get("excerpt", "").strip()
                        if excerpt:
                            quoted_excerpt = excerpt.replace("\n", "\n> ")
                            st.markdown(f"> {quoted_excerpt}")
                        st.caption(
                            f"Type: {source['document_type']} · "
                            f"Retrieval score: {source['score']:.3f}"
                        )

            if evaluation_enabled:
                set_evaluation_candidate(
                    benchmark=benchmark_for_run,
                    provider=P["local"],
                    answer_mode=answer_mode,
                    model=local_result["model"],
                    answer=local_result["answer"],
                    response_seconds=performance.get("total_seconds"),
                    prompt_tokens=performance.get("prompt_tokens"),
                    output_tokens=performance.get("output_tokens"),
                    sources=local_result["sources"],
                )
        except Exception as exc:
            st.error(f"{P['failed']} {exc}")
    elif ask_openai is None:
        st.error(L["package_missing"])
    elif not get_streamlit_secret("OPENAI_API_KEY"):
        st.error(L["missing"])
    else:
        if answer_mode == L["data_only"]:
            openai_mode = "data_only"
            data_context = build_ai_data_context(question)
        elif not get_streamlit_secret(OPENAI_VECTOR_STORE_SECRET):
            st.error(L["missing"])
            st.stop()
        elif answer_mode == L["book_only"]:
            openai_mode = "theory_only"
            data_context = "No dataset supplied in this mode."
        else:
            openai_mode = "combined"
            data_context = build_ai_data_context(question)

        with st.spinner("OpenAI is analysing the available data and indexed sources..."):
            try:
                openai_result = ask_openai(
                    question=question,
                    mode=openai_mode,
                    data_context=data_context,
                    api_key=get_streamlit_secret("OPENAI_API_KEY"),
                    vector_store_id=get_streamlit_secret(OPENAI_VECTOR_STORE_SECRET),
                    model=get_streamlit_secret("OPENAI_MODEL", "gpt-5-mini"),
                )

                st.markdown(f"### {L['answer']}")
                st.markdown(openai_result["answer"])
                performance = openai_result.get("performance", {})
                st.caption(
                    f"Model used: {openai_result['model']} · "
                    f"Response time: {performance.get('total_seconds', 0):.1f} seconds · "
                    f"Prompt tokens: {performance.get('prompt_tokens') or 0} · "
                    f"Output tokens: {performance.get('output_tokens') or 0}"
                )

                if openai_result["sources"]:
                    st.caption(L["with_book"])
                    with st.expander(P["sources"]):
                        st.caption(P["source_note"])
                        for source in openai_result["sources"]:
                            st.markdown(
                                f"**SOURCE {source['source_number']} — {source['citation']}**"
                            )
                            if source.get("excerpt"):
                                quoted = source["excerpt"].replace("\n", "\n> ")
                                st.markdown(f"> {quoted}")
                            score = source.get("score")
                            if isinstance(score, (int, float)):
                                st.caption(f"Retrieval score: {score:.3f}")

                if evaluation_enabled:
                    set_evaluation_candidate(
                        benchmark=benchmark_for_run,
                        provider=P["openai"],
                        answer_mode=answer_mode,
                        model=openai_result["model"],
                        answer=openai_result["answer"],
                        response_seconds=performance.get("total_seconds"),
                        prompt_tokens=performance.get("prompt_tokens"),
                        output_tokens=performance.get("output_tokens"),
                        sources=openai_result["sources"],
                    )

            except Exception as e:
                st.error("OpenAI request failed.")
                st.exception(e)

if evaluation_enabled:
    render_evaluation_panel()


# ============================================================
# Data table and downloads
# ============================================================

st.divider()

with st.expander(tr("data_table")):
    table = df.copy()
    table["community_display"] = table["community"].map(community_label)
    table["solution_display"] = table["variable"].map(solution_label)
    table["response_display"] = table["response_category"].map(response_label)

    st.dataframe(table, use_container_width=True)

    st.download_button(
        label=tr("download_filtered"),
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="cyprus_solution_landscape_data.csv",
        mime="text/csv",
    )


# ============================================================
# Footer
# ============================================================

st.caption(tr("footer"))

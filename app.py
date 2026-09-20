
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import itertools
import re
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UK Top 50 Playlist Intelligence",
    page_icon="🎵",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎵 United Kingdom Top 50 Playlist Intelligence")

st.markdown(
    """
    ### Market Structure • Artist Diversity • Collaboration •
    Content Strategy • Track Duration
    """
)

st.caption(
    "UK Spotify Top 50 playlist analysis dashboard"
)


# ============================================================
# FIND DATASET AUTOMATICALLY
# ============================================================

@st.cache_data
def find_dataset():

    base = Path(__file__).resolve().parent

    possible = [

        base / "data" / "Atlantic_United_Kingdom_clean.csv",

        base / "Atlantic_United_Kingdom_clean.csv",

        base / "Atlantic_United_Kingdom.csv",

    ]

    for path in possible:

        if path.exists():
            return path

    # Recursive search
    matches = list(
        base.rglob("Atlantic_United_Kingdom*.csv")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Atlantic_United_Kingdom.csv was not found. "
        "Place the CSV in the same folder as app.py "
        "or inside a data folder."
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    path = find_dataset()

    df = pd.read_csv(path)

    # Date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True
    )

    # Numeric columns
    for col in [
        "position",
        "popularity",
        "duration_ms",
        "total_tracks"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Text cleaning
    df["song"] = (
        df["song"]
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    df["artist"] = (
        df["artist"]
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    df["album_type"] = (
        df["album_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Remove duplicate date-position rows
    df = df.drop_duplicates(
        ["date", "position"],
        keep="first"
    ).copy()

    # Duration
    df["duration_min"] = (
        df["duration_ms"] / 60000
    )

    # Collaboration
    df["is_collaboration"] = (
        df["artist"]
        .str.contains(
            r"\s*&\s*",
            regex=True,
            na=False
        )
    )

    # Rank bucket
    df["rank_bucket"] = pd.cut(
        df["position"],
        bins=[
            0,
            10,
            20,
            30,
            40,
            50
        ],
        labels=[
            "1–10",
            "11–20",
            "21–30",
            "31–40",
            "41–50"
        ]
    )

    return df


# ============================================================
# ARTIST CREDIT PROCESSING
# ============================================================

PROTECTED_ARTISTS = [
    "Chase & Status",
    "Earth, Wind & Fire"
]


def split_artist(value):

    value = str(value)

    for i, name in enumerate(
        PROTECTED_ARTISTS
    ):

        value = value.replace(
            name,
            f"__PROTECTED_{i}__"
        )

    parts = re.split(
        r"\s*&\s*",
        value
    )

    parts = [
        x.strip()
        for x in parts
        if x.strip()
    ]

    restored = []

    for part in parts:

        for i, name in enumerate(
            PROTECTED_ARTISTS
        ):

            if part == (
                f"__PROTECTED_{i}__"
            ):

                part = name

        restored.append(part)

    return restored


@st.cache_data
def create_credits(df):

    temp = df[
        [
            "date",
            "position",
            "song",
            "artist",
            "popularity",
            "album_type",
            "is_explicit",
            "duration_min"
        ]
    ].copy()

    temp["artist_parts"] = (
        temp["artist"]
        .apply(split_artist)
    )

    credits = (
        temp
        .explode("artist_parts")
        .rename(
            columns={
                "artist_parts":
                "artist_credit"
            }
        )
    )

    credits["artist_credit"] = (
        credits["artist_credit"]
        .astype(str)
        .str.strip()
    )

    # Normalize case variations
    credits["_key"] = (
        credits["artist_credit"]
        .str.casefold()
    )

    display_map = (
        credits
        .groupby("_key")["artist_credit"]
        .agg(
            lambda x:
            x.value_counts().index[0]
        )
        .to_dict()
    )

    credits["artist_credit"] = (
        credits["_key"]
        .map(display_map)
    )

    credits.drop(
        columns="_key",
        inplace=True
    )

    return credits


# ============================================================
# LOAD
# ============================================================

try:

    df = load_data()

    credits = create_credits(df)

except Exception as e:

    st.error(
        f"Dataset loading error: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Dashboard Filters")


# ------------------------------------------------------------
# DATE
# ------------------------------------------------------------

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(
        min_date,
        max_date
    ),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 1:

    start_date = date_range[0]
    end_date = date_range[0]

else:

    start_date = date_range[0]
    end_date = date_range[1]


# ------------------------------------------------------------
# ARTIST
# ------------------------------------------------------------

artist_options = sorted(
    credits[
        "artist_credit"
    ]
    .dropna()
    .unique()
)

selected_artists = st.sidebar.multiselect(
    "Artist",
    artist_options
)


# ------------------------------------------------------------
# SOLO / COLLABORATION
# ------------------------------------------------------------

mode = st.sidebar.radio(
    "Content Type",
    [
        "All",
        "Solo",
        "Collaboration"
    ]
)


# ------------------------------------------------------------
# ALBUM TYPE
# ------------------------------------------------------------

album_options = sorted(
    df["album_type"]
    .dropna()
    .unique()
)

selected_album_types = st.sidebar.multiselect(
    "Album Type",
    album_options,
    default=album_options
)


# ============================================================
# FILTER DATA
# ============================================================

filtered = df[
    df["date"].between(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date)
    )
].copy()


# Artist filtering
if selected_artists:

    selected_keys = {
        x.casefold()
        for x in selected_artists
    }

    filtered = filtered[
        filtered["artist"].apply(
            lambda x:
            any(
                artist.casefold()
                in selected_keys
                for artist
                in split_artist(x)
            )
        )
    ].copy()


# Solo / collaboration
if mode == "Solo":

    filtered = filtered[
        ~filtered["is_collaboration"]
    ]

elif mode == "Collaboration":

    filtered = filtered[
        filtered["is_collaboration"]
    ]


# Album type
if selected_album_types:

    filtered = filtered[
        filtered["album_type"].isin(
            selected_album_types
        )
    ]


# ============================================================
# FILTER CREDITS
# ============================================================

keys = set(
    zip(
        filtered["date"],
        filtered["position"]
    )
)

filtered_credits = credits[
    credits.apply(
        lambda row:
        (
            row["date"],
            row["position"]
        ) in keys,
        axis=1
    )
].copy()


# ============================================================
# EMPTY RESULT
# ============================================================

if filtered.empty:

    st.warning(
        "No data matches the selected filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

artist_counts = (
    filtered_credits[
        "artist_credit"
    ]
    .value_counts()
)

top5_share = (
    artist_counts
    .head(5)
    .sum()
    /
    len(filtered_credits)
)


daily_unique = (
    filtered
    .groupby("date")["artist"]
    .nunique()
)

daily_entries = (
    filtered
    .groupby("date")["position"]
    .count()
)

daily_diversity = (
    daily_unique /
    daily_entries
).mean()


collab_rate = (
    filtered["is_collaboration"]
    .mean()
)

explicit_rate = (
    filtered["is_explicit"]
    .mean()
)

single_count = (
    filtered["album_type"]
    .eq("single")
    .sum()
)

album_count = (
    filtered["album_type"]
    .eq("album")
    .sum()
)

single_album_ratio = (
    single_count /
    album_count
    if album_count
    else 0
)


# Content Variety Index
type_probs = (
    filtered["album_type"]
    .value_counts(
        normalize=True
    )
)

if len(type_probs) > 1:

    variety_index = (
        -(
            type_probs *
            np.log(type_probs)
        ).sum()
        /
        np.log(
            len(type_probs)
        )
    )

else:

    variety_index = 0


# ============================================================
# HEADER KPIs
# ============================================================

st.subheader("📊 Market Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Entries",
    f"{len(filtered):,}"
)

c2.metric(
    "Artists",
    f"{filtered['artist'].nunique():,}"
)

c3.metric(
    "Top-5 Share",
    f"{top5_share:.1%}"
)

c4.metric(
    "Daily Diversity",
    f"{daily_diversity:.1%}"
)

c5.metric(
    "Collaboration",
    f"{collab_rate:.1%}"
)

c6.metric(
    "Explicit",
    f"{explicit_rate:.1%}"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Overview",
        "👑 Artist Diversity",
        "🤝 Collaboration",
        "🎧 Content Strategy",
        "⏱️ Duration",
        "📚 Methodology"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.header(
        "Playlist Performance Overview"
    )

    daily = (
        filtered
        .groupby("date")
        .agg(
            avg_popularity=(
                "popularity",
                "mean"
            ),
            explicit_share=(
                "is_explicit",
                "mean"
            ),
            collaboration_share=(
                "is_collaboration",
                "mean"
            )
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    # Popularity trend
    with col1:

        fig = px.line(
            daily,
            x="date",
            y="avg_popularity",
            title=
            "Average Spotify Popularity Over Time"
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Popularity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Content trend
    with col2:

        fig = px.line(
            daily,
            x="date",
            y=[
                "explicit_share",
                "collaboration_share"
            ],
            title=
            "Explicit & Collaboration Trends"
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Share",
            yaxis_tickformat=".0%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Additional KPIs

    a, b, c, d = st.columns(4)

    a.metric(
        "Average Popularity",
        f"{filtered['popularity'].mean():.1f}"
    )

    b.metric(
        "Average Duration",
        f"{filtered['duration_min'].mean():.2f} min"
    )

    c.metric(
        "Single / Album",
        f"{single_album_ratio:.2f}"
    )

    d.metric(
        "Content Variety Index",
        f"{variety_index:.3f}"
    )


    # Download
    st.download_button(
        "⬇️ Download Filtered Dataset",
        filtered.to_csv(
            index=False
        ).encode("utf-8"),
        "uk_top50_filtered.csv",
        "text/csv"
    )


# ============================================================
# TAB 2 — ARTIST DIVERSITY
# ============================================================

with tab2:

    st.header(
        "👑 Artist Dominance & Diversity"
    )

    dominance = (

        filtered_credits

        .groupby(
            "artist_credit"
        )

        .agg(

            appearances=(
                "song",
                "size"
            ),

            unique_songs=(
                "song",
                "nunique"
            ),

            average_rank=(
                "position",
                "mean"
            ),

            best_rank=(
                "position",
                "min"
            ),

            average_popularity=(
                "popularity",
                "mean"
            ),

            rank_weighted_score=(
                "position",
                lambda x:
                (51 - x).sum()
            )
        )

        .sort_values(
            "rank_weighted_score",
            ascending=False
        )
    )

    top_n = st.slider(
        "Number of artists",
        5,
        30,
        15
    )

    chart_data = (
        dominance
        .head(top_n)
        .sort_values(
            "rank_weighted_score"
        )
    )

    fig = px.bar(
        chart_data,
        x="rank_weighted_score",
        y=chart_data.index,
        orientation="h",
        title=
        "Rank-Weighted Artist Dominance",
        hover_data=[
            "appearances",
            "unique_songs",
            "average_rank",
            "best_rank",
            "average_popularity"
        ]
    )

    fig.update_layout(
        yaxis_title="Artist",
        xaxis_title=
        "Σ (51 − Chart Position)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Artist Leaderboard"
    )

    st.dataframe(
        dominance.head(top_n),
        use_container_width=True
    )


    # Daily diversity
    daily_artist = (
        filtered
        .groupby("date")["artist"]
        .nunique()
        .reset_index(
            name="unique_artists"
        )
    )

    fig = px.line(
        daily_artist,
        x="date",
        y="unique_artists",
        title=
        "Daily Billed Artist Diversity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 3 — COLLABORATION
# ============================================================

with tab3:

    st.header(
        "🤝 Collaboration Network"
    )

    st.write(
        f"Collaboration entries: "
        f"**{filtered['is_collaboration'].sum():,}** "
        f"({collab_rate:.1%})"
    )


    collaboration_rows = filtered_credits[
        filtered_credits[
            "artist"
        ].str.contains(
            r"\s*&\s*",
            regex=True,
            na=False
        )
    ]


    pairs = []

    for _, group in collaboration_rows.groupby(
        [
            "date",
            "position",
            "song"
        ]
    ):

        artists = list(
            dict.fromkeys(
                group[
                    "artist_credit"
                ]
                .dropna()
                .tolist()
            )
        )

        for a, b in itertools.combinations(
            sorted(artists),
            2
        ):

            pairs.append(
                (a, b)
            )


    if pairs:

        pair_df = (

            pd.DataFrame(
                pairs,
                columns=[
                    "Artist 1",
                    "Artist 2"
                ]
            )

            .value_counts()

            .reset_index(
                name="Collaborations"
            )

            .sort_values(
                "Collaborations",
                ascending=False
            )
        )


        st.subheader(
            "Top Collaboration Pairs"
        )

        st.dataframe(
            pair_df.head(20),
            use_container_width=True
        )


        # Network
        network_data = pair_df.head(35)

        G = nx.Graph()

        for _, row in network_data.iterrows():

            G.add_edge(
                row["Artist 1"],
                row["Artist 2"],
                weight=row[
                    "Collaborations"
                ]
            )


        pos = nx.spring_layout(
            G,
            seed=42,
            k=1.2
        )


        edge_x = []
        edge_y = []

        for u, v in G.edges():

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            edge_x.extend(
                [x0, x1, None]
            )

            edge_y.extend(
                [y0, y1, None]
            )


        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none",
            line=dict(
                width=1
            )
        )


        node_x = []
        node_y = []
        node_text = []
        node_size = []


        for node in G.nodes():

            x, y = pos[node]

            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

            node_size.append(
                12 +
                5 *
                G.degree(node)
            )


        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertemplate=
            "%{text}<extra></extra>",
            marker=dict(
                size=node_size
            )
        )


        fig = go.Figure(
            data=[
                edge_trace,
                node_trace
            ]
        )


        fig.update_layout(
            title=
            "UK Top 50 Collaboration Network",
            showlegend=False,
            xaxis=dict(
                visible=False
            ),
            yaxis=dict(
                visible=False
            ),
            height=650
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No collaboration relationships "
            "available under these filters."
        )


# ============================================================
# TAB 4 — CONTENT STRATEGY
# ============================================================

with tab4:

    st.header(
        "🎧 Content & Release Strategy"
    )

    col1, col2 = st.columns(2)


    # Release distribution
    with col1:

        release_counts = (
            filtered[
                "album_type"
            ]
            .value_counts()
            .reset_index()
        )

        release_counts.columns = [
            "album_type",
            "entries"
        ]

        fig = px.pie(
            release_counts,
            names="album_type",
            values="entries",
            title=
            "Release Format Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Release type popularity
    with col2:

        release_stats = (

            filtered

            .groupby(
                "album_type"
            )

            .agg(
                average_popularity=(
                    "popularity",
                    "mean"
                )
            )

            .reset_index()
        )

        fig = px.bar(
            release_stats,
            x="album_type",
            y="average_popularity",
            title=
            "Average Popularity by Release Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Rank band release type
    rank_release = (

        filtered

        .groupby(
            [
                "rank_bucket",
                "album_type"
            ],
            observed=True
        )

        .size()

        .reset_index(
            name="entries"
        )
    )


    fig = px.bar(
        rank_release,
        x="rank_bucket",
        y="entries",
        color="album_type",
        barmode="stack",
        title=
        "Release Type by Chart Position"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Explicit analysis
    explicit_rank = (

        filtered

        .groupby(
            "rank_bucket",
            observed=True
        )["is_explicit"]

        .mean()

        .reset_index(
            name="explicit_share"
        )
    )


    fig = px.bar(
        explicit_rank,
        x="rank_bucket",
        y="explicit_share",
        title=
        "Explicit Content Share by Rank Band",
        text_auto=".0%"
    )

    fig.update_layout(
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Release size
    st.subheader(
        "Release Size"
    )

    filtered["track_bucket"] = pd.cut(
        filtered["total_tracks"],
        bins=[
            0,
            1,
            5,
            10,
            15,
            20,
            30,
            50,
            np.inf
        ],
        labels=[
            "1",
            "2–5",
            "6–10",
            "11–15",
            "16–20",
            "21–30",
            "31–50",
            "51+"
        ]
    )

    track_size = (
        filtered
        .groupby(
            "track_bucket",
            observed=True
        )
        .size()
        .reset_index(
            name="entries"
        )
    )

    fig = px.bar(
        track_size,
        x="track_bucket",
        y="entries",
        title=
        "Playlist Entries by Release Size"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 5 — DURATION
# ============================================================

with tab5:

    st.header(
        "⏱️ Track Duration Analysis"
    )

    col1, col2 = st.columns(2)


    with col1:

        fig = px.histogram(
            filtered,
            x="duration_min",
            nbins=35,
            title=
            "Track Duration Distribution"
        )

        fig.update_layout(
            xaxis_title=
            "Duration (minutes)",
            yaxis_title=
            "Number of Tracks"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        sample = filtered.sample(
            min(
                3000,
                len(filtered)
            ),
            random_state=42
        )

        fig = px.scatter(
            sample,
            x="duration_min",
            y="popularity",
            color="album_type",
            hover_data=[
                "song",
                "artist",
                "position"
            ],
            title=
            "Duration vs Popularity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    duration_rank = (

        filtered

        .groupby(
            "rank_bucket",
            observed=True
        )["duration_min"]

        .mean()

        .reset_index()
    )


    fig = px.line(
        duration_rank,
        x="rank_bucket",
        y="duration_min",
        markers=True,
        title=
        "Average Duration by Rank Band"
    )

    fig.update_layout(
        yaxis_title=
        "Duration (minutes)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    duration_corr = (
        filtered[
            [
                "duration_min",
                "popularity"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )


    rank_corr = (
        filtered[
            [
                "position",
                "popularity"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )


    a, b = st.columns(2)

    a.metric(
        "Duration ↔ Popularity",
        f"{duration_corr:.3f}"
    )

    b.metric(
        "Rank ↔ Popularity",
        f"{rank_corr:.3f}"
    )


# ============================================================
# TAB 6 — METHODOLOGY
# ============================================================

with tab6:

    st.header(
        "📚 Insights & Methodology"
    )

    st.subheader(
        "Project Scope"
    )

    st.write(
        """
        This dashboard analyses the United Kingdom Top 50
        playlist using descriptive and diagnostic analytics.

        The main dimensions are:

        • Artist dominance
        • Artist diversity
        • Collaboration structure
        • Explicit vs clean content
        • Album / single strategy
        • Release size
        • Track duration
        • Chart rank
        • Spotify popularity
        """
    )


    st.subheader(
        "Artist Dominance"
    )

    st.write(
        """
        Artist dominance is measured using a rank-weighted
        score:

        Σ (51 − chart position)

        Higher-ranked chart entries therefore contribute
        more to the artist's dominance score.
        """
    )


    st.subheader(
        "Artist Diversity"
    )

    st.write(
        """
        Daily diversity is calculated as:

        Unique billed artists / number of chart entries

        The dashboard reports the average of this ratio
        across observed snapshots.
        """
    )


    st.subheader(
        "Content Variety Index"
    )

    st.write(
        """
        Content Variety Index uses normalized Shannon entropy:

        H = −Σ pᵢ ln(pᵢ) / ln(k)

        where pᵢ is the proportion of each release type.
        """
    )


    st.subheader(
        "Important Data Limitation"
    )

    st.warning(
        """
        The supplied dataset does NOT contain artist nationality,
        country of origin, or domestic/international classification.

        Therefore this dashboard does not fabricate a
        domestic-vs-international result.

        An external artist metadata mapping can be added later
        if localization analysis is required.
        """
    )


    st.subheader(
        "Why ML is not included"
    )

    st.info(
        """
        The project is fundamentally an EDA and market-structure
        analysis. The supplied dataset does not define a prediction
        target, so adding machine learning would not be analytically
        justified.

        A future extension could predict next-snapshot chart rank
        using historical rank and popularity features.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "UK Top 50 Playlist Intelligence • "
    "Data Analytics Project"
)

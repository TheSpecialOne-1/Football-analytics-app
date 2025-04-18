import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pandas import json_normalize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import random

# Use relative paths (✅ cloud-ready)
# --- PATHS ---
data_path = "data"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data")
events_path = os.path.join(data_path, "events")
matches_path = os.path.join(data_path, "matches")
competitions_path = os.path.join(data_path, "competitions.json")

# Set page config
st.set_page_config(page_title="Soccer Analysis App", layout="wide")

st.title("⚽ Soccer Analysis Toolkit")
st.markdown("""
<style>
    html, body, [class*="css"]  {
        background-color: #0c1a2b;
        color: #f2f2f2;
    }
    .stApp {
        background-color: #0c1a2b;
    }
    .stButton>button, .stSelectbox>div>div>div>div {
        background-color: #1f3b5c;
        color: #f2f2f2;
        border-radius: 5px;
        font-weight: bold;
    }
    .stDataFrame, .css-1d391kg, .css-1offfwp, .stTextInput>div>div>input {
        color: #ffffff !important;
        background-color: #1f3b5c !important;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# -- The rest of your app code continues here unchanged --
# -- All references to events_path, matches_path, competitions_path will now work on any OS or cloud --

# Sidebar menu
st.sidebar.title("⚙️ Select Analysis Module")
option = st.sidebar.radio(
    "Choose a module:",
    (
        "📂 1. Load Match Data",
        "🔥 2. Danger Pass Heatmap",
        "📊 3. Pass Comparison",
        "🔗 4. Possession Chain Viewer"
    )
)




# --- MODULE 1: Load Match Data ---
if option.endswith("1. Load Match Data"):
    st.header("📂 Match Loader")
    st.markdown("Explore available competitions, seasons, and match fixtures.")

    with open(competitions_path, encoding='utf-8') as f:
        comps = json.load(f)
    comp_df = pd.DataFrame(comps)

    st.subheader("Available Competitions")
    st.dataframe(comp_df[['competition_id', 'season_id', 'competition_name', 'season_name']])

    comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1)
    season_id = st.number_input("Enter Season ID:", min_value=0, step=1)

    match_file = os.path.join(matches_path, f"{comp_id}/{season_id}.json")
    if os.path.exists(match_file):
        with open(match_file, encoding='utf-8') as f:
            matches = json.load(f)
        match_df = pd.json_normalize(matches, sep="_")
        st.subheader("Match Fixtures")
        st.dataframe(match_df[['match_id', 'home_team_home_team_name', 'away_team_away_team_name', 'match_date']])
    else:
        st.warning("❌ Match file not found for this competition/season.")

# --- MODULE 2: Danger Pass Heatmap ---
elif option.endswith("2. Danger Pass Heatmap"):
    st.header("🔥 Danger Pass Heatmap")
    st.markdown("Visualize key passes that occur before a shot attempt.")

    comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1, key="comp2")
    season_id = st.number_input("Enter Season ID:", min_value=0, step=1, key="season2")
    team_required = st.text_input("Enter Team Name:")

    def load_matches(cid, sid):
        path = os.path.join(matches_path, f"{cid}/{sid}.json")
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def get_passes(match_id, team):
        path = os.path.join(events_path, f"{match_id}.json")
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        df = json_normalize(data, sep="_").assign(match_id=match_id)
        df = df[df['team_name'] == team]
        passes = df[df['type_name'] == 'Pass'].set_index('id')
        return df, passes

    if st.button("🎯 Generate Heatmap"):
        try:
            match_ids = [m['match_id'] for m in load_matches(comp_id, season_id)
                         if team_required in [m['home_team']['home_team_name'], m['away_team']['away_team_name']]]
            all_danger_passes = []

            for match_id in match_ids:
                df, passes = get_passes(match_id, team_required)
                shot_times = (df['minute'] * 60 + df['second']).tolist()
                shot_window = 15
                shot_start = [max(0, st - shot_window) for st in shot_times]
                pass_times = (passes['minute'] * 60 + passes['second']).tolist()
                pass_to_shot = [any(start < pt < st for start, st in zip(shot_start, shot_times)) for pt in pass_times]
                is_corner = passes['pass_type_name'] == 'Corner'
                danger_passes = passes[np.logical_and(pass_to_shot, ~is_corner)]
                all_danger_passes.append(danger_passes)

            all_passes = pd.concat(all_danger_passes, ignore_index=True)
            pitchLengthX, pitchWidthY = 120, 80
            x = all_passes['location'].apply(lambda loc: loc[0])
            y = pitchWidthY - all_passes['location'].apply(lambda loc: loc[1])
            H_Pass, _, _ = np.histogram2d(y, x, bins=5, range=[[0, pitchWidthY], [0, pitchLengthX]])

            fig, ax = plt.subplots(figsize=(10, 7))
            pos = ax.imshow(H_Pass / len(match_ids), extent=[0, 120, 0, 80], aspect='auto', cmap=plt.cm.Reds)
            fig.colorbar(pos, ax=ax)
            ax.set_title(f"Danger Pass Heatmap: {team_required}")
            ax.set_xlim((-1, 121))
            ax.set_ylim((83, -3))
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# --- MODULE 3: Pass Comparison ---
elif option.endswith("3. Pass Comparison"):
    st.header("📊 Team Pass Comparison")
    
    # Function for finding passes before shot
    def in_range(pass_time, start, finish):
        return True in ((start < pass_time) & (pass_time < finish)).unique()

    # Pitch size
    pitchLengthX = 120
    pitchWidthY = 80

    # Load competitions
    with open(competitions_path, encoding='utf-8') as f:
        competitions = json.load(f)
    
    # User inputs
    comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1, key="comp3")
    season_id = st.number_input("Enter Season ID:", min_value=0, step=1, key="season3")
    
    if st.button("📈 Compare Teams"):
        try:
            # Load matches
            match_file = os.path.join(matches_path, f"{comp_id}/{season_id}.json")
            with open(match_file, encoding='utf-8') as f:
                matches = json.load(f)

            # Initialize lists
            teams = []
            match_ids = []
            danger_passes_by = {}
            number_of_matches = {}
            passshot_data = []

            # Extract teams and match IDs
            for match in matches:
                home = match['home_team']['home_team_name']
                away = match['away_team']['away_team_name']
                
                if home not in teams:
                    teams.append(home)
                if away not in teams:
                    teams.append(away)
                    
                match_ids.append(match['match_id'])

            # Process each match
            for match in matches:
                match_id = match['match_id']
                file_path = os.path.join(events_path, f"{match_id}.json")
                
                if not os.path.exists(file_path):
                    st.warning(f"Missing file: {file_path}")
                    continue
                
                with open(file_path, encoding='utf-8') as data_file:
                    data = json.load(data_file)
                
                dfall = json_normalize(data, sep="_").assign(match_id=match_id)
                
                for theteam in [match['home_team']['home_team_name'], match['away_team']['away_team_name']]:
                    team_actions = dfall['team_name'] == theteam
                    df = dfall[team_actions]
                    
                    passes_match = df[df['type_name'] == 'Pass'].set_index('id')
                    shots_match = df[df['type_name'] == 'Shot'].set_index('id')

                    shot_times = shots_match['minute'] * 60 + shots_match['second']
                    shot_start = shot_times - 15  # shot_window = 15
                    pass_times = passes_match['minute'] * 60 + passes_match['second']
                    pass_to_shot = pass_times.apply(lambda x: in_range(x, shot_start, shot_times))
                    is_corner = passes_match['pass_type_name'] == 'Corner'
                    danger_passes = passes_match[np.logical_and(pass_to_shot, ~is_corner)]
                    
                    # Track danger passes
                    if theteam in danger_passes_by:
                        danger_passes_by[theteam] = pd.concat([danger_passes_by[theteam], danger_passes])
                        number_of_matches[theteam] += 1
                    else:
                        danger_passes_by[theteam] = danger_passes
                        number_of_matches[theteam] = 1
                    
                    passshot_data.append({
                        "Team": theteam,
                        "Passes": len(passes_match),
                        "Shots": len(shots_match),
                        "Goals": match['home_score'] if theteam == match['home_team']['home_team_name'] else match['away_score'],
                        "Danger Passes": len(danger_passes)
                    })

            # Create DataFrame
            passshot_df = pd.DataFrame(passshot_data)
            st.dataframe(passshot_df)

            # Plot Passes vs Shots
            fig, ax = plt.subplots(num=1)
            ax.plot('Passes', 'Shots', data=passshot_df, linestyle='none', markersize=4, marker='o', color='grey')

            # Highlight option
            highlight_team = st.selectbox("Highlight a team:", ["None"] + teams)
            if highlight_team != "None":
                team_of_interest_matches = (passshot_df['Team'] == highlight_team)
                ax.plot('Passes', 'Shots', data=passshot_df[team_of_interest_matches], 
                         linestyle='none', markersize=6, marker='o', color='red')

            ax.set_xticks(np.arange(0, 1000, step=100))
            ax.set_yticks(np.arange(0, 40, step=5))
            ax.set_xlabel('Passes (x)')
            ax.set_ylabel('Shots (y)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # OLS model
            passshot_df['Shots'] = pd.to_numeric(passshot_df['Shots'])
            passshot_df['Passes'] = pd.to_numeric(passshot_df['Passes'])
            model_fit = smf.ols(formula='Shots ~ Passes', data=passshot_df[['Shots', 'Passes']]).fit()
            
            # Regression line
            b = model_fit.params
            x = np.arange(0, 1000, step=0.5)
            y = b[0] + b[1] * x
            ax.plot(x, y, linestyle='-', color='black')
            ax.set_ylim(0, 40)
            ax.set_xlim(0, 800)
            
            st.pyplot(fig)

            # Poisson model
            passshot_df['Goals'] = pd.to_numeric(passshot_df['Goals'])
            poisson_model = smf.glm(formula="Goals ~ Passes + Team", 
                                   data=passshot_df, 
                                   family=sm.families.Poisson()).fit()
            
            # Show model summary
            st.subheader("Poisson Model Summary")
            st.text(str(poisson_model.summary()))

            # Create pass heatmaps
            x_all, y_all = [], []
            H_Pass = {}

            for team in teams:
                if team not in danger_passes_by:
                    continue

                dp = danger_passes_by[team]
                st.write(f"{team} has {len(dp)} danger passes")
                
                x = [p['location'][0] for _, p in dp.iterrows()]
                y = [pitchWidthY - p['location'][1] for _, p in dp.iterrows()]
                
                H_Pass[team] = np.histogram2d(y, x, bins=5, range=[[0, pitchWidthY], [0, pitchLengthX]])
                
                x_all += x
                y_all += y

            H_Pass_All = np.histogram2d(y_all, x_all, bins=5, range=[[0, pitchWidthY], [0, pitchLengthX]])

            # Team selection for heatmap
            heatmap_team = st.selectbox("Select team for heatmap visualization:", teams)
            
            if heatmap_team in H_Pass:
                # Create pitch
                fig, ax = plt.subplots(figsize=(10, 7))
                # Pitch Outline
                plt.plot([0, 0, pitchLengthX, pitchLengthX, 0], [0, pitchWidthY, pitchWidthY, 0, 0], color='black')
                plt.plot([pitchLengthX/2, pitchLengthX/2], [0, pitchWidthY], color='black')
                # Penalty areas
                plt.plot([16.5, 16.5], [pitchWidthY/2 +16.5, pitchWidthY/2-16.5], color='black')
                plt.plot([0, 16.5], [pitchWidthY/2 +16.5, pitchWidthY/2 +16.5], color='black')
                plt.plot([16.5, 0], [pitchWidthY/2 -16.5, pitchWidthY/2 -16.5], color='black')
                plt.plot([pitchLengthX, pitchLengthX-16.5], [pitchWidthY/2 +16.5, pitchWidthY/2 +16.5], color='black')
                plt.plot([pitchLengthX-16.5, pitchLengthX-16.5], [pitchWidthY/2 +16.5, pitchWidthY/2-16.5], color='black')
                plt.plot([pitchLengthX-16.5, pitchLengthX], [pitchWidthY/2 -16.5, pitchWidthY/2 -16.5], color='black')
                # Six-yard boxes
                plt.plot([0, 5.5], [pitchWidthY/2 +5.5, pitchWidthY/2 +5.5], color='black')
                plt.plot([5.5, 5.5], [pitchWidthY/2 +5.5, pitchWidthY/2-5.5], color='black')
                plt.plot([5.5, 0], [pitchWidthY/2 -5.5, pitchWidthY/2 -5.5], color='black')
                plt.plot([pitchLengthX, pitchLengthX-5.5], [pitchWidthY/2 +5.5, pitchWidthY/2 +5.5], color='black')
                plt.plot([pitchLengthX-5.5, pitchLengthX-5.5], [pitchWidthY/2 +5.5, pitchWidthY/2-5.5], color='black')
                plt.plot([pitchLengthX-5.5, pitchLengthX], [pitchWidthY/2 -5.5, pitchWidthY/2 -5.5], color='black')
                # Center circle
                centreCircle = plt.Circle((pitchLengthX/2, pitchWidthY/2), 9.15, color='black', fill=False)
                centreSpot = plt.Circle((pitchLengthX/2, pitchWidthY/2), 0.8, color='black')
                ax.add_patch(centreCircle)
                ax.add_patch(centreSpot)
                
                # Plot heatmap
                pos = ax.imshow(
                    H_Pass[heatmap_team][0] / number_of_matches[heatmap_team] - H_Pass_All[0] / (len(matches) * 2),
                    extent=[0, 120, 0, 80],
                    aspect='auto',
                    cmap=plt.cm.seismic,
                    vmin=-3,
                    vmax=3
                )

                ax.set_title(f'Number of passes per match by {heatmap_team}')
                plt.xlim((-1, 121))
                plt.ylim((83, -3))
                plt.tight_layout()
                plt.gca().set_aspect('equal', adjustable='box')
                fig.colorbar(pos, ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# --- MODULE 4: Possession Chain Viewer ---
elif option.endswith("4. Possession Chain Viewer"):
    st.header("🔗 Possession Chain Viewer")

    match_id = st.text_input("Enter Match ID:")
    home_team = st.text_input("Enter Home Team Name:")
    away_team = st.text_input("Enter Away Team Name:")
    selected_team = st.text_input("Enter team to show possession chains for:")

    if match_id and home_team and away_team and selected_team:
        file_path = os.path.join(events_path, f"{match_id}.json")
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as data_file:
                data = json.load(data_file)
            
            match_events = json_normalize(data, sep="_")
            
            # List all unique event types used
            all_event_types = match_events['type_name'].dropna().unique()
            all_event_types = sorted(all_event_types)
            
            # Event type selection
            st.subheader("Select Event Types to Include")
            selected_event_types = st.multiselect("Choose event types:", all_event_types)
            
            if selected_event_types:
                # Filter possessions for the selected team
                team_possessions = match_events[
                    (match_events['possession_team_name'] == selected_team)
                ]['possession'].unique()
                
                # Create plotly figure
                fig = go.Figure()
                fig.update_layout(
                    width=900 * 1.388,
                    height=900,
                    autosize=False,
                    plot_bgcolor="#0c1a2b",
                    paper_bgcolor="#0c1a2b",
                )
                fig.update_xaxes(range=[-0.03, 1.03], visible=False)
                fig.update_yaxes(range=[-0.03, 1.03], visible=False)
                
                # Draw pitch
                fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(color="white"))
                fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="white"))
                fig.add_shape(type="circle", x0=0.5-0.0915*0.72, y0=0.5-0.0915, 
                             x1=0.5+0.0915*0.72, y1=0.5+0.0915, line=dict(color="white"))
                fig.add_shape(type="rect", x0=0, y0=0.211, x1=0.170, y1=0.789, line=dict(color="white"))
                fig.add_shape(type="rect", x0=0, y0=0.368, x1=0.058, y1=0.632, line=dict(color="white"))
                fig.add_shape(type="rect", x0=-0.01, y0=0.447, x1=0.0, y1=0.553, line=dict(color="white"))
                fig.add_shape(type="rect", x0=0.83, y0=0.211, x1=1.0, y1=0.789, line=dict(color="white"))
                fig.add_shape(type="rect", x0=0.942, y0=0.368, x1=1.0, y1=0.632, line=dict(color="white"))
                fig.add_shape(type="rect", x0=1.0, y0=0.447, x1=1.01, y1=0.553, line=dict(color="white"))
                
                # Add possession chains
                for possession in team_possessions:
                    df = match_events[match_events['possession'] == possession]
                    df = df[df['type_name'].isin(selected_event_types)]
                    
                    chain_x, chain_y, text, colors, outcomes, times = [], [], [], [], [], []
                    for _, row in df.iterrows():
                        loc = row['location']
                        if isinstance(loc, list) and len(loc) == 2:
                            x = round((loc[0] * (100 / 120)) / 100, 3)
                            y = round(((80 - loc[1]) * (100 / 80)) / 100, 3)
                            
                            if x in chain_x and y in chain_y:
                                x += random.choice([-0.001, 0.001])
                                y += random.choice([-0.001, 0.001])
                            
                            chain_x.append(x)
                            chain_y.append(y)
                            text.append(row['type_name'])
                            times.append(f"{row['minute']}:{row['second']}")
                            outcomes.append('circle')
                            colors.append("#1F77B4")
                    
                    if chain_x:
                        fig.add_trace(go.Scatter(
                            x=chain_x,
                            y=chain_y,
                            mode='markers+lines+text',
                            text=text,
                            textposition='top center',
                            hovertemplate="Time: %{hovertext}<br>Event: %{text}",
                            hovertext=times,
                            marker=dict(color=colors, size=8, symbol=outcomes),
                            line=dict(color="#7F7F7F"),
                            name=f"{selected_team}"
                        ))
                
                fig.update_layout(title=f"Possession Chains for {selected_team}", title_font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one event type")
        else:
            st.warning(f"Match file not found: {file_path}")

# --- Footer ---
st.markdown("---")

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

# Add custom CSS
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0c1a2b;
    color: #f2f2f2;
}
.stApp {
    background-color: #0c1a2b;
}
.stButton > button, .stSelectbox > div > div > div > div {
    background-color: #1f3b5c;
    color: #f2f2f2;
    border-radius: 5px;
    font-weight: bold;
}
.stDataFrame, .css-1d391kg, .css-1offfwp, .stTextInput > div > div > input {
    color: #ffffff !important;
    background-color: #1f3b5c !important;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# Helper function to safely load JSON
def safe_load_json(file_path):
    """Safely load JSON file with error handling"""
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

# Helper function to safely normalize JSON data
def safe_json_normalize(data, sep="_"):
    """Safely normalize JSON data with error handling"""
    try:
        if data is None:
            return pd.DataFrame()
        return json_normalize(data, sep=sep)
    except Exception as e:
        st.error(f"Error normalizing data: {str(e)}")
        return pd.DataFrame()

# Sidebar menu
st.sidebar.title("⚙️ Select Analysis Module")
option = st.sidebar.radio(
    "Choose a module:",
    [
        "📂 1. Load Match Data",
        "🔥 2. Danger Pass Heatmap", 
        "📊 3. Pass Comparison",
        "🔗 4. Possession Chain Viewer"
    ]
)

# --- MODULE 1: Load Match Data ---
if option.endswith("1. Load Match Data"):
    st.header("📂 Match Loader")
    st.markdown("Explore available competitions, seasons, and match fixtures.")
    
    # Load competitions safely
    comps = safe_load_json(competitions_path)
    if comps is not None:
        comp_df = pd.DataFrame(comps)
        st.subheader("Available Competitions")
        
        # Check if required columns exist
        required_cols = ['competition_id', 'season_id', 'competition_name', 'season_name']
        existing_cols = [col for col in required_cols if col in comp_df.columns]
        if existing_cols:
            st.dataframe(comp_df[existing_cols])
        else:
            st.dataframe(comp_df)
        
        comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1)
        season_id = st.number_input("Enter Season ID:", min_value=0, step=1)
        
        match_file = os.path.join(matches_path, f"{comp_id}", f"{season_id}.json")
        if os.path.exists(match_file):
            matches = safe_load_json(match_file)
            if matches is not None:
                match_df = safe_json_normalize(matches, sep="_")
                if not match_df.empty:
                    st.subheader("Match Fixtures")
                    # Display available columns
                    available_cols = [col for col in match_df.columns if 'match_id' in col or 'team' in col or 'date' in col]
                    if available_cols:
                        st.dataframe(match_df[available_cols[:5]])  # Show first 5 relevant columns
                    else:
                        st.dataframe(match_df.head())
                else:
                    st.warning("❌ No match data found or data format error.")
            else:
                st.warning("❌ Error loading match file.")
        else:
            st.warning("❌ Match file not found for this competition/season.")
    else:
        st.error("❌ Could not load competitions file. Please check if the data directory exists.")

# --- MODULE 2: Danger Pass Heatmap ---
elif option.endswith("2. Danger Pass Heatmap"):
    st.header("🔥 Danger Pass Heatmap")
    st.markdown("Visualize key passes that occur before a shot attempt.")
    
    comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1, key="comp2")
    season_id = st.number_input("Enter Season ID:", min_value=0, step=1, key="season2")
    team_required = st.text_input("Enter Team Name:")
    
    def load_matches(cid, sid):
        """Load matches for given competition and season"""
        path = os.path.join(matches_path, f"{cid}", f"{sid}.json")
        return safe_load_json(path)
    
    def get_passes(match_id, team):
        """Get passes for a specific match and team"""
        path = os.path.join(events_path, f"{match_id}.json")
        data = safe_load_json(path)
        if data is None:
            return pd.DataFrame(), pd.DataFrame()
        
        df = safe_json_normalize(data, sep="_")
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        df = df.assign(match_id=match_id)
        
        # Check if team_name column exists
        if 'team_name' in df.columns:
            df = df[df['team_name'] == team]
        else:
            # Try alternative column names
            team_cols = [col for col in df.columns if 'team' in col.lower() and 'name' in col.lower()]
            if team_cols:
                df = df[df[team_cols[0]] == team]
        
        # Get passes
        if 'type_name' in df.columns:
            passes = df[df['type_name'] == 'Pass']
            if 'id' in passes.columns:
                passes = passes.set_index('id')
        else:
            passes = pd.DataFrame()
        
        return df, passes
    
    if st.button("🎯 Generate Heatmap"):
        if not team_required:
            st.warning("Please enter a team name.")
        else:
            try:
                matches = load_matches(comp_id, season_id)
                if matches is None:
                    st.error("Could not load matches for this competition/season.")
                else:
                    # Find match IDs for the team
                    match_ids = []
                    for m in matches:
                        if isinstance(m, dict):
                            # Check various possible team name fields
                            home_team = ""
                            away_team = ""
                            
                            if 'home_team' in m:
                                if isinstance(m['home_team'], dict) and 'home_team_name' in m['home_team']:
                                    home_team = m['home_team']['home_team_name']
                                elif isinstance(m['home_team'], str):
                                    home_team = m['home_team']
                            
                            if 'away_team' in m:
                                if isinstance(m['away_team'], dict) and 'away_team_name' in m['away_team']:
                                    away_team = m['away_team']['away_team_name']
                                elif isinstance(m['away_team'], str):
                                    away_team = m['away_team']
                            
                            if team_required in [home_team, away_team] and 'match_id' in m:
                                match_ids.append(m['match_id'])
                    
                    if not match_ids:
                        st.warning(f"No matches found for team: {team_required}")
                    else:
                        all_danger_passes = []
                        
                        for match_id in match_ids:
                            df, passes = get_passes(match_id, team_required)
                            
                            if df.empty or passes.empty:
                                continue
                            
                            # Get shot times (if any shots exist)
                            if 'type_name' in df.columns:
                                shots = df[df['type_name'] == 'Shot']
                                if not shots.empty and 'minute' in shots.columns and 'second' in shots.columns:
                                    shot_times = (shots['minute'] * 60 + shots['second']).tolist()
                                else:
                                    shot_times = []
                            else:
                                shot_times = []
                            
                            if shot_times and not passes.empty:
                                shot_window = 15
                                shot_start = [max(0, st - shot_window) for st in shot_times]
                                
                                if 'minute' in passes.columns and 'second' in passes.columns:
                                    pass_times = (passes['minute'] * 60 + passes['second']).tolist()
                                    pass_to_shot = [any(start < pt < st for start, st in zip(shot_start, shot_times)) for pt in pass_times]
                                    
                                    # Check for corner passes
                                    if 'pass_type_name' in passes.columns:
                                        is_corner = passes['pass_type_name'] == 'Corner'
                                        is_corner = is_corner.fillna(False)
                                    else:
                                        is_corner = pd.Series([False] * len(passes), index=passes.index)
                                    
                                    danger_passes = passes[np.logical_and(pass_to_shot, ~is_corner)]
                                    all_danger_passes.append(danger_passes)
                        
                        if all_danger_passes:
                            all_passes = pd.concat(all_danger_passes, ignore_index=True)
                            
                            if not all_passes.empty and 'location' in all_passes.columns:
                                pitchLengthX, pitchWidthY = 120, 80
                                
                                # Extract coordinates safely
                                valid_locations = all_passes['location'].dropna()
                                x_coords = []
                                y_coords = []
                                
                                for loc in valid_locations:
                                    if isinstance(loc, list) and len(loc) >= 2:
                                        x_coords.append(loc[0])
                                        y_coords.append(pitchWidthY - loc[1])
                                
                                if x_coords and y_coords:
                                    H_Pass, _, _ = np.histogram2d(y_coords, x_coords, bins=5, 
                                                                range=[[0, pitchWidthY], [0, pitchLengthX]])
                                    
                                    fig, ax = plt.subplots(figsize=(10, 7))
                                    pos = ax.imshow(H_Pass / len(match_ids), extent=[0, 120, 0, 80], 
                                                  aspect='auto', cmap=plt.cm.Reds)
                                    fig.colorbar(pos, ax=ax)
                                    ax.set_title(f"Danger Pass Heatmap: {team_required}")
                                    ax.set_xlim((-1, 121))
                                    ax.set_ylim((83, -3))
                                    st.pyplot(fig)
                                else:
                                    st.warning("No valid location data found for passes.")
                            else:
                                st.warning("No location data available in passes.")
                        else:
                            st.warning("No danger passes found for the selected team.")
                            
            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")

# --- MODULE 3: Pass Comparison ---
elif option.endswith("3. Pass Comparison"):
    st.header("📊 Team Pass Comparison")
    
    comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1, key="comp3")
    season_id = st.number_input("Enter Season ID:", min_value=0, step=1, key="season3")
    
    if st.button("📈 Compare Teams"):
        try:
            # Load matches
            match_file = os.path.join(matches_path, f"{comp_id}", f"{season_id}.json")
            matches = safe_load_json(match_file)
            
            if matches is None:
                st.error("Could not load matches for this competition/season.")
            else:
                # Initialize data structures
                teams = []
                passshot_data = []
                
                # Process each match
                for match in matches:
                    if not isinstance(match, dict) or 'match_id' not in match:
                        continue
                    
                    match_id = match['match_id']
                    file_path = os.path.join(events_path, f"{match_id}.json")
                    
                    data = safe_load_json(file_path)
                    if data is None:
                        continue
                    
                    dfall = safe_json_normalize(data, sep="_")
                    if dfall.empty:
                        continue
                    
                    dfall = dfall.assign(match_id=match_id)
                    
                    # Extract team names
                    home_team = ""
                    away_team = ""
                    
                    if 'home_team' in match:
                        if isinstance(match['home_team'], dict) and 'home_team_name' in match['home_team']:
                            home_team = match['home_team']['home_team_name']
                        elif isinstance(match['home_team'], str):
                            home_team = match['home_team']
                    
                    if 'away_team' in match:
                        if isinstance(match['away_team'], dict) and 'away_team_name' in match['away_team']:
                            away_team = match['away_team']['away_team_name']
                        elif isinstance(match['away_team'], str):
                            away_team = match['away_team']
                    
                    if home_team not in teams and home_team:
                        teams.append(home_team)
                    if away_team not in teams and away_team:
                        teams.append(away_team)
                    
                    # Process both teams
                    for theteam in [home_team, away_team]:
                        if not theteam:
                            continue
                        
                        # Filter team data
                        if 'team_name' in dfall.columns:
                            team_actions = dfall['team_name'] == theteam
                        else:
                            continue
                        
                        df = dfall[team_actions]
                        
                        if 'type_name' in df.columns:
                            passes_match = df[df['type_name'] == 'Pass']
                            shots_match = df[df['type_name'] == 'Shot']
                        else:
                            passes_match = pd.DataFrame()
                            shots_match = pd.DataFrame()
                        
                        # Get scores
                        home_score = match.get('home_score', 0) if isinstance(match.get('home_score'), int) else 0
                        away_score = match.get('away_score', 0) if isinstance(match.get('away_score'), int) else 0
                        
                        team_score = home_score if theteam == home_team else away_score
                        
                        passshot_data.append({
                            "Team": theteam,
                            "Passes": len(passes_match),
                            "Shots": len(shots_match),
                            "Goals": team_score
                        })
                
                if passshot_data:
                    # Create DataFrame
                    passshot_df = pd.DataFrame(passshot_data)
                    st.dataframe(passshot_df)
                    
                    # Create visualization if we have data
                    if len(passshot_df) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(passshot_df['Passes'], passshot_df['Shots'], alpha=0.7)
                        ax.set_xlabel('Passes')
                        ax.set_ylabel('Shots')
                        ax.set_title('Passes vs Shots')
                        
                        # Add trend line if possible
                        if len(passshot_df) > 2:
                            try:
                                z = np.polyfit(passshot_df['Passes'], passshot_df['Shots'], 1)
                                p = np.poly1d(z)
                                ax.plot(passshot_df['Passes'], p(passshot_df['Passes']), "r--", alpha=0.8)
                            except:
                                pass
                        
                        st.pyplot(fig)
                    
                else:
                    st.warning("No valid data found for comparison.")
                    
        except Exception as e:
            st.error(f"Error in pass comparison: {str(e)}")

# --- MODULE 4: Possession Chain Viewer ---
elif option.endswith("4. Possession Chain Viewer"):
    st.header("🔗 Possession Chain Viewer")
    
    match_id = st.text_input("Enter Match ID:")
    selected_team = st.text_input("Enter team to show possession chains for:")
    
    if match_id and selected_team:
        file_path = os.path.join(events_path, f"{match_id}.json")
        data = safe_load_json(file_path)
        
        if data is not None:
            match_events = safe_json_normalize(data, sep="_")
            
            if not match_events.empty:
                # List all unique event types used
                if 'type_name' in match_events.columns:
                    all_event_types = match_events['type_name'].dropna().unique()
                    all_event_types = sorted(all_event_types)
                    
                    # Event type selection
                    st.subheader("Select Event Types to Include")
                    selected_event_types = st.multiselect("Choose event types:", all_event_types)
                    
                    if selected_event_types:
                        # Filter possessions for the selected team
                        if 'possession_team_name' in match_events.columns:
                            team_possessions = match_events[
                                match_events['possession_team_name'] == selected_team
                            ]['possession'].unique() if 'possession' in match_events.columns else []
                        else:
                            team_possessions = []
                        
                        if len(team_possessions) > 0:
                            # Create plotly figure
                            fig = go.Figure()
                            fig.update_layout(
                                width=900,
                                height=600,
                                autosize=False,
                                plot_bgcolor="#0c1a2b",
                                paper_bgcolor="#0c1a2b",
                            )
                            
                            fig.update_xaxes(range=[-0.03, 1.03], visible=False)
                            fig.update_yaxes(range=[-0.03, 1.03], visible=False)
                            
                            # Draw pitch outline
                            fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(color="white"))
                            fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="white"))
                            
                            # Add possession chains
                            for possession in team_possessions[:5]:  # Limit to first 5 possessions
                                df = match_events[match_events['possession'] == possession]
                                df = df[df['type_name'].isin(selected_event_types)]
                                
                                chain_x, chain_y, text, times = [], [], [], []
                                
                                for _, row in df.iterrows():
                                    loc = row.get('location')
                                    if isinstance(loc, list) and len(loc) >= 2:
                                        x = round((loc[0] * (100 / 120)) / 100, 3)
                                        y = round(((80 - loc[1]) * (100 / 80)) / 100, 3)
                                        
                                        chain_x.append(x)
                                        chain_y.append(y)
                                        text.append(row.get('type_name', 'Unknown'))
                                        times.append(f"{row.get('minute', 0)}:{row.get('second', 0)}")
                                
                                if chain_x:
                                    fig.add_trace(go.Scatter(
                                        x=chain_x,
                                        y=chain_y,
                                        mode='markers+lines',
                                        text=text,
                                        hovertemplate="Time: %{hovertext}<br>Event: %{text}",
                                        hovertext=times,
                                        marker=dict(size=8),
                                        name=f"Possession {possession}"
                                    ))
                            
                            fig.update_layout(
                                title=f"Possession Chains for {selected_team}",
                                title_font=dict(color='white')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No possessions found for team: {selected_team}")
                    else:
                        st.info("Please select at least one event type")
                else:
                    st.warning("No event type information found in match data.")
            else:
                st.warning("No events data available for this match.")
        else:
            st.error(f"Match file not found: {file_path}")

# --- Footer ---
st.markdown("---")
st.markdown("⚽ **Soccer Analysis Toolkit** - Built with Streamlit")

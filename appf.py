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

# Use relative paths (cloud-ready)
# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data")
events_path = os.path.join(data_path, "events")
lineups_path = os.path.join(data_path, "lineups") 
matches_path = os.path.join(data_path, "matches")
competitions_path = os.path.join(data_path, "competitions.json")

def safe_load_json(file_path, encoding='utf-8'):
    """Safely load JSON file with multiple encoding attempts and error handling"""
    try:
        if not os.path.exists(file_path):
            return None
        
        # Try different encodings
        encodings = [encoding, 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                    # Clean content - remove any potential problematic characters
                    content = content.replace('\ufeff', '')  # Remove BOM
                    content = content.strip()
                    if content:
                        return json.loads(content)
                    return None
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        
        # If all encodings fail, try to read as bytes and clean
        with open(file_path, 'rb') as f:
            content = f.read()
            # Try to decode with error handling
            content = content.decode('utf-8', errors='ignore')
            content = content.replace('\ufeff', '').strip()
            if content:
                return json.loads(content)
        
        return None
        
    except Exception as e:
        st.error(f"Error loading {os.path.basename(file_path)}: {str(e)}")
        return None

def safe_json_normalize(data, sep="_"):
    """Safely normalize JSON data with error handling"""
    try:
        if data is None or len(data) == 0:
            return pd.DataFrame()
        return json_normalize(data, sep=sep)
    except Exception as e:
        st.error(f"Error normalizing data: {str(e)}")
        return pd.DataFrame()

def get_team_matches(comp_id, season_id, team_name):
    """Get match IDs for a specific team"""
    match_file = os.path.join(matches_path, str(comp_id), f"{season_id}.json")
    matches_data = safe_load_json(match_file)
    
    if matches_data is None:
        return []
    
    match_ids = []
    for match in matches_data:
        if isinstance(match, dict) and 'match_id' in match:
            home_team = ""
            away_team = ""
            
            # Extract home team name
            if 'home_team' in match:
                if isinstance(match['home_team'], dict):
                    home_team = match['home_team'].get('home_team_name', '')
                elif isinstance(match['home_team'], str):
                    home_team = match['home_team']
            
            # Extract away team name  
            if 'away_team' in match:
                if isinstance(match['away_team'], dict):
                    away_team = match['away_team'].get('away_team_name', '')
                elif isinstance(match['away_team'], str):
                    away_team = match['away_team']
            
            if team_name.lower() in [home_team.lower(), away_team.lower()]:
                match_ids.append(match['match_id'])
                
    return match_ids

def get_available_teams(comp_id, season_id):
    """Get list of available teams for the competition/season"""
    match_file = os.path.join(matches_path, str(comp_id), f"{season_id}.json")
    matches_data = safe_load_json(match_file)
    
    if matches_data is None:
        return []
    
    teams = set()
    for match in matches_data:
        if isinstance(match, dict):
            # Extract home team name
            if 'home_team' in match:
                if isinstance(match['home_team'], dict):
                    home_team = match['home_team'].get('home_team_name', '')
                elif isinstance(match['home_team'], str):
                    home_team = match['home_team']
                if home_team:
                    teams.add(home_team)
            
            # Extract away team name  
            if 'away_team' in match:
                if isinstance(match['away_team'], dict):
                    away_team = match['away_team'].get('away_team_name', '')
                elif isinstance(match['away_team'], str):
                    away_team = match['away_team']
                if away_team:
                    teams.add(away_team)
                    
    return sorted(list(teams))

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
        
        # Display competitions in a nice format
        display_cols = ['competition_id', 'season_id', 'competition_name', 'season_name', 'country_name']
        available_cols = [col for col in display_cols if col in comp_df.columns]
        
        if available_cols:
            st.dataframe(comp_df[available_cols].head(10))
        else:
            st.dataframe(comp_df.head())
        
        # User inputs - no default values
        comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1, value=0)
        season_id = st.number_input("Enter Season ID:", min_value=0, step=1, value=0)
        
        if st.button("🔍 Load Matches"):
            if comp_id == 0 or season_id == 0:
                st.warning("Please enter valid Competition ID and Season ID.")
            else:
                match_file = os.path.join(matches_path, str(comp_id), f"{season_id}.json")
                if os.path.exists(match_file):
                    matches = safe_load_json(match_file)
                    if matches is not None:
                        st.success(f"✅ Found {len(matches)} matches")
                        
                        # Show sample matches
                        if matches:
                            sample_match = matches[0]
                            st.subheader("Sample Match Data:")
                            st.json(sample_match)
                            
                            # Show available teams
                            teams = get_available_teams(comp_id, season_id)
                            if teams:
                                st.subheader("Available Teams:")
                                st.write(", ".join(teams[:10]))  # Show first 10 teams
                            
                    else:
                        st.error("❌ Error loading match file.")
                else:
                    st.warning("❌ Match file not found for this competition/season.")
    else:
        st.error("❌ Could not load competitions file.")

# --- MODULE 2: Danger Pass Heatmap ---
elif option.endswith("2. Danger Pass Heatmap"):
    st.header("🔥 Danger Pass Heatmap")
    st.markdown("Visualize key passes that occur before a shot attempt.")
    
    # User inputs - no default values
    comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1, value=0, key="comp2")
    season_id = st.number_input("Enter Season ID:", min_value=0, step=1, value=0, key="season2")
    
    # Get available teams for this competition/season (but only show selection, not the list)
    available_teams = []
    if comp_id > 0 and season_id > 0:
        available_teams = get_available_teams(comp_id, season_id)
    
    if available_teams:
        team_required = st.selectbox("Select Team:", options=[""] + available_teams)
    else:
        team_required = st.text_input("Enter Team Name:")
    
    if st.button("🎯 Generate Heatmap"):
        if comp_id == 0 or season_id == 0:
            st.warning("Please enter valid Competition ID and Season ID.")
        elif not team_required:
            st.warning("Please enter/select a team name.")
        else:
            try:
                match_ids = get_team_matches(comp_id, season_id, team_required)
                
                if not match_ids:
                    st.warning(f"No matches found for team: {team_required}")
                else:
                    st.info(f"Processing {len(match_ids)} matches for {team_required}")
                    
                    all_danger_passes = []
                    processed_matches = 0
                    
                    for match_id in match_ids[:5]:  # Limit to first 5 matches for performance
                        event_file = os.path.join(events_path, f"{match_id}.json")
                        
                        if not os.path.exists(event_file):
                            st.warning(f"Event file not found for match {match_id}")
                            continue
                            
                        events_data = safe_load_json(event_file)
                        if events_data is None:
                            continue
                            
                        df = safe_json_normalize(events_data)
                        if df.empty:
                            continue
                        
                        processed_matches += 1
                        
                        # Filter for the team
                        team_filter = df['team_name'] == team_required if 'team_name' in df.columns else pd.Series([False] * len(df))
                        team_events = df[team_filter]
                        
                        if team_events.empty:
                            continue
                        
                        # Get passes and shots
                        passes = team_events[team_events['type_name'] == 'Pass'] if 'type_name' in team_events.columns else pd.DataFrame()
                        shots = team_events[team_events['type_name'] == 'Shot'] if 'type_name' in team_events.columns else pd.DataFrame()
                        
                        if passes.empty or shots.empty:
                            continue
                        
                        # Calculate danger passes (passes 15 seconds before shots)
                        shot_times = []
                        if 'minute' in shots.columns and 'second' in shots.columns:
                            shot_times = (shots['minute'] * 60 + shots['second']).tolist()
                        
                        if shot_times and not passes.empty:
                            shot_window = 15
                            danger_passes = []
                            
                            for _, pass_row in passes.iterrows():
                                if 'minute' in pass_row and 'second' in pass_row:
                                    pass_time = pass_row['minute'] * 60 + pass_row['second']
                                    
                                    # Check if pass is within 15 seconds before any shot
                                    for shot_time in shot_times:
                                        if shot_time - shot_window <= pass_time <= shot_time:
                                            # Skip corner passes
                                            is_corner = pass_row.get('pass_type_name') == 'Corner'
                                            if not is_corner and 'location' in pass_row and pass_row['location'] is not None:
                                                danger_passes.append(pass_row)
                                            break
                            
                            if danger_passes:
                                all_danger_passes.extend(danger_passes)
                    
                    st.info(f"Processed {processed_matches} matches")
                    
                    if all_danger_passes:
                        # Create heatmap
                        pitchLengthX, pitchWidthY = 120, 80
                        x_coords = []
                        y_coords = []
                        
                        for pass_data in all_danger_passes:
                            location = pass_data.get('location')
                            if isinstance(location, list) and len(location) >= 2:
                                x_coords.append(location[0])
                                y_coords.append(pitchWidthY - location[1])  # Flip Y-axis
                        
                        if x_coords and y_coords:
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Create heatmap
                            H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=10, 
                                                             range=[[0, pitchLengthX], [0, pitchWidthY]])
                            
                            # Plot heatmap
                            extent = [0, pitchLengthX, 0, pitchWidthY]
                            im = ax.imshow(H.T, extent=extent, origin='lower', cmap='Reds', alpha=0.8)
                            
                            # Add pitch lines
                            ax.plot([0, pitchLengthX], [pitchWidthY/2, pitchWidthY/2], 'white', linewidth=2)  # Halfway line
                            ax.plot([pitchLengthX/2, pitchLengthX/2], [0, pitchWidthY], 'white', linewidth=2)  # Center line
                            
                            # Add goals
                            goal_width = 7.32
                            goal_y_start = (pitchWidthY - goal_width) / 2
                            goal_y_end = (pitchWidthY + goal_width) / 2
                            ax.plot([0, 0], [goal_y_start, goal_y_end], 'white', linewidth=4)  # Left goal
                            ax.plot([pitchLengthX, pitchLengthX], [goal_y_start, goal_y_end], 'white', linewidth=4)  # Right goal
                            
                            ax.set_title(f"Danger Pass Heatmap: {team_required}\n({len(all_danger_passes)} danger passes from {processed_matches} matches)", 
                                        fontsize=14, color='white')
                            ax.set_xlabel("Pitch Length (m)", color='white')
                            ax.set_ylabel("Pitch Width (m)", color='white')
                            ax.set_xlim(0, pitchLengthX)
                            ax.set_ylim(0, pitchWidthY)
                            ax.set_facecolor('darkgreen')
                            
                            # Add colorbar
                            cbar = plt.colorbar(im, ax=ax)
                            cbar.set_label('Danger Pass Density', color='white')
                            cbar.ax.yaxis.set_tick_params(color='white')
                            cbar.ax.yaxis.label.set_color('white')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        else:
                            st.warning("No valid location data found for danger passes.")
                    else:
                        st.warning("No danger passes found for the selected team and matches.")
                        
            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")

# --- MODULE 3: Pass Comparison ---
elif option.endswith("3. Pass Comparison"):
    st.header("📊 Team Pass Comparison")
    
    # User inputs - no default values
    comp_id = st.number_input("Enter Competition ID:", min_value=0, step=1, value=0, key="comp3")
    season_id = st.number_input("Enter Season ID:", min_value=0, step=1, value=0, key="season3")
    
    if st.button("📈 Compare Teams"):
        if comp_id == 0 or season_id == 0:
            st.warning("Please enter valid Competition ID and Season ID.")
        else:
            try:
                match_file = os.path.join(matches_path, str(comp_id), f"{season_id}.json")
                matches_data = safe_load_json(match_file)
                
                if matches_data is None:
                    st.error("Could not load matches for this competition/season.")
                else:
                    team_stats = []
                    processed_matches = 0
                    
                    # Process each match
                    for match in matches_data[:10]:  # Limit to first 10 matches for performance
                        if not isinstance(match, dict) or 'match_id' not in match:
                            continue
                        
                        match_id = match['match_id']
                        event_file = os.path.join(events_path, f"{match_id}.json")
                        
                        if not os.path.exists(event_file):
                            continue
                            
                        events_data = safe_load_json(event_file)
                        if events_data is None:
                            continue
                        
                        df = safe_json_normalize(events_data)
                        if df.empty or 'team_name' not in df.columns or 'type_name' not in df.columns:
                            continue
                        
                        processed_matches += 1
                        
                        # Extract team names from match
                        home_team = ""
                        away_team = ""
                        
                        if 'home_team' in match and isinstance(match['home_team'], dict):
                            home_team = match['home_team'].get('home_team_name', '')
                        if 'away_team' in match and isinstance(match['away_team'], dict):
                            away_team = match['away_team'].get('away_team_name', '')
                        
                        # Process both teams
                        for team in [home_team, away_team]:
                            if not team:
                                continue
                            
                            team_events = df[df['team_name'] == team]
                            
                            if team_events.empty:
                                continue
                            
                            passes = len(team_events[team_events['type_name'] == 'Pass'])
                            shots = len(team_events[team_events['type_name'] == 'Shot'])
                            
                            # Get goals from match data
                            home_score = match.get('home_score', 0) if isinstance(match.get('home_score'), (int, float)) else 0
                            away_score = match.get('away_score', 0) if isinstance(match.get('away_score'), (int, float)) else 0
                            
                            goals = home_score if team == home_team else away_score
                            
                            team_stats.append({
                                "Team": team,
                                "Match_ID": match_id,
                                "Passes": passes,
                                "Shots": shots,
                                "Goals": goals
                            })
                    
                    if team_stats:
                        df_stats = pd.DataFrame(team_stats)
                        
                        # Aggregate by team
                        team_aggregated = df_stats.groupby('Team').agg({
                            'Passes': 'sum',
                            'Shots': 'sum', 
                            'Goals': 'sum',
                            'Match_ID': 'count'
                        }).rename(columns={'Match_ID': 'Matches_Played'})
                        
                        # Calculate averages
                        team_aggregated['Avg_Passes'] = team_aggregated['Passes'] / team_aggregated['Matches_Played']
                        team_aggregated['Avg_Shots'] = team_aggregated['Shots'] / team_aggregated['Matches_Played']
                        team_aggregated['Goals_Per_Match'] = team_aggregated['Goals'] / team_aggregated['Matches_Played']
                        
                        st.subheader(f"Team Comparison ({processed_matches} matches processed)")
                        st.dataframe(team_aggregated.round(2))
                        
                        # Create visualization
                        if len(team_aggregated) > 1:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            # Passes vs Shots scatter plot
                            ax1.scatter(team_aggregated['Avg_Passes'], team_aggregated['Avg_Shots'], 
                                       s=team_aggregated['Goals']*50+50, alpha=0.7, c=team_aggregated['Goals'], cmap='viridis')
                            ax1.set_xlabel('Average Passes per Match')
                            ax1.set_ylabel('Average Shots per Match')
                            ax1.set_title('Passes vs Shots (bubble size = total goals)')
                            
                            # Add team labels
                            for team, row in team_aggregated.iterrows():
                                ax1.annotate(team[:10], (row['Avg_Passes'], row['Avg_Shots']), 
                                            xytext=(5, 5), textcoords='offset points', fontsize=8)
                            
                            # Goals per match bar chart
                            teams_sorted = team_aggregated.sort_values('Goals_Per_Match', ascending=True)
                            ax2.barh(range(len(teams_sorted)), teams_sorted['Goals_Per_Match'])
                            ax2.set_yticks(range(len(teams_sorted)))
                            ax2.set_yticklabels([team[:15] for team in teams_sorted.index])
                            ax2.set_xlabel('Goals per Match')
                            ax2.set_title('Goals per Match by Team')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.warning("No valid data found for comparison.")
                        
            except Exception as e:
                st.error(f"Error in pass comparison: {str(e)}")

# --- MODULE 4: Possession Chain Viewer ---
elif option.endswith("4. Possession Chain Viewer"):
    st.header("🔗 Possession Chain Viewer")
    
    # User inputs - no default values, manual entry
    match_id = st.text_input("Enter Match ID:")
    selected_team = st.text_input("Enter Team Name:")
    
    if match_id and selected_team:
        if st.button("🔍 Load Possession Chains"):
            try:
                match_id_int = int(match_id)
                event_file = os.path.join(events_path, f"{match_id_int}.json")
                
                if os.path.exists(event_file):
                    events_data = safe_load_json(event_file)
                    
                    if events_data is not None:
                        df = safe_json_normalize(events_data)
                        
                        if not df.empty:
                            st.info(f"Loaded {len(df)} events from match {match_id}")
                            
                            # Debug: Show available columns
                            st.write(f"**Available columns:** {list(df.columns)}")
                            
                            # Check if team exists in the data
                            if 'team_name' in df.columns:
                                available_teams_in_match = df['team_name'].unique()
                                st.write(f"**Teams in this match:** {list(available_teams_in_match)}")
                                
                                # Check for possession column
                                if 'possession' in df.columns:
                                    team_events = df[df['team_name'] == selected_team]
                                    
                                    if not team_events.empty:
                                        possessions = team_events['possession'].unique()
                                        st.info(f"Found {len(possessions)} possessions for {selected_team}")
                                        
                                        # Event type selection
                                        if 'type_name' in df.columns:
                                            event_types = sorted(df['type_name'].dropna().unique())
                                            st.write(f"**Available event types:** {event_types}")
                                            
                                            selected_events = st.multiselect(
                                                "Select Event Types to visualize:", 
                                                options=event_types,
                                                default=['Pass', 'Shot'] if all(x in event_types for x in ['Pass', 'Shot']) else event_types[:3]
                                            )
                                            
                                            if selected_events:
                                                # Create pitch visualization
                                                fig = go.Figure()
                                                
                                                # Add pitch outline
                                                fig.add_shape(
                                                    type="rect",
                                                    x0=0, y0=0, x1=120, y1=80,
                                                    line=dict(color="white", width=2),
                                                    fillcolor="green",
                                                    opacity=0.3
                                                )
                                                
                                                # Add center line
                                                fig.add_shape(
                                                    type="line",
                                                    x0=60, y0=0, x1=60, y1=80,
                                                    line=dict(color="white", width=2)
                                                )
                                                
                                                # Add center circle
                                                fig.add_shape(
                                                    type="circle",
                                                    x0=50, y0=35, x1=70, y1=45,
                                                    line=dict(color="white", width=2)
                                                )
                                                
                                                colors = ['red', 'blue', 'green', 'orange', 'purple']
                                                chains_plotted = 0
                                                
                                                # Plot possession chains (limit to first 5 for clarity)
                                                for i, possession in enumerate(possessions[:5]):
                                                    poss_events = team_events[
                                                        (team_events['possession'] == possession) & 
                                                        (team_events['type_name'].isin(selected_events))
                                                    ]
                                                    
                                                    if not poss_events.empty and 'location' in poss_events.columns:
                                                        x_coords = []
                                                        y_coords = []
                                                        event_info = []
                                                        
                                                        for _, event in poss_events.iterrows():
                                                            location = event.get('location')
                                                            if isinstance(location, list) and len(location) >= 2:
                                                                x_coords.append(location[0])
                                                                y_coords.append(location[1])
                                                                minute = event.get('minute', '?')
                                                                second = event.get('second', '?')
                                                                event_type = event.get('type_name', 'Unknown')
                                                                player = event.get('player_name', 'Unknown')
                                                                event_info.append(f"{minute}:{second:02d} - {event_type} by {player}")
                                                        
                                                        if len(x_coords) > 0:
                                                            chains_plotted += 1
                                                            color = colors[i % len(colors)]
                                                            fig.add_trace(go.Scatter(
                                                                x=x_coords,
                                                                y=y_coords,
                                                                mode='markers+lines',
                                                                name=f'Possession {possession} ({len(x_coords)} events)',
                                                                line=dict(color=color, width=3),
                                                                marker=dict(color=color, size=8),
                                                                text=event_info,
                                                                hovertemplate='%{text}<extra></extra>'
                                                            ))
                                                
                                                # Update layout
                                                fig.update_layout(
                                                    title=f"Possession Chains for {selected_team} - Match {match_id}<br>{chains_plotted} chains plotted",
                                                    xaxis_title="Pitch Length (m)",
                                                    yaxis_title="Pitch Width (m)",
                                                    xaxis=dict(range=[0, 120], showgrid=True),
                                                    yaxis=dict(range=[0, 80], showgrid=True),
                                                    showlegend=True,
                                                    width=900,
                                                    height=600,
                                                    plot_bgcolor='rgba(0,0,0,0)',
                                                    paper_bgcolor='rgba(0,0,0,0)'
                                                )
                                                
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                                if chains_plotted == 0:
                                                    st.warning("No possession chains with location data found for the selected events.")
                                            else:
                                                st.info("Please select at least one event type.")
                                        else:
                                            st.warning("No event type information found in the data.")
                                    else:
                                        st.warning(f"Team '{selected_team}' not found in match {match_id}. Available teams: {list(available_teams_in_match)}")
                                else:
                                    # Try to use different possession field names
                                    poss_fields = [col for col in df.columns if 'possess' in col.lower()]
                                    if poss_fields:
                                        st.warning(f"'possession' column not found. Found similar columns: {poss_fields}")
                                    else:
                                        st.warning("No possession information found in the data.")
                            else:
                                st.warning("No team information found in the event data.")
                        else:
                            st.warning("Event data is empty.")
                    else:
                        st.error("Could not load event data - file may be corrupted.")
                else:
                    st.error(f"Event file not found for match {match_id}. File path: {event_file}")
                    
            except ValueError:
                st.error("Please enter a valid numeric Match ID.")
            except Exception as e:
                st.error(f"Error loading possession chains: {str(e)}")
                st.write(f"Debug info: {str(e)}")
    else:
        st.info("Please enter both Match ID and Team Name to load possession chains.")

# --- Footer ---
st.markdown("---")
st.markdown("⚽ **Soccer Analysis Toolkit** - Built with Streamlit")
st.markdown("*Note: This app processes StatsBomb football data in JSON format.*")

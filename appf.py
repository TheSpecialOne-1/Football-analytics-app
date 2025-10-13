import streamlit as st
import sqlite3
import hashlib
import re
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
from datetime import datetime

# Database setup
def init_database():
    """Initialize the user database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Create users table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            nationality TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            date_of_birth DATE,
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)

    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number format"""
    pattern = r'^[+]?[1-9]?[0-9]{7,15}$'
    return re.match(pattern, phone) is not None

def register_user(username, email, password, full_name, nationality, phone_number, date_of_birth):
    """Register a new user"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            conn.close()
            return False, "Username or email already exists"

        # Insert new user
        password_hash = hash_password(password)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, nationality, phone_number, date_of_birth)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username, email, password_hash, full_name, nationality, phone_number, date_of_birth))

        conn.commit()
        conn.close()
        return True, "Registration successful!"

    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def login_user(username_or_email, password):
    """Authenticate user login"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Check if input is email or username
        if '@' in username_or_email:
            cursor.execute('SELECT * FROM users WHERE email = ?', (username_or_email,))
        else:
            cursor.execute('SELECT * FROM users WHERE username = ?', (username_or_email,))

        user = cursor.fetchone()

        if user and verify_password(password, user[3]):  # user[3] is password_hash
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                         (datetime.now(), user[0]))
            conn.commit()
            conn.close()
            return True, user
        else:
            conn.close()
            return False, None

    except Exception as e:
        return False, None

def get_user_profile(user_id):
    """Get user profile information"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        return user
    except:
        return None

# Initialize database
init_database()

# Set page config
st.set_page_config(page_title="Soccer Analysis App", layout="wide")

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'show_register' not in st.session_state:
    st.session_state.show_register = False

# Authentication UI functions
# --- LOGIN ---
def show_login():
    st.title("🔐 Login")
    login_mode = st.radio("Login with", ["Email", "Username"], horizontal=True, key="login_mode")

    with st.form("login_form", clear_on_submit=False):
        if login_mode == "Email":
            identifier = st.text_input("Email", placeholder="name@example.com")
        else:
            identifier = st.text_input("Username", placeholder="your_username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if login_mode == "Email":
            if not validate_email(identifier):
                st.error("Please enter a valid email address")
                return
            success, user_data = login_user(identifier, password)  # login_user already detects '@'
        else:
            success, user_data = login_user(identifier, password)

        if success:
            st.session_state.logged_in = True
            st.session_state.user_data = user_data
            st.success("Welcome back!")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.caption("No account yet?")
    if st.button("Create an account"):
        st.session_state.show_register = True
        st.rerun()

# --- REGISTER ---
def show_register():
    st.title("📝 Create Account")
    registered_now = False

    with st.form("register_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username*")
            email = st.text_input("Email*")
            password = st.text_input("Password*", type="password")
            confirm = st.text_input("Confirm Password*", type="password")
        with col2:
            full_name = st.text_input("Full Name*")
            nationality = st.selectbox(
                "Nationality*",
                ["", "Afghanistan", "Albania", "Algeria", "Argentina", "Australia", "Austria", "Bangladesh",
                 "Belgium", "Brazil", "Canada", "China", "Denmark", "Egypt", "Finland", "France", "Germany",
                 "India", "Indonesia", "Iran", "Iraq", "Italy", "Japan", "Mexico", "Netherlands", "Norway",
                 "Pakistan", "Poland", "Portugal", "Russia", "Saudi Arabia", "South Africa", "Spain", "Sweden",
                 "Switzerland", "Turkey", "United Kingdom", "United States", "Other"]
            )
            phone = st.text_input("Phone Number*", placeholder="+1234567890")
            dob = st.date_input("Date of Birth",
                   min_value=datetime(1920, 1, 1).date(),
                   max_value=datetime.now().date(),
                   value=datetime(1990, 1, 1).date())

        submit = st.form_submit_button("Create Account")

    if submit:
        errs = []
        if not username or len(username) < 3:
            errs.append("Username must be at least 3 characters")
        if not email or not validate_email(email):
            errs.append("Enter a valid email address")
        if not password or len(password) < 6:
            errs.append("Password must be at least 6 characters")
        if password != confirm:
            errs.append("Passwords do not match")
        if not full_name:
            errs.append("Full name is required")
        if not nationality:
            errs.append("Please select your nationality")
        if not phone or not validate_phone(phone):
            errs.append("Enter a valid phone number")

        if errs:
            for e in errs:
                st.error(e)
        else:
            ok, msg = register_user(username, email, password, full_name, nationality, phone, dob)
            if ok:
                st.success("Registration successful! You can now login with your credentials.")
                # Set a flag to switch screens after the form completes rendering
                st.session_state.show_register = False
                st.session_state.just_registered = True
                registered_now = True
            else:
                st.error(msg)

    # Move navigation buttons OUTSIDE the form to avoid APIException
    if not registered_now:
        st.markdown("---")
        if st.button("Back to Login"):
            st.session_state.show_register = False
            st.rerun()

# After defining these functions, add this small hook in your main logic
if st.session_state.get("just_registered"):
    # Clear the flag and rerun to show the login screen cleanly
    st.session_state.just_registered = False
    st.rerun()


def show_user_profile():
    """Show user profile in sidebar"""
    if st.session_state.logged_in and st.session_state.user_data:
        user = st.session_state.user_data
        st.sidebar.markdown("### 👤 User Profile")
        st.sidebar.write(f"**Welcome, {user[4]}!**")  # full_name
        st.sidebar.write(f"Username: {user[1]}")      # username
        st.sidebar.write(f"Email: {user[2]}")         # email
        st.sidebar.write(f"Nationality: {user[5]}")   # nationality

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_data = None
            st.session_state.show_register = False
            st.rerun()

# Main app logic
if not st.session_state.logged_in:
    if st.session_state.show_register:
        show_register()
    else:
        show_login()
else:
    # Main app content - All original functionality from appf.py
    st.title("⚽ Soccer Analysis Toolkit")

    # Show user profile in sidebar
    show_user_profile()

    # Add custom CSS
    st.markdown("""

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

    # Initialize session state for original app functionality
    if 'possession_data_loaded' not in st.session_state:
        st.session_state.possession_data_loaded = False
    if 'possession_df' not in st.session_state:
        st.session_state.possession_df = None
    if 'possession_team_events' not in st.session_state:
        st.session_state.possession_team_events = None
    if 'possession_possessions' not in st.session_state:
        st.session_state.possession_possessions = []
    if 'possession_event_types' not in st.session_state:
        st.session_state.possession_event_types = []
    if 'possession_match_id' not in st.session_state:
        st.session_state.possession_match_id = ""
    if 'possession_team' not in st.session_state:
        st.session_state.possession_team = ""

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

                            # Create match summary table with team matchups and match IDs
                            match_data = []
                            for match in matches:
                                if isinstance(match, dict) and 'match_id' in match:
                                    match_id = match['match_id']

                                    # Extract team names
                                    home_team = "Unknown"
                                    away_team = "Unknown"

                                    if 'home_team' in match:
                                        if isinstance(match['home_team'], dict):
                                            home_team = match['home_team'].get('home_team_name', 'Unknown')
                                        elif isinstance(match['home_team'], str):
                                            home_team = match['home_team']

                                    if 'away_team' in match:
                                        if isinstance(match['away_team'], dict):
                                            away_team = match['away_team'].get('away_team_name', 'Unknown')
                                        elif isinstance(match['away_team'], str):
                                            away_team = match['away_team']

                                    # Get match date if available
                                    match_date = match.get('match_date', 'Unknown')

                                    # Get scores if available
                                    home_score = match.get('home_score', '?')
                                    away_score = match.get('away_score', '?')

                                    match_data.append({
                                        'Match_ID': match_id,
                                        'Home_Team': home_team,
                                        'Away_Team': away_team,
                                        'Score': f"{home_score} - {away_score}",
                                        'Date': match_date,
                                        'Matchup': f"{home_team} vs {away_team}"
                                    })

                            if match_data:
                                matches_df = pd.DataFrame(match_data)
                                st.subheader("Match Fixtures:")
                                st.dataframe(matches_df, use_container_width=True)

                                # Show available teams
                                teams = get_available_teams(comp_id, season_id)
                                if teams:
                                    st.subheader("Available Teams:")
                                    team_cols = st.columns(4)
                                    for i, team in enumerate(teams):
                                        with team_cols[i % 4]:
                                            st.write(f"• {team}")
                            else:
                                st.warning("No valid match data found.")
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
                        errors_count = 0

                        # Process ALL matches for the team (removed limit)
                        for i, match_id in enumerate(match_ids, 1):
                            st.write(f"Processing match {i}/{len(match_ids)}: {match_id}")

                            event_file = os.path.join(events_path, f"{match_id}.json")
                            if not os.path.exists(event_file):
                                st.warning(f"Event file not found for match {match_id}")
                                errors_count += 1
                                continue

                            events_data = safe_load_json(event_file)
                            if events_data is None:
                                st.warning(f"Could not load event data for match {match_id}")
                                errors_count += 1
                                continue

                            df = safe_json_normalize(events_data)
                            if df.empty:
                                st.warning(f"Empty event data for match {match_id}")
                                errors_count += 1
                                continue

                            processed_matches += 1

                            # Filter for the team
                            team_filter = df['team_name'] == team_required if 'team_name' in df.columns else pd.Series([False] * len(df))
                            team_events = df[team_filter]

                            if team_events.empty:
                                st.warning(f"No events found for {team_required} in match {match_id}")
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
                                    st.write(f" → Found {len(danger_passes)} danger passes")

                        st.success(f"✅ Processed {processed_matches}/{len(match_ids)} matches successfully")
                        if errors_count > 0:
                            st.warning(f"⚠️ {errors_count} matches had errors")

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
                        total_matches = len(matches_data)

                        # Process ALL matches (removed limit)
                        for i, match in enumerate(matches_data, 1):
                            if not isinstance(match, dict) or 'match_id' not in match:
                                continue

                            match_id = match['match_id']
                            st.write(f"Processing match {i}/{total_matches}: {match_id}")

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

                            st.success(f"✅ Processed {processed_matches} matches")
                            st.subheader(f"Team Comparison - All {len(team_aggregated)} Teams")
                            st.dataframe(team_aggregated.round(2), use_container_width=True)

                            # Create visualization
                            if len(team_aggregated) > 1:
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

                                # Passes vs Shots scatter plot
                                scatter = ax1.scatter(team_aggregated['Avg_Passes'], team_aggregated['Avg_Shots'],
                                                    s=team_aggregated['Goals']*30+50, alpha=0.7,
                                                    c=team_aggregated['Goals'], cmap='viridis')
                                ax1.set_xlabel('Average Passes per Match')
                                ax1.set_ylabel('Average Shots per Match')
                                ax1.set_title('Passes vs Shots (bubble size = total goals)')

                                # Add team labels
                                for team, row in team_aggregated.iterrows():
                                    ax1.annotate(team[:8], (row['Avg_Passes'], row['Avg_Shots']),
                                               xytext=(3, 3), textcoords='offset points', fontsize=7)

                                # Add colorbar
                                plt.colorbar(scatter, ax=ax1, label='Total Goals')

                                # Goals per match bar chart
                                teams_sorted = team_aggregated.sort_values('Goals_Per_Match', ascending=True)
                                bars = ax2.barh(range(len(teams_sorted)), teams_sorted['Goals_Per_Match'])
                                ax2.set_yticks(range(len(teams_sorted)))
                                ax2.set_yticklabels([team[:12] for team in teams_sorted.index], fontsize=8)
                                ax2.set_xlabel('Goals per Match')
                                ax2.set_title(f'Goals per Match - All {len(teams_sorted)} Teams')
                                ax2.grid(axis='x', alpha=0.3)

                                # Add value labels on bars
                                for i, bar in enumerate(bars):
                                    width = bar.get_width()
                                    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                           f'{width:.2f}', ha='left', va='center', fontsize=7)

                                plt.tight_layout()
                                st.pyplot(fig)

                                # Summary statistics
                                st.subheader("Competition Summary")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Teams", len(team_aggregated))
                                with col2:
                                    st.metric("Total Goals", int(team_aggregated['Goals'].sum()))
                                with col3:
                                    st.metric("Avg Goals/Team", f"{team_aggregated['Goals_Per_Match'].mean():.2f}")
                                with col4:
                                    st.metric("Top Scorer", f"{team_aggregated['Goals'].idxmax()}")
                        else:
                            st.warning("No valid data found for comparison.")

                except Exception as e:
                    st.error(f"Error in pass comparison: {str(e)}")

    # --- MODULE 4: Possession Chain Viewer ---
    elif option.endswith("4. Possession Chain Viewer"):
        st.header("🔗 Possession Chain Viewer")

        # User inputs
        match_id = st.text_input("Enter Match ID:")
        selected_team = st.text_input("Enter Team Name:")

        # Load data button
        if st.button("🔍 Load Match Data"):
            if match_id and selected_team:
                try:
                    match_id_int = int(match_id)
                    event_file = os.path.join(events_path, f"{match_id_int}.json")

                    if os.path.exists(event_file):
                        events_data = safe_load_json(event_file)
                        if events_data is not None:
                            df = safe_json_normalize(events_data)
                            if not df.empty:
                                st.success(f"✅ Loaded {len(df)} events from match {match_id}")

                                # Check if team exists in the data
                                if 'team_name' in df.columns:
                                    available_teams_in_match = df['team_name'].unique()
                                    st.info(f"Teams in this match: {', '.join(available_teams_in_match)}")

                                    # Check for possession column
                                    if 'possession' in df.columns:
                                        team_events = df[df['team_name'] == selected_team]
                                        if not team_events.empty:
                                            possessions = team_events['possession'].unique()
                                            st.info(f"Found {len(possessions)} possessions for {selected_team}")

                                            # Store data in session state
                                            st.session_state.possession_data_loaded = True
                                            st.session_state.possession_df = df
                                            st.session_state.possession_team_events = team_events
                                            st.session_state.possession_possessions = possessions
                                            st.session_state.possession_match_id = match_id
                                            st.session_state.possession_team = selected_team

                                            # Get event types
                                            if 'type_name' in df.columns:
                                                event_types = sorted(df['type_name'].dropna().unique())
                                                st.session_state.possession_event_types = event_types
                                            else:
                                                st.warning("No event type information found in the data.")
                                                st.session_state.possession_data_loaded = False
                                        else:
                                            st.warning(f"Team '{selected_team}' not found in match {match_id}. Available teams: {list(available_teams_in_match)}")
                                            st.session_state.possession_data_loaded = False
                                    else:
                                        st.warning("No possession information found in the data.")
                                        st.session_state.possession_data_loaded = False
                                else:
                                    st.warning("No team information found in the event data.")
                                    st.session_state.possession_data_loaded = False
                            else:
                                st.warning("Event data is empty.")
                                st.session_state.possession_data_loaded = False
                        else:
                            st.error("Could not load event data - file may be corrupted.")
                            st.session_state.possession_data_loaded = False
                    else:
                        st.error(f"Event file not found for match {match_id}")
                        st.session_state.possession_data_loaded = False

                except ValueError:
                    st.error("Please enter a valid numeric Match ID.")
                    st.session_state.possession_data_loaded = False
                except Exception as e:
                    st.error(f"Error loading match data: {str(e)}")
                    st.session_state.possession_data_loaded = False
            else:
                st.warning("Please enter both Match ID and Team Name.")
                st.session_state.possession_data_loaded = False

        # Show event selection and visualization if data is loaded
        if st.session_state.possession_data_loaded and st.session_state.possession_event_types:
            st.markdown("---")
            st.subheader("Event Selection & Visualization")

            # Event type selection
            selected_events = st.multiselect(
                "Select Event Types to visualize:",
                options=st.session_state.possession_event_types,
                default=[],
                key="event_selection"
            )

            # Generate visualization button
            if st.button("🎨 Generate Visualization") or selected_events:
                if selected_events:
                    try:
                        team_events = st.session_state.possession_team_events
                        possessions = st.session_state.possession_possessions

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

                        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan']
                        chains_plotted = 0

                        # Plot possession chains
                        for i, possession in enumerate(possessions[:8]):
                            try:
                                poss_events = team_events[
                                    (team_events['possession'] == possession) &
                                    (team_events['type_name'].isin(selected_events))
                                ]

                                if not poss_events.empty and 'location' in poss_events.columns:
                                    x_coords = []
                                    y_coords = []
                                    event_info = []

                                    for _, event in poss_events.iterrows():
                                        try:
                                            location = event.get('location')
                                            if isinstance(location, list) and len(location) >= 2:
                                                x_coords.append(float(location[0]))
                                                y_coords.append(float(location[1]))

                                                minute = event.get('minute', '?')
                                                second = event.get('second', '?')
                                                event_type = event.get('type_name', 'Unknown')
                                                player = event.get('player_name', 'Unknown')

                                                # Format seconds safely
                                                if isinstance(second, (int, float)):
                                                    second_str = f"{int(second):02d}"
                                                else:
                                                    second_str = str(second)

                                                event_info.append(f"{minute}:{second_str} - {event_type} by {player}")
                                        except Exception:
                                            continue

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
                                            hovertemplate='%{text}'
                                        ))

                            except Exception as e:
                                st.warning(f"Error processing possession {possession}: {str(e)}")
                                continue

                        # Update layout
                        fig.update_layout(
                            title=f"Possession Chains for {st.session_state.possession_team} - Match {st.session_state.possession_match_id} ({chains_plotted} chains plotted)",
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
                            st.warning("No possession chains with valid location data found for the selected events.")

                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                else:
                    st.info("Please select at least one event type to visualize.")

    # --- Footer ---
    st.markdown("---")
    st.markdown("⚽ **Soccer Analysis Toolkit** - Built with Streamlit")
    st.markdown("*Note: This app processes StatsBomb football data in JSON format.*")



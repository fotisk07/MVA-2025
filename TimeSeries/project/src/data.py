import numpy as np
import pandas as pd
from soccercpd.record_manager import RecordManager
from soccercpd.match import Match
from scipy.stats import shapiro
from sklearn.cluster import KMeans
from statsbombpy import sb

rng = np.random.default_rng(0)

def simulate_roles(T=400, cp=200, sigma=2.0):
    """
    Simulates role trajectories with a single formation change.
    """
    F1 = np.array([
    (-40,  20), (-40,   7), (-40,  -7), (-40, -20),   # Defenders (4)
    (-10,  20), (-10,   7), (-10,  -7), (-10, -20),   # Midfield (4)
    ( 25,   8), ( 25,  -8)                            # Forwards (2)
    ])
    F2 = np.array([
        (-40,  10), (-40,   0), (-40, -10),               # Defenders (3)
        (-10,  20), (-10,   7), (-10,  -7), (-10, -20),   # Midfield (4)
        ( 25,  18), ( 25,   0), ( 25, -18)                # Forwards (3)
    ])
    N = len(F1)
    V = np.zeros((T, N, 2))

    for t in range(T):
        mu = F1 if t < cp else F2
        V[t] = mu + sigma * np.random.randn(N,2)

    return V

def simulate_swap(T=300, sigma=0.5):
    # 6-role formation
    mu = np.array([
        [-8,  2], [-2, 2], [4, 2],
        [-4, -2], [2, -2],
        [0, -6]
    ])

    N = len(mu)
    X = np.zeros((T, N, 2))

    # players assigned fixed roles initially
    roles = np.arange(N)

    # choose two roles to swap
    k1, k2 = 3, 4

    for t in range(T):
        alpha = 0
        if T//3 <= t <= 2*T//3:
            alpha = (t - T//3) / (T//3)

        for i in range(N):
            k = roles[i]
            center = mu[k]

            # interpolate swap
            if k == k1:
                center = (1-alpha)*mu[k1] + alpha*mu[k2]
            if k == k2:
                center = (1-alpha)*mu[k2] + alpha*mu[k1]

            X[t, i] = center + sigma * rng.standard_normal(2)

    return X, mu

def generate_synthetic_role_data(T=450, N=10, switch_times=[150, 300], noise_level=0.1):
    """
    Generates a sequence of role permutations.
    Returns: Array of shape (T, N) where row t is the permutation at time t.
    """
    # Phase 1
    # Base permutation: Identity
    base_roles = np.arange(N)
    
    # Phase 2
    # Tactical Change: Swap roles 7 and 9
    phase2_roles = base_roles.copy()
    phase2_roles[7], phase2_roles[9] = 9, 7

    # Phase 3
    # Tactical Change: Swap roles 3 and 4
    phase3_roles = base_roles.copy()
    phase3_roles[3], phase3_roles[4] = 4, 3

    permutations = []
    
    for t in range(T):
        # Determine dominant tactic
        if t < switch_times[0]:
            current_perm = base_roles.copy()
        elif t < switch_times[1]:
            current_perm = phase2_roles.copy()
        else:
            current_perm = phase3_roles.copy()
            
        # Add temporary noise (random temporary swaps)
        if np.random.rand() < noise_level:
            # Randomly swap two players to simulate temporary cover/overlap
            i, j = np.random.choice(N, 2, replace=False)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
        permutations.append(current_perm)
        
    return np.array(permutations)

def load_and_prepare_match(activity_id=17985):
    rm = RecordManager()
    activity_record, player_periods, roster, ugp = rm.load_activity_data(activity_id)

    match = Match(activity_record, player_periods, roster, ugp)
    match.construct_inplay_ugp()
    match.rotate_pitch()

    return match


class FitogetherDiagnosis:
    """
    Handles loading, diagnosis, and validation of 10Hz GPS tracking data used by the Fitogether Inc. (SoccerCPD) researchers.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None

    def check_and_fix_orientation(self):
        """
        Ensures the team is always attacking from left to right in metric coordinates.
        """
        print("\n--- [Fitogether] Orientation Check ---")
        df = self.data

        medians = df.groupby('player_id')['x'].median()
        min_x_player = medians.min()
        max_x_player = medians.max()
        print(f"Player X Medians Range: {min_x_player:.2f} to {max_x_player:.2f}")

        # Check where the deepest player is
        # Since there are no goalkeepers, we assume the deepest player indicates orientation (attackers go right)
        if abs(min_x_player) > abs(max_x_player):
            df['x'] = -df['x']
            df['y'] = -df['y']
            print("Orientation corrected: Team now attacks left to right.")
            self.data = df
        else:
            print("Orientation is left to right. No changes made.")

    def load_or_generate_data(self, PITCH_LENGTH,PITCH_WIDTH):
        print(f"Loading data from {self.data_path}.")
        # K-League (SoccerCPD) data is available in a .ugp format.
        print(self.data_path.split(".")[-1])
        if self.data_path.split(".")[-1] == "parquet":
                self.data = pd.read_parquet(self.data_path)
        else:
            self.data = pd.read_pickle(self.data_path)
        
        if 'unixtime' in self.data.columns:
            self.data = self.data.rename(columns={'unixtime': 'frame'})

        # Convert from cm to m if necessary
        if self.data['x'].abs().max() > 150:
            self.data['x'] = self.data['x'] / 100.0
            self.data['y'] = self.data['y'] / 100.0

        # Transform from SoccerCPD coordinates to Metric coordinates
        min_x, max_x = self.data['x'].min(), self.data['x'].max()
        min_y, max_y = self.data['y'].min(), self.data['y'].max()

        # Case A: Already Metric (Centered, ~ -52 to 52)
        if min_x < -10:
            print("Format: Metric (Centered). No transformation needed.")
            
        # Case B: Last Row Format (0-100 on X and Y)
        # Detected if Y goes above 85 (since standard format stops at 80)
        elif max_y > 85.0:
            print("Format: Last Row (0-100). Transforming...")
            # Scale 100 -> 105m (Length) and 100 -> 68m (Width)
            self.data['x'] = (self.data['x'] / 100.0 * PITCH_LENGTH) - (PITCH_LENGTH / 2)
            self.data['y'] = (PITCH_WIDTH / 2) - (self.data['y'] / 100.0 * PITCH_WIDTH)

        # Case C: SoccerCPD/Fitogether Format (0-120 X, 0-80 Y)
        # Default for positive coordinates
        else:
            print("Format: SoccerCPD (0-120). Transforming...")
            self.data['x'] = (self.data['x'] / 120.0 * PITCH_LENGTH) - (PITCH_LENGTH / 2)
            self.data['y'] = (PITCH_WIDTH / 2) - (self.data['y'] / 80.0 * PITCH_WIDTH)

        self.check_and_fix_orientation()
            
        return self.data

    # def _generate_synthetic_data(self, n_frames=5400, n_players=10):
    #     """
    #     Generates realistic-looking dummy tracking data.
    #     """
    #     records = []
        
    #     # Define 4-3-3 Formation Centers (10 Outfield Players)
    #     # Coordinates assume metric, center origin (0,0)
    #     # Order: LB, LCB, RCB, RB, CDM, LCM, RCM, LW, CF, RW
    #     formation_433 = np.array([
    #         [-10.0,  20.0],  # Left Back
    #         [-20.0,   7.0],  # Left Center Back
    #         [-20.0,  -7.0],  # Right Center Back
    #         [-10.0, -20.0],  # Right Back
    #         [ -5.0,   0.0],  # Defensive Mid
    #         [ 10.0,  15.0],  # Left Center Mid
    #         [ 10.0, -15.0],  # Right Center Mid
    #         [ 30.0,  25.0],  # Left Wing
    #         [ 35.0,   0.0],  # Center Forward
    #         [ 30.0, -25.0]   # Right Wing
    #     ])

    #     # Initialize player role centers
    #     if n_players <= len(formation_433):
    #         role_centers = formation_433[:n_players].copy()
    #     else:
    #         # Add random positions if more than 10 players requested
    #         extra_needed = n_players - len(formation_433)
    #         extra_roles = np.random.uniform(-30, 30, size=(extra_needed, 2))
    #         role_centers = np.vstack([formation_433, extra_roles])

    #     for t in range(n_frames):
    #         for p_id in range(n_players):
    #             # Add Gaussian noise (position) and high-freq jitter
    #             center = role_centers[p_id]
    #             noise = np.random.normal(0, 4, 2)  # Tactical movement
    #             jitter = np.random.normal(0, 0.1, 2) # GPS Jitter
                
    #             # Drift centers slightly over time
    #             role_centers[p_id] += np.random.normal(0, 0.05, 2)
                
    #             pos = center + noise + jitter
                
    #             # Clip to pitch
    #             pos[0] = np.clip(pos[0], *METRIC_X_RANGE)
    #             pos[1] = np.clip(pos[1], *METRIC_Y_RANGE)
                
    #             records.append({
    #                 'player_id': p_id,
    #                 'frame': t,
    #                 'x': pos[0],
    #                 'y': pos[1],
    #                 'speed': np.linalg.norm(np.random.normal(0, 2, 2))
    #             })
    #     return pd.DataFrame(records)

    def diagnose_signal_quality(self):
        """
        Performs data diagnosis:
        1. Missing Value Check
        2. High-Frequency Jitter Analysis
        """
        print("\n--- [Fitogether] Signal Diagnosis ---")
        df = self.data
        
        # Missing Values
        missing_count = df[['x', 'y']].isnull().sum().sum()
        total_points = len(df) * 2
        missing_pct = (missing_count / total_points) * 100
        
        print(f"Completeness Check: {100 - missing_pct:.4f}% available (Missing: {missing_pct:.4f}%)")
        if missing_pct > 1.0:
            print("Warning: Missing data exceeds 1% threshold. Interpolation required.")
        else:
            print("Signal completeness is acceptable.")

        unique_players = df['player_id'].unique()
        if len(unique_players) == 0:
            print("Error: No player data found.")
            return
        
        sample_player_id = unique_players[0]

        # Jitter Analysis (Velocity Variance): Calculate frame-to-frame displacement for a sample player
        p0 = df[df['player_id'] == sample_player_id].sort_values('frame')
        displacements = np.diff(p0[['x', 'y']].values, axis=0)
        # 10Hz => dt = 0.1s
        inst_speeds = np.linalg.norm(displacements, axis=1) * 10 
        
        speed_std = np.std(inst_speeds)
        print(f"Jitter Analysis (Speed Std Dev) for player {sample_player_id}: {speed_std:.2f} m/s")
        if speed_std > 1: # Threshold for "noisy"
            print("High-frequency jitter detected (consistent with raw GPS).")

    def verify_gaussian_assumption(self):
        """
        Verifies the Gaussian assumption (Normality) of player roles.
        Since we use our own pipeline, we use K-Means to approximate 'roles' and test their distribution.
        """
        print("\n--- [Fitogether] Gaussian Assumption Check ---")
        df = self.data
        
        # Approximate Roles using K-Means
        coords = df[['x', 'y']].values
        kmeans = KMeans(n_clusters=10, random_state=0).fit(coords)
        df['proxy_role'] = kmeans.labels_
        
        # Shapiro-Wilk Test on the largest cluster
        # Sample one role
        role_id = df['proxy_role'].value_counts().idxmax()
        role_data = df[df['proxy_role'] == role_id][['x', 'y']]
        
        # Downsample for Shapiro (it's sensitive to large N)
        sample = role_data.sample(n=min(500, len(role_data)), random_state=0)
        
        stat_x, p_x = shapiro(sample['x'])
        stat_y, p_y = shapiro(sample['y'])
        
        print(f"Testing Role Cluster #{role_id} (N={len(sample)}):")
        print(f"P-value (X-coord): {p_x:.2e}")
        print(f"P-value (Y-coord): {p_y:.2e}")
        
        if p_x < 0.05 or p_y < 0.05:
            print("Significant deviation from Normality (p < 0.05) found.")
        else:
            print("Data is consistent with Gaussian assumption.")


class StatsBombAdapter:
    def __init__(self, match_id=15946): # Default: Barcelona 2010/11 Season
        self.match_id = match_id
        self.events = None

    def fetch_and_transform(self, PITCH_LENGTH,PITCH_WIDTH):
        print(f"\n--- [StatsBomb] Fetching Event Data (Match ID: {self.match_id}) ---")
        try:
            # Fetch
            events = sb.events(match_id=self.match_id)
            loc_events = events[events['location'].notna()].copy().reset_index(drop=True)
            
            locs = np.vstack(loc_events['location'].values)
            loc_events['sb_x'] = locs[:, 0]
            loc_events['sb_y'] = locs[:, 1]

            # Normalize Period 2 (Flip X and Y)
            # Assumption: Team switches sides. If Period 1 was L->R, Period 2 is R->L.
            p2_mask = loc_events['period'] == 2
            loc_events.loc[p2_mask, 'sb_x'] = 120.0 - loc_events.loc[p2_mask, 'sb_x']
            loc_events.loc[p2_mask, 'sb_y'] = 80.0 - loc_events.loc[p2_mask, 'sb_y']

            # Now Transform to Metric (Centered)
            # X: 0..120 -> -52.5..52.5
            loc_events['x_metric'] = (loc_events['sb_x'] / 120.0 * PITCH_LENGTH) - (PITCH_LENGTH / 2)

            # Y: 0..80 -> 34..-34
            loc_events['y_metric'] = (PITCH_WIDTH / 2) - (loc_events['sb_y'] / 80.0 * PITCH_WIDTH)
            self.events = loc_events
            return True
            
        except Exception as e:
            print(f"Failed to fetch StatsBomb data: {e}")
            return False

    def diagnose_sparsity(self):
        """
        Diagnoses the sparsity challenge of the StatsBomb event data.
        """
        print("\n--- [StatsBomb] Sparsity Diagnosis ---")
        duration_min = 90
        n_events = len(self.events)
        density = n_events / duration_min
        
        print(f"Total Location Events: {n_events}")
        print(f"Temporal Density: {density:.2f} events/min (for whole team)")
        print(f"Per Player Density: ~{density/22:.2f} events/min")
        print(f"Comparison: {density/22:.2f} events/min per player vs 600 frames/min in Fitogether GPS.")

    def construct_pseudo_trajectories(self, window_min=5):
        """
        Implements the 'Pseudo-Trajectory Construction' adaptation.
        Aggregates events over windows to approximate formation structure.
        """
        print(f"\n--- [StatsBomb] Constructing Pseudo-Trajectories ({window_min} min windows) ---")
        df = self.events.copy()
        
        # Convert timestamp to minutes
        df['minute_bin'] = (df['minute'] // window_min) * window_min
        
        # Aggregate by Window + Player to get "Average Position" (Centroid)
        # This approximates the 'Mean Role Location' for that window
        pseudo_traj = df.groupby(['minute_bin', 'player'])[['x_metric', 'y_metric']].mean().reset_index()
        
        print(f"Generated {len(pseudo_traj)} pseudo-trajectory points (centroids).")
        return pseudo_traj
    

def get_stitched_tensor(df, n_slots=10):
    """
    Stitches disjoint player trajectories into 'n_slots' continuous tracks.
    This handles substitutions (e.g. Player A off, Player B on -> Slot 1).
    """
    # Identify temporal range for each player
    player_ranges = []
    for pid, group in df.groupby('player_id'):
        start = group['frame'].min()
        end = group['frame'].max()
        duration = len(group)
        player_ranges.append({
            'player_id': pid,
            'start': start,
            'end': end,
            'duration': duration,
            'data': group.set_index('frame')[['x', 'y']]
        })
    
    # Sort players by duration (longest playing time first) to identify core players
    player_ranges.sort(key=lambda x: x['duration'], reverse=True)
    
    # Assign players to slots
    # List of dicts: {'end_frame': int, 'data': series}
    slots = []
    
    for p in player_ranges:
        assigned = False
        # Try to fit into an existing slot (substitution)
        for s in slots:
            # Check if this player starts after the slot ends (plus a small buffer)
            if p['start'] > s['end_frame'] + 10: # 10 frames buffer
                s['parts'].append(p)
                s['end_frame'] = max(s['end_frame'], p['end'])
                assigned = True
                print(f"Stitched Player {p['player_id']} into Slot {s['id']}")
                break
        
        # If not assigned, create a new slot
        if not assigned:
            slots.append({
                'id': len(slots),
                'end_frame': p['end'],
                'parts': [p]
            })
            
    # Filter to top N slots
    if len(slots) > n_slots:
        print(f"Found {len(slots)} distinct tracks. Keeping top {n_slots} by duration.")
        for s in slots:
            s['total_duration'] = sum(p['duration'] for p in s['parts'])
        slots.sort(key=lambda x: x['total_duration'], reverse=True)
        slots = slots[:n_slots]
        
    # Construct the Tensor
    frames = sorted(df['frame'].unique())
    n_frames = len(frames)
    frame_map = {f: i for i, f in enumerate(frames)}
    
    X_tensor = np.full((n_frames, n_slots, 2), np.nan)
    
    for i, s in enumerate(slots):
        for part in s['parts']:
            indices = [frame_map[f] for f in part['data'].index if f in frame_map]
            X_tensor[indices, i, :] = part['data'].loc[list(part['data'].index)].values
            
    return X_tensor
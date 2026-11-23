
import pandas as pd

# Read the original dataset
df = pd.read_excel("V0001_AQA_dataset.xlsx")

# Features used as model inputs
features = [
    "detection_rate",
    "average_confidence",
    "low_confidence_percentage",
    "mean_valid_keypoint_percentage",
    "mean_valid_keypoint_confidence",

    "left_knee_angle_mean",
    "left_knee_angle_std",
    "right_knee_angle_mean",
    "right_knee_angle_std",

    "left_hip_angle_mean",
    "left_hip_angle_std",
    "right_hip_angle_mean",
    "right_hip_angle_std",

    "left_elbow_angle_mean",
    "left_elbow_angle_std",
    "right_elbow_angle_mean",
    "right_elbow_angle_std",

    "body_inclination_mean",
    "body_inclination_std",

    "body_center_speed_mean",
    "body_center_speed_std"
]

# Human quality prediction targets
targets = [
    "execution_quality",
    "balance_control",
    "body_position",
    "flow",
    "technical_execution"
]

# Check that all required columns exist
required = features + targets + ["video_id", "action_id", "action_type"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError("Missing columns: " + str(missing))

# Create X and y
X = df[features].copy()
y = df[["video_id", "action_id"] + targets].copy()

# Keep only rows with all five human quality ratings
valid_rows = y[targets].notna().all(axis=1)

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

# Combined dataset for inspection
modeling_data = pd.concat(
    [
        y[["video_id", "action_id"]],
        df.loc[valid_rows, ["action_type"]].reset_index(drop=True),
        X,
        y[targets]
    ],
    axis=1
)

# Save files
X.to_csv("V0001_X.csv", index=False)
y.to_csv("V0001_y.csv", index=False)
modeling_data.to_csv("V0001_modeling_dataset.csv", index=False)

print("Files created successfully!")
print("Number of actions:", len(X))
print("Number of input features:", len(features))
print("Number of target variables:", len(targets))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing feature values:", X.isna().sum().sum())

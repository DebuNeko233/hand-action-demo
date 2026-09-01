from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MAX_NUM_HANDS = 1
SEQUENCE_LENGTH = 30
INPUT_SIZE = 66
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 6
CLASS_NAMES = ["idle", "grab", "release", "move", "rotate", "wipe"]
CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 5
INFERENCE_STRIDE = 3
NO_HAND_RESET_FRAMES = 5
MODEL_PATH = ROOT_DIR / "models" / "hand_action_lstm.pth"
HAND_LANDMARKER_MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"

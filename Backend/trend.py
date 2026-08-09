from collections import deque

# Order matters: worse (wetter) → better (drier)
CONDITION_SEVERITY = {
    "dry": 0,
    "drying": 1,
    "damp": 2,
    "wet": 3,
}

# Keep last N readings in memory
HISTORY = deque(maxlen=10)

def add_reading(label: str):
    HISTORY.append(label)

def get_trend():
    if len(HISTORY) < 2:
        return {
            "direction": "not enough data",
            "suggestion": "Keep monitoring",
            "history": list(HISTORY)
        }

    first_half = [CONDITION_SEVERITY[l] for l in list(HISTORY)[:len(HISTORY)//2]]
    second_half = [CONDITION_SEVERITY[l] for l in list(HISTORY)[len(HISTORY)//2:]]

    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)

    if avg_second > avg_first + 0.3:
        direction = "getting wetter"
        suggestion = "Consider tire change soon — track is deteriorating"
    elif avg_second < avg_first - 0.3:
        direction = "getting drier"
        suggestion = "Conditions improving — hold current tires"
    else:
        direction = "stable"
        suggestion = "No change needed"

    return {
        "direction": direction,
        "suggestion": suggestion,
        "history": list(HISTORY)
    }
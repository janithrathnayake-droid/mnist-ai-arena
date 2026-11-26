import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ============================================================
# 1. PAGE CONFIGURATION AND CONSTANTS
# ============================================================
st.set_page_config(page_title="MNIST AI ARENA", page_icon="🏆", layout="wide")

TOTAL_ROUNDS = 10
MODES = ["AI Arena", "AI Jury", "AI Jury Battle"]
BENCHMARK_SIZE = 500
ELITE_SIZE = 500
VALIDATION_SIZE = 500
FINAL_TEST_SIZE = 1000
EXPERIMENT_SEED = 2026
BENCHMARK_SEED = EXPERIMENT_SEED
ELITE_SEED = EXPERIMENT_SEED + 1
WEIGHTED_SEED = EXPERIMENT_SEED + 2
ARENA_SEED = EXPERIMENT_SEED + 3
JURY_SAMPLE_SEED = EXPERIMENT_SEED + 4

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "mnist_models.joblib"
DATA_PATH = BASE_DIR / "data" / "mnist_test.npz"

# ============================================================
# 2. LOAD MODELS AND DATA
# ============================================================
@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    with np.load(DATA_PATH) as data:
        return data["X_test"], data["y_test"]


try:
    mnist_models = load_models()
    if not isinstance(mnist_models, dict) or not mnist_models:
        raise ValueError("The model bundle must be a non-empty dictionary.")
    st.sidebar.success(f"{len(mnist_models)} AI models loaded!")
    if len(mnist_models) != 10:
        st.sidebar.warning(f"Expected 10 models, but found {len(mnist_models)}.")
except Exception as exc:
    st.error(f"Model loading error: {exc}")
    st.stop()

try:
    X_test, y_test = load_dataset()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=int)
except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.stop()

if len(X_test) != len(y_test):
    st.error("Dataset error: X_test and y_test have different lengths.")
    st.stop()
if X_test.ndim != 2 or X_test.shape[1] != 784:
    st.error("Expected MNIST test images with shape (N, 784).")
    st.stop()
if len(X_test) == 0:
    st.error("The MNIST test dataset is empty.")
    st.stop()
if not np.all(np.isfinite(X_test)) or not np.all(np.isfinite(y_test)):
    st.error("Dataset error: the images or labels contain NaN/Inf values.")
    st.stop()
if np.any((y_test < 0) | (y_test > 9)):
    st.error("Dataset error: labels must be digits from 0 to 9.")
    st.stop()
if X_test.size and np.max(X_test) > 1.0:
    if np.max(X_test) <= 255.0 and np.min(X_test) >= 0:
        X_test = X_test / 255.0
    else:
        st.error("Dataset error: pixel values must be in [0, 1] or [0, 255].")
        st.stop()
if X_test.size and (np.min(X_test) < 0 or np.max(X_test) > 1.0):
    st.error("Dataset error: normalized pixel values must be between 0 and 1.")
    st.stop()

# ============================================================
# 3. MODE SELECTION AND TITLE
# ============================================================
if st.session_state.get("selected_mode") not in MODES:
    st.session_state.selected_mode = "AI Arena"

selected_mode = st.sidebar.radio(
    "Choose your experience:",
    MODES,
    index=MODES.index(st.session_state.selected_mode),
)
st.session_state.selected_mode = selected_mode

st.sidebar.title("🎮 Game Navigation")
st.sidebar.caption("Choose one of the three main experiences.")
st.title("🏆 MNIST AI ARENA")
st.write(
    "An interactive platform that explores how humans and machine-learning "
    "models recognize handwritten digits from the MNIST dataset."
)
with st.expander("ℹ️ Project overview – how to use MNIST AI ARENA", expanded=False):
    st.markdown("""
    **Project goal**  
    Compare human digit recognition with ten trained machine-learning models, then
    investigate how combining model predictions can improve or change decisions.

    **Three experiences**
    - **AI Arena:** compete against the models in a ten-round challenge.
    - **AI Jury:** compare individual models, equal voting, Elite voting and weighted voting
      using separate benchmark, validation and final-evaluation subsets.
    - **AI Jury Battle:** draw a digit, inspect model predictions and compare the three
      jury strategies. The intended digit is self-reported by the participant.

    **Important methodology note:** the experiments use disjoint subsets of the supplied
    10,000-image MNIST test file. The final evaluation subset is held out from model
    selection and weight calculation, but it is not an external dataset.
    """)
st.sidebar.markdown("---")
st.sidebar.caption("Current mode")
st.sidebar.write(f"**{selected_mode}**")

# ============================================================
# 4. SHARED HELPER FUNCTIONS
# ============================================================

def validate_prediction_vector(predictions, expected_length, model_name="Model"):
    """Validate a batch of digit predictions before scoring or voting."""
    values = np.asarray(predictions)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(f"{model_name} returned an invalid prediction shape/length.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{model_name} returned non-finite predictions.")
    if np.any((values < 0) | (values > 9)):
        raise ValueError(f"{model_name} returned predictions outside 0-9.")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{model_name} returned non-integer digit predictions.")
    return values.astype(int)


def json_safe(value):
    """Convert NumPy/Pandas values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def build_experiment_export():
    """Package current experiment settings, split indices and available results."""
    payload = {
        "project": "MNIST AI ARENA",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "random_seeds": {
            "benchmark": BENCHMARK_SEED,
            "elite_evaluation": ELITE_SEED,
            "weighted_split": WEIGHTED_SEED,
            "arena": ARENA_SEED,
            "jury_sample": JURY_SAMPLE_SEED,
        },
        "dataset": {
            "file": "data/mnist_test.npz",
            "total_images": int(len(X_test)),
            "feature_count": int(X_test.shape[1]),
            "benchmark_size": BENCHMARK_SIZE,
            "elite_size": ELITE_SIZE,
            "validation_size": VALIDATION_SIZE,
            "final_test_size": FINAL_TEST_SIZE,
            "note": "All subsets are disjoint portions of the same supplied MNIST test file.",
        },
        "models": list(mnist_models.keys()),
        "benchmark_indices": st.session_state.get("benchmark_indices", np.array([], dtype=int)),
        "elite_evaluation_indices": st.session_state.get("elite_evaluation_indices", np.array([], dtype=int)),
        "weighted_validation_indices": st.session_state.get("weighted_validation_indices", np.array([], dtype=int)),
        "weighted_final_test_indices": st.session_state.get("weighted_final_test_indices", np.array([], dtype=int)),
        "benchmark_results": st.session_state.get("benchmark_results"),
        "elite_evaluation": st.session_state.get("elite_evaluation"),
    }
    weighted = st.session_state.get("independent_weighted_results")
    if weighted:
        payload["weighted_experiment"] = {
            key: weighted.get(key) for key in [
                "weights", "original", "elite", "weighted", "elite_names",
                "original_names", "successful_models", "validation_size", "test_size",
                "final_test_indices",
            ]
        }
    return json_safe(payload)


def reset_competition():
    st.session_state.scores = {
        "Human": 0,
        **{name: 0 for name in mnist_models.keys()},
    }
    st.session_state.history = []
    st.session_state.round_indices = None
    st.session_state.current_round = 0
    st.session_state.phase = "ready"
    st.session_state.round_result = None


def create_leaderboard():
    completed = len(st.session_state.history)
    rows = []
    for name, score in st.session_state.scores.items():
        accuracy = score / completed * 100 if completed else 0
        rows.append({
            "Participant": name,
            "Score": score,
            "Accuracy": f"{accuracy:.1f}%",
        })
    rows.sort(key=lambda item: item["Score"], reverse=True)
    frame = pd.DataFrame(rows)
    frame.insert(0, "Rank", range(1, len(frame) + 1))
    return frame


def evaluate_equal_jury(predictions_dict, model_names, actual):
    """Plurality voting. Ties count as incorrect in overall accuracy."""
    actual = np.asarray(actual, dtype=int)
    if actual.ndim != 1:
        raise ValueError("Actual labels must be a one-dimensional array.")
    if np.any((actual < 0) | (actual > 9)):
        raise ValueError("Actual labels must be digits from 0 to 9.")
    if not model_names:
        raise ValueError("At least one model is required for jury evaluation.")
    for name in model_names:
        if name not in predictions_dict:
            raise KeyError(f"Predictions are missing for model: {name}")
        predictions_dict[name] = validate_prediction_vector(
            predictions_dict[name], len(actual), name
        )

    correct = 0
    ties = 0
    for i, true_digit in enumerate(actual):
        votes = np.zeros(10, dtype=int)
        for name in model_names:
            digit = int(predictions_dict[name][i])
            if digit < 0 or digit > 9:
                raise ValueError(f"Invalid digit prediction from {name}: {digit}")
            votes[digit] += 1
        winners = np.flatnonzero(votes == votes.max())
        if len(winners) == 1:
            correct += int(winners[0] == int(true_digit))
        else:
            ties += 1

    total = len(actual)
    decided = total - ties
    return {
        "correct": correct,
        "accuracy": correct / total * 100 if total else 0.0,
        "coverage": decided / total * 100 if total else 0.0,
        "ties": ties,
        "decided_accuracy": correct / decided * 100 if decided else 0.0,
        "total": total,
    }


def calculate_elite_predictions(results):
    """Reconstruct equal-vote Elite Jury predictions; -1 means tie."""
    actual = np.asarray(results["final_y"], dtype=int)
    predictions = results["final_predictions"]
    elite_names = results["elite_names"]
    output = np.full(len(actual), -1, dtype=int)
    for i in range(len(actual)):
        votes = np.zeros(10, dtype=int)
        for name in elite_names:
            votes[int(predictions[name][i])] += 1
        winners = np.flatnonzero(votes == votes.max())
        if len(winners) == 1:
            output[i] = int(winners[0])
    return output


def preprocess_handwritten_canvas(canvas_rgba):
    if canvas_rgba is None:
        return None, None
    canvas_rgba = np.asarray(canvas_rgba)
    if canvas_rgba.ndim != 3 or canvas_rgba.shape[2] not in (3, 4):
        raise ValueError("Canvas image must have 3 (RGB) or 4 (RGBA) channels.")
    if not np.all(np.isfinite(canvas_rgba)):
        raise ValueError("Canvas image contains invalid pixel values.")
    if canvas_rgba.shape[2] == 3:
        alpha = np.full((*canvas_rgba.shape[:2], 1), 255, dtype=np.uint8)
        canvas_rgba = np.concatenate([canvas_rgba, alpha], axis=2)
    original_pil = Image.fromarray(
        np.clip(canvas_rgba, 0, 255).astype(np.uint8), mode="RGBA"
    ).convert("L")
    original_array = np.asarray(original_pil, dtype=np.uint8)
    coords = np.argwhere(original_array > 30)
    if len(coords) == 0:
        return original_array, None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = original_array[y_min:y_max + 1, x_min:x_max + 1]

    digit_image = Image.fromarray(cropped)
    digit_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    centered = Image.new("L", (28, 28), 0)
    x_offset = (28 - digit_image.width) // 2
    y_offset = (28 - digit_image.height) // 2
    centered.paste(digit_image, (x_offset, y_offset))

    pixels = np.asarray(centered, dtype=np.float32)
    total_intensity = pixels.sum()
    if total_intensity == 0:
        return original_array, None

    yy, xx = np.indices(pixels.shape)
    center_x = (xx * pixels).sum() / total_intensity
    center_y = (yy * pixels).sum() / total_intensity
    shift_x = round(13.5 - center_x)
    shift_y = round(13.5 - center_y)

    processed = centered.transform(
        (28, 28),
        Image.Transform.AFFINE,
        (1, 0, -shift_x, 0, 1, -shift_y),
        resample=Image.Resampling.BICUBIC,
    )
    return original_array, np.asarray(processed, dtype=np.float32) / 255.0


def predict_models(model_input):
    """Predict independently; a probability error does not discard prediction."""
    predictions, details, failed = {}, {}, []
    for name, model in mnist_models.items():
        try:
            prediction = int(model.predict(model_input)[0])
            if prediction < 0 or prediction > 9:
                raise ValueError("Model returned a value outside 0-9.")
            predictions[name] = prediction
        except Exception:
            failed.append(name)
            continue

        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(model_input)[0]
                classes = np.asarray(model.classes_)
                matching = np.flatnonzero(classes == prediction)
                if len(matching):
                    confidence = float(probabilities[matching[0]]) * 100
            except Exception:
                confidence = None
        details[name] = {
            "prediction": prediction,
            "confidence": confidence,
        }
    return predictions, details, failed


def jury_decision(votes):
    votes = np.asarray(votes, dtype=float)
    if votes.size == 0 or not np.all(np.isfinite(votes)):
        return -1
    winners = np.flatnonzero(np.isclose(votes, np.max(votes)))
    return int(winners[0]) if len(winners) == 1 else -1


def select_elite_models(benchmark_individual, count=4):
    """Top individual benchmark accuracies; original model order breaks ties."""
    ranked = [
        (position, item)
        for position, item in enumerate(benchmark_individual)
        if isinstance(item.get("Accuracy"), (int, float))
        and np.isfinite(item["Accuracy"])
    ]
    ranked.sort(key=lambda pair: (-pair[1]["Accuracy"], pair[0]))
    return [item["Model"] for _, item in ranked[:count]]


def make_weighted_predictions(final_predictions, elite_names, weights, sample_count):
    """Return weighted decisions; -1 means tied maximum weighted vote."""
    if len(elite_names) != 4:
        raise ValueError("Exactly four Elite models are required.")
    for name in elite_names:
        if name not in final_predictions or name not in weights:
            raise ValueError(f"Missing final predictions or weight for {name}.")
        if not np.isfinite(weights[name]) or weights[name] < 0:
            raise ValueError(f"Invalid voting weight for {name}.")
        final_predictions[name] = validate_prediction_vector(
            final_predictions[name], sample_count, name
        )

    output = np.full(sample_count, -1, dtype=int)
    for i in range(sample_count):
        digit_weights = np.zeros(10, dtype=float)
        for name in elite_names:
            digit = int(final_predictions[name][i])
            if digit < 0 or digit > 9:
                raise ValueError(f"Invalid prediction from {name}: {digit}")
            digit_weights[digit] += float(weights[name])
        output[i] = jury_decision(digit_weights)
    return output


def evaluate_prediction_array(predictions, actual):
    actual = np.asarray(actual, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if predictions.ndim != 1 or len(predictions) != len(actual):
        raise ValueError("Prediction and label arrays have different lengths or dimensions.")
    if np.any((predictions < -1) | (predictions > 9)):
        raise ValueError("Predictions must be digits 0-9 or -1 for a tie.")
    if np.any((actual < 0) | (actual > 9)):
        raise ValueError("Actual labels must be digits from 0 to 9.")
    ties = predictions == -1
    correct = (predictions == actual) & ~ties
    total = len(actual)
    decided = int(np.sum(~ties))
    correct_count = int(np.sum(correct))
    tie_count = int(np.sum(ties))
    return {
        "correct": correct_count,
        "accuracy": correct_count / total * 100 if total else 0.0,
        "coverage": decided / total * 100 if total else 0.0,
        "ties": tie_count,
        "decided_accuracy": correct_count / decided * 100 if decided else 0.0,
        "total": total,
    }


def save_battle_result(result, actual, history):
    strategy_predictions = {
        "Original Jury": result["original_decision"],
        "Elite Jury": result["elite_decision"],
        "Weighted Jury": result["weighted_decision"],
    }
    strategy_results = {}
    for name, prediction in strategy_predictions.items():
        if prediction == -1:
            outcome = "Tie"
        elif prediction == actual:
            outcome = "Correct"
        else:
            outcome = "Incorrect"
        strategy_results[name] = {
            "prediction": int(prediction),
            "result": outcome,
        }

    model_predictions = result["model_predictions"]
    correct_models = sum(int(p) == actual for p in model_predictions.values())
    total_models = len(model_predictions)
    history.append({
        "Battle": len(history) + 1,
        "Actual Digit": int(actual),
        "Correct Models": int(correct_models),
        "Total Models": int(total_models),
        "Individual Accuracy (%)": round(
            correct_models / total_models * 100, 2
        ) if total_models else 0.0,
        "strategies": strategy_results,
    })


def create_battle_leaderboard(history):
    strategies = ["Original Jury", "Elite Jury", "Weighted Jury"]
    rows = []
    for strategy in strategies:
        correct = sum(b["strategies"][strategy]["result"] == "Correct" for b in history)
        ties = sum(b["strategies"][strategy]["result"] == "Tie" for b in history)
        total = len(history)
        decided = total - ties
        rows.append({
            "Jury Strategy": strategy,
            "Correct": correct,
            "Incorrect": total - correct - ties,
            "Ties": ties,
            "Accuracy (%)": round(correct / total * 100, 2) if total else 0.0,
            "Decision Coverage (%)": round(decided / total * 100, 2) if total else 0.0,
            "Decided Accuracy (%)": round(correct / decided * 100, 2) if decided else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        "Accuracy (%)", ascending=False
    ).reset_index(drop=True)


def clear_current_battle():
    st.session_state.battle_result = None
    st.session_state.battle_image = None
    st.session_state.battle_original_image = None
    st.session_state.battle_saved = False
    st.session_state.battle_canvas_key += 1
    st.session_state.pop("battle_revealed", None)
    st.session_state.pop("battle_actual_digit", None)


# ============================================================
# 5. INITIALIZE SESSION STATE
# ============================================================
if "phase" not in st.session_state:
    reset_competition()

if "battle_history" not in st.session_state:
    st.session_state.battle_history = []
if "battle_saved" not in st.session_state:
    st.session_state.battle_saved = False

# ============================================================
# 6. AI ARENA MODE
# ============================================================
if st.session_state.selected_mode == "AI Arena":
    st.subheader("⚔️ Human vs 10 AI Models")
    st.write(
        "Can you beat ten machine-learning models in a "
        "ten-round handwritten digit recognition challenge?"
    )
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Competition rounds", TOTAL_ROUNDS)
    c2.metric("AI models", len(mnist_models))
    c3.metric("Maximum score", TOTAL_ROUNDS)
    st.info("Correct prediction = 1 point. Incorrect prediction = 0 points.")
    st.caption(f"Round selection seed: {ARENA_SEED} (fixed for reproducible demonstrations).")

    if st.session_state.phase != "ready":
        completed = len(st.session_state.history)
        st.subheader("📊 Live Competition Progress")
        st.progress(completed / TOTAL_ROUNDS)
        st.write(f"Completed rounds: {completed} / {TOTAL_ROUNDS}")
        st.dataframe(create_leaderboard(), hide_index=True, use_container_width=True)

    if st.session_state.phase == "ready":
        st.write(
            "You will compete against all ten AI models over "
            "ten different handwritten digits."
        )
        if st.button("🚀 Start 10-Round Competition", type="primary"):
            st.session_state.round_indices = np.random.default_rng(ARENA_SEED).choice(
                len(X_test), size=TOTAL_ROUNDS, replace=False
            )
            st.session_state.current_round = 0
            st.session_state.phase = "playing"
            st.rerun()

    if st.session_state.phase == "playing":
        round_no = st.session_state.current_round + 1
        index = st.session_state.round_indices[st.session_state.current_round]
        st.divider()
        st.subheader(f"✍️ Round {round_no} / {TOTAL_ROUNDS}")
        st.progress(round_no / TOTAL_ROUNDS)
        st.write("Identify the handwritten digit.")
        st.image(X_test[index].reshape(28, 28), width=280, clamp=True)
        human_guess = st.selectbox(
            "Your prediction", list(range(10)), key=f"human_guess_{round_no}"
        )
        if st.button("🔒 Lock My Answer", type="primary"):
            st.session_state.human_guess = int(human_guess)
            st.session_state.phase = "locked"
            st.rerun()

    if st.session_state.phase == "locked":
        st.subheader("🤖 AI Challenge")
        st.info(f"Your answer is locked: {st.session_state.human_guess}")
        if st.button("Reveal AI Predictions", type="primary"):
            round_no = st.session_state.current_round + 1
            index = st.session_state.round_indices[st.session_state.current_round]
            actual = int(y_test[index])
            model_input = X_test[index].reshape(1, -1)
            human_guess = int(st.session_state.human_guess)
            human_correct = human_guess == actual
            if human_correct:
                st.session_state.scores["Human"] += 1

            predictions = []
            for name, model in mnist_models.items():
                try:
                    prediction = int(model.predict(model_input)[0])
                    correct = prediction == actual
                    if correct:
                        st.session_state.scores[name] += 1
                    predictions.append({
                        "Model": name,
                        "Prediction": prediction,
                        "Result": "Correct" if correct else "Incorrect",
                    })
                except Exception:
                    predictions.append({
                        "Model": name,
                        "Prediction": "Error",
                        "Result": "Not evaluated",
                    })

            round_result = {
                "Round": round_no,
                "Actual": actual,
                "Human": human_guess,
                "Human_Result": "Correct" if human_correct else "Incorrect",
                "Predictions": predictions,
            }
            st.session_state.history.append(round_result)
            st.session_state.round_result = round_result
            st.session_state.phase = "results"
            st.rerun()

    if st.session_state.phase == "results":
        result = st.session_state.round_result
        st.divider()
        st.subheader(f"🏆 Round {result['Round']} Results")
        c1, c2 = st.columns(2)
        c1.metric("Actual digit", result["Actual"])
        c2.metric("Your prediction", result["Human"])
        if result["Human_Result"] == "Correct":
            st.success("Your prediction is correct! +1 point")
        else:
            st.error("Your prediction is incorrect. No point this round.")

        st.subheader("AI Model Predictions")
        st.dataframe(pd.DataFrame(result["Predictions"]), hide_index=True, use_container_width=True)
        correct_models = sum(item["Result"] == "Correct" for item in result["Predictions"])
        st.metric("AI models correct this round", f"{correct_models}/{len(mnist_models)}")
        st.subheader("📊 Updated Leaderboard")
        st.dataframe(create_leaderboard(), hide_index=True, use_container_width=True)

        if result["Round"] < TOTAL_ROUNDS:
            if st.button("➡️ Next Round", type="primary"):
                st.session_state.current_round += 1
                st.session_state.phase = "playing"
                st.rerun()
        elif st.button("🏆 View Final Competition Results", type="primary"):
            st.session_state.phase = "finished"
            st.rerun()

    if st.session_state.phase == "finished":
        st.divider()
        st.title("🏆 Competition Completed!")
        st.balloons()
        leaderboard = create_leaderboard()
        st.subheader("Final Leaderboard")
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)
        st.success(f"Competition leader: {leaderboard.iloc[0]['Participant']}")
        st.subheader("Final Scores")
        st.bar_chart(pd.DataFrame({"Score": st.session_state.scores}))
        st.subheader("Round-by-Round Human Performance")
        history_df = pd.DataFrame([
            {
                "Round": item["Round"],
                "Actual Digit": item["Actual"],
                "Human Prediction": item["Human"],
                "Result": item["Human_Result"],
            }
            for item in st.session_state.history
        ])
        st.dataframe(history_df, hide_index=True, use_container_width=True)
        if st.button("🔄 Start New Competition", type="primary"):
            reset_competition()
            st.rerun()

# ============================================================
# 7. AI JURY MODE
# ============================================================
elif st.session_state.selected_mode == "AI Jury":
    st.subheader("⚖️ AI Jury - Majority Voting")
    st.caption(f"Reproducible experiment seeds are enabled (base seed: {EXPERIMENT_SEED}).")
    st.write(
        "Ten AI models independently predict a handwritten digit. "
        "The digit receiving the most votes becomes the jury decision."
    )

    if "jury_index" not in st.session_state:
        st.session_state.jury_index = None
    if "jury_result" not in st.session_state:
        st.session_state.jury_result = None

    if st.button("🎲 Generate Handwritten Digit"):
        st.session_state.jury_index = int(np.random.default_rng(JURY_SAMPLE_SEED).integers(0, len(X_test)))
        st.session_state.jury_result = None
        st.rerun()

    if st.session_state.jury_index is not None:
        index = st.session_state.jury_index
        st.divider()
        st.subheader("🖼️ Image for the Jury")
        st.image(X_test[index].reshape(28, 28), width=280, clamp=True)
        if st.button("⚖️ Call the AI Jury", type="primary"):
            model_predictions = {}
            predictions = []
            for name, model in mnist_models.items():
                try:
                    prediction = int(model.predict(X_test[index].reshape(1, -1))[0])
                    model_predictions[name] = prediction
                    predictions.append({"Model": name, "Vote": prediction})
                except Exception:
                    predictions.append({"Model": name, "Vote": None})

            valid_votes = list(model_predictions.values())
            if valid_votes:
                vote_counts = np.bincount(valid_votes, minlength=10)
                winners = np.flatnonzero(vote_counts == vote_counts.max())
                st.session_state.jury_result = {
                    "predictions": predictions,
                    "vote_counts": vote_counts,
                    "winning_digits": winners.tolist(),
                    "actual": int(y_test[index]),
                }
            else:
                st.session_state.jury_result = {
                    "error": "All models failed to make predictions."
                }
            st.rerun()

    if st.session_state.jury_result is not None:
        result = st.session_state.jury_result
        if "error" in result:
            st.error(result["error"])
        else:
            st.divider()
            st.subheader("🗳️ Individual Model Votes")
            st.dataframe(pd.DataFrame(result["predictions"]), hide_index=True, use_container_width=True)
            st.subheader("📊 Vote Distribution")
            vote_df = pd.DataFrame({"Digit": range(10), "Votes": result["vote_counts"]})
            st.bar_chart(vote_df.set_index("Digit"))
            st.subheader("⚖️ Final Jury Decision")
            winners = result["winning_digits"]
            if len(winners) == 1:
                winner = winners[0]
                st.success(f"The jury predicts digit: {winner}")
                if winner == result["actual"]:
                    st.success("The jury made the correct decision!")
                else:
                    st.error(f"The actual digit was {result['actual']}.")
            else:
                st.warning(f"The jury is tied between digits: {winners}")
                st.write(f"Actual digit: {result['actual']}")
            evaluated_count = sum(p["Vote"] is not None for p in result["predictions"])
            st.metric("Maximum votes received", f"{max(result['vote_counts'])}/{evaluated_count}")

    # ORIGINAL BENCHMARK
    st.divider()
    st.subheader("🏆 AI Jury Performance Benchmark")
    st.write(
        f"Evaluate all models and the collective jury using "
        f"{BENCHMARK_SIZE} randomly selected MNIST test images."
    )

    if st.button("🚀 Run 500-Image Benchmark", type="primary"):
        if len(X_test) < BENCHMARK_SIZE:
            st.error("The test dataset contains fewer than 500 images.")
        else:
            indices = np.random.default_rng(BENCHMARK_SEED).choice(len(X_test), size=BENCHMARK_SIZE, replace=False)
            st.session_state.benchmark_indices = indices.copy()
            st.session_state.benchmark_seed = BENCHMARK_SEED
            for key in [
                "benchmark_results", "elite_evaluation", "elite_evaluation_indices",
                "independent_weighted_results", "weighted_validation_indices",
                "weighted_final_test_indices",
            ]:
                st.session_state.pop(key, None)

            benchmark_X, benchmark_y = X_test[indices], y_test[indices]
            individual, prediction_dict = [], {}
            progress, status = st.progress(0), st.empty()
            for i, (name, model) in enumerate(mnist_models.items()):
                status.write(f"Evaluating {name}...")
                try:
                    predictions = validate_prediction_vector(
                        model.predict(benchmark_X), BENCHMARK_SIZE, name
                    )
                    prediction_dict[name] = predictions
                    correct = int(np.sum(predictions == benchmark_y))
                    individual.append({
                        "Model": name, "Correct": correct,
                        "Incorrect": BENCHMARK_SIZE - correct,
                        "Accuracy": round(correct / BENCHMARK_SIZE * 100, 2),
                    })
                except Exception:
                    individual.append({
                        "Model": name, "Correct": "Error",
                        "Incorrect": "Error", "Accuracy": "Error",
                    })
                progress.progress((i + 1) / len(mnist_models))
            progress.empty()
            status.empty()

            if not prediction_dict:
                st.error("No models completed the benchmark.")
            else:
                metrics = evaluate_equal_jury(
                    prediction_dict, list(prediction_dict), benchmark_y
                )
                st.session_state.benchmark_results = {
                    "individual": individual,
                    "jury_accuracy": metrics["accuracy"],
                    "decided_accuracy": metrics["decided_accuracy"],
                    "coverage": metrics["coverage"],
                    "ties": metrics["ties"],
                    "successful_models": len(prediction_dict),
                    "sample_size": BENCHMARK_SIZE,
                }
                st.success("500-image benchmark completed!")

    if "benchmark_results" in st.session_state:
        benchmark = st.session_state.benchmark_results
        st.subheader("📊 Individual Model Performance")
        st.dataframe(pd.DataFrame(benchmark["individual"]), hide_index=True, use_container_width=True)
        st.subheader("⚖️ AI Jury Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jury accuracy", f"{benchmark['jury_accuracy']:.2f}%")
        c2.metric("Decision coverage", f"{benchmark['coverage']:.2f}%")
        c3.metric("Tied decisions", benchmark["ties"])
        st.metric("Accuracy when jury reaches a unique decision", f"{benchmark['decided_accuracy']:.2f}%")
        st.write(f"Successfully evaluated models: {benchmark['successful_models']}/{len(mnist_models)}")
        st.caption(
            "Jury accuracy counts tied decisions as incorrect. Decision coverage "
            "is the percentage of images with a unique winning digit. "
            "Unique-decision accuracy excludes ties."
        )
        comparison_data = {
            item["Model"]: item["Accuracy"]
            for item in benchmark["individual"]
            if isinstance(item["Accuracy"], (int, float))
        }
        comparison_data["AI Jury"] = benchmark["jury_accuracy"]
        st.subheader("📈 Individual Models vs AI Jury")
        st.bar_chart(pd.DataFrame.from_dict(
            comparison_data, orient="index", columns=["Accuracy (%)"]
        ))

    # ELITE JURY EXPERIMENT
    st.divider()
    st.subheader("👑 Elite Jury Experiment")
    st.write(
        "The four strongest models from the original benchmark form a new jury, "
        f"evaluated on {ELITE_SIZE} fresh images."
    )

    if "benchmark_results" not in st.session_state or "benchmark_indices" not in st.session_state:
        st.info("First run the original benchmark to identify the strongest models.")
    else:
        original_results = st.session_state.benchmark_results
        elite_names = select_elite_models(original_results["individual"], count=4)
        st.caption(
            "Selection criterion: the four highest individual accuracies on the "
            "original benchmark. If two models have equal accuracy, their original "
            "model order is used as the tie-break."
        )
        st.write("**Selected Elite Jury members:**")
        accuracy_lookup = {
            item["Model"]: item["Accuracy"]
            for item in original_results["individual"]
            if isinstance(item["Accuracy"], (int, float))
        }
        for rank, name in enumerate(elite_names, start=1):
            st.write(f"{rank}. {name} — {accuracy_lookup[name]:.2f}%")

        if len(elite_names) < 4:
            st.warning("Fewer than four models completed the benchmark.")
        elif st.button("👑 Evaluate Elite Jury on 500 Fresh Images", type="primary"):
            used = np.asarray(st.session_state.benchmark_indices, dtype=int)
            available = np.setdiff1d(np.arange(len(X_test)), used)
            if len(available) < ELITE_SIZE:
                st.error("Not enough unused images for Elite Jury evaluation.")
            else:
                indices = np.random.default_rng(ELITE_SEED).choice(available, size=ELITE_SIZE, replace=False)
                st.session_state.elite_evaluation_indices = indices.copy()
                st.session_state.elite_seed = ELITE_SEED
                for key in [
                    "elite_evaluation", "independent_weighted_results",
                    "weighted_validation_indices", "weighted_final_test_indices",
                ]:
                    st.session_state.pop(key, None)

                evaluation_X, evaluation_y = X_test[indices], y_test[indices]
                predictions_dict, individual = {}, []
                progress, status = st.progress(0), st.empty()
                for i, (name, model) in enumerate(mnist_models.items()):
                    status.write(f"Evaluating {name}...")
                    try:
                        predictions = validate_prediction_vector(
                            model.predict(evaluation_X), ELITE_SIZE, name
                        )
                        predictions_dict[name] = predictions
                        correct = int(np.sum(predictions == evaluation_y))
                        individual.append({
                            "Model": name, "Correct": correct,
                            "Accuracy": round(correct / ELITE_SIZE * 100, 2),
                        })
                    except Exception:
                        individual.append({
                            "Model": name, "Correct": "Error", "Accuracy": "Error"
                        })
                    progress.progress((i + 1) / len(mnist_models))
                progress.empty()
                status.empty()

                if not all(name in predictions_dict for name in elite_names):
                    st.error("All four Elite Jury models must be successfully evaluated.")
                else:
                    full_metrics = evaluate_equal_jury(
                        predictions_dict, list(predictions_dict), evaluation_y
                    )
                    elite_metrics = evaluate_equal_jury(
                        predictions_dict, elite_names, evaluation_y
                    )
                    st.session_state.elite_evaluation = {
                        "elite_names": elite_names,
                        "individual": individual,
                        "full_jury": full_metrics,
                        "elite_jury": elite_metrics,
                    }
                    st.success("Elite Jury evaluation completed!")

    if "elite_evaluation" in st.session_state:
        experiment = st.session_state.elite_evaluation
        st.subheader("📊 Fresh 500-Image Results")
        st.write("Individual models and both juries were evaluated using the same fresh 500 images.")
        st.dataframe(pd.DataFrame(experiment["individual"]), hide_index=True, use_container_width=True)
        comparison = pd.DataFrame([
            {
                "Jury": "Original 10-Model Jury",
                "Accuracy": round(experiment["full_jury"]["accuracy"], 2),
                "Coverage": round(experiment["full_jury"]["coverage"], 2),
                "Ties": experiment["full_jury"]["ties"],
            },
            {
                "Jury": "Elite 4-Model Jury",
                "Accuracy": round(experiment["elite_jury"]["accuracy"], 2),
                "Coverage": round(experiment["elite_jury"]["coverage"], 2),
                "Ties": experiment["elite_jury"]["ties"],
            },
        ])
        st.subheader("⚖️ Original Jury vs Elite Jury")
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Original Jury Accuracy", f"{experiment['full_jury']['accuracy']:.2f}%")
        c2.metric("Elite Jury Accuracy", f"{experiment['elite_jury']['accuracy']:.2f}%")
        st.metric("Elite Jury Unique-Decision Accuracy", f"{experiment['elite_jury']['decided_accuracy']:.2f}%")
        chart_data = {
            item["Model"]: item["Accuracy"]
            for item in experiment["individual"]
            if isinstance(item["Accuracy"], (int, float))
        }
        chart_data["Original Jury"] = experiment["full_jury"]["accuracy"]
        chart_data["Elite Jury"] = experiment["elite_jury"]["accuracy"]
        st.subheader("📈 Accuracy Comparison")
        st.bar_chart(pd.DataFrame.from_dict(chart_data, orient="index", columns=["Accuracy (%)"]))
        st.caption("Tied decisions count as incorrect for overall accuracy. Unique-decision accuracy excludes ties.")

    # INDEPENDENT WEIGHTED JURY
    st.divider()
    st.subheader("⚖️ Weighted Jury - Independent Evaluation")
    st.write(
        "Voting weights are calculated using a separate validation dataset. "
        "Final performance is measured using held-out test images not used "
        "for model selection or weight calculation."
    )
    prerequisites = all(key in st.session_state for key in [
        "benchmark_results", "benchmark_indices",
        "elite_evaluation", "elite_evaluation_indices",
    ])

    if not prerequisites:
        st.info("Complete the original benchmark and Elite Jury experiment first.")
    elif st.button("🚀 Run Independent Weighted Jury Experiment", type="primary"):
        elite_names = st.session_state.elite_evaluation["elite_names"]
        benchmark_indices = np.asarray(st.session_state.benchmark_indices, dtype=int)
        elite_indices = np.asarray(st.session_state.elite_evaluation_indices, dtype=int)

        # Task 10: explicit data separation checks.
        if np.intersect1d(benchmark_indices, elite_indices).size:
            st.error("Data leakage detected: benchmark and Elite evaluation sets overlap.")
        else:
            used = np.union1d(benchmark_indices, elite_indices)
            remaining = np.setdiff1d(np.arange(len(X_test)), used)
            required = VALIDATION_SIZE + FINAL_TEST_SIZE

            if len(remaining) < required:
                st.error("Not enough unused test images for validation and final evaluation.")
            else:
                rng = np.random.default_rng(WEIGHTED_SEED)
                new_indices = rng.choice(remaining, size=required, replace=False)
                validation_indices = new_indices[:VALIDATION_SIZE]
                final_indices = new_indices[VALIDATION_SIZE:]

                split_overlaps = (
                    np.intersect1d(benchmark_indices, validation_indices).size
                    or np.intersect1d(benchmark_indices, final_indices).size
                    or np.intersect1d(elite_indices, validation_indices).size
                    or np.intersect1d(elite_indices, final_indices).size
                    or np.intersect1d(validation_indices, final_indices).size
                )
                if split_overlaps:
                    st.error("Data leakage detected: one or more experiment splits overlap.")
                else:
                    st.session_state.weighted_validation_indices = validation_indices.copy()
                    st.session_state.weighted_final_test_indices = final_indices.copy()
                    st.session_state.weighted_seed = WEIGHTED_SEED
                    validation_X, validation_y = X_test[validation_indices], y_test[validation_indices]
                    final_X, final_y = X_test[final_indices], y_test[final_indices]

                    st.write("### Stage 1: Calculate Voting Weights")
                    weights, failed = {}, []
                    progress, status = st.progress(0), st.empty()
                    for i, name in enumerate(elite_names):
                        status.write(f"Validating {name}...")
                        try:
                            pred = validate_prediction_vector(
                                mnist_models[name].predict(validation_X), VALIDATION_SIZE, name
                            )
                            weights[name] = float(np.mean(pred == validation_y) * 100)
                        except Exception:
                            failed.append(name)
                        progress.progress((i + 1) / len(elite_names))
                    progress.empty()
                    status.empty()

                    if failed or len(weights) != 4:
                        st.error(
                            "All four Elite models must complete validation. Failed: "
                            + ", ".join(failed)
                        )
                    else:
                        st.write("### Stage 2: Independent Final Evaluation")
                        final_predictions = {}
                        progress, status = st.progress(0), st.empty()
                        for i, (name, model) in enumerate(mnist_models.items()):
                            status.write(f"Testing {name}...")
                            try:
                                pred = validate_prediction_vector(
                                    model.predict(final_X), FINAL_TEST_SIZE, name
                                )
                                final_predictions[name] = pred
                            except Exception:
                                pass
                            progress.progress((i + 1) / len(mnist_models))
                        progress.empty()
                        status.empty()

                        available_elite = [name for name in elite_names if name in final_predictions]
                        if len(available_elite) != 4:
                            st.error("All four Elite models must predict the final test images.")
                        elif not final_predictions:
                            st.error("No models completed final evaluation.")
                        else:
                            original_names = list(final_predictions)
                            original_metrics = evaluate_equal_jury(
                                final_predictions, original_names, final_y
                            )
                            elite_metrics = evaluate_equal_jury(
                                final_predictions, available_elite, final_y
                            )
                            try:
                                weighted_predictions = make_weighted_predictions(
                                    final_predictions, available_elite, weights, len(final_y)
                                )
                                weighted_metrics = evaluate_prediction_array(
                                    weighted_predictions, final_y
                                )
                            except Exception as exc:
                                st.error(f"Weighted voting error: {exc}")
                                weighted_predictions = None
                                weighted_metrics = None

                            if weighted_predictions is not None:
                                st.session_state.independent_weighted_results = {
                                    "weights": weights.copy(),
                                    "original": original_metrics,
                                    "elite": elite_metrics,
                                    "weighted": weighted_metrics,
                                    "final_predictions": {
                                        name: pred.copy()
                                        for name, pred in final_predictions.items()
                                    },
                                    "final_y": final_y.copy(),
                                    "weighted_predictions": weighted_predictions.copy(),
                                    "elite_names": available_elite.copy(),
                                    "original_names": original_names.copy(),
                                    "final_test_indices": final_indices.copy(),
                                    "successful_models": len(final_predictions),
                                    "validation_size": VALIDATION_SIZE,
                                    "test_size": len(final_y),
                                }
                                st.success("Independent Weighted Jury experiment completed!")

    # DISPLAY WEIGHTED RESULTS
    if "independent_weighted_results" in st.session_state:
        results = st.session_state.independent_weighted_results
        st.divider()
        st.subheader("📊 Stage 1: Validation Voting Weights")
        weights_df = pd.DataFrame([
            {"Model": name, "Validation Accuracy (%)": round(weight, 2)}
            for name, weight in results["weights"].items()
        ])
        st.dataframe(weights_df, hide_index=True, use_container_width=True)
        st.caption("Weights were calculated using the separate 500-image validation dataset.")

        st.subheader("🏆 Stage 2: Final Test Results")
        st.write(
            f"All jury systems were evaluated on the same {results['test_size']:,} "
            "held-out images. These images were not used for Elite model selection "
            "or voting-weight calculation."
        )
        comparison = pd.DataFrame([
            {
                "Jury": "Original 10-Model Jury",
                "Accuracy (%)": round(results["original"]["accuracy"], 2),
                "Coverage (%)": round(results["original"]["coverage"], 2),
                "Ties": results["original"]["ties"],
            },
            {
                "Jury": "Elite 4-Model Jury",
                "Accuracy (%)": round(results["elite"]["accuracy"], 2),
                "Coverage (%)": round(results["elite"]["coverage"], 2),
                "Ties": results["elite"]["ties"],
            },
            {
                "Jury": "Weighted 4-Model Jury",
                "Accuracy (%)": round(results["weighted"]["accuracy"], 2),
                "Coverage (%)": round(results["weighted"]["coverage"], 2),
                "Ties": results["weighted"]["ties"],
            },
        ])
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Jury", f"{results['original']['accuracy']:.2f}%")
        c2.metric("Elite Jury", f"{results['elite']['accuracy']:.2f}%")
        c3.metric("Weighted Jury", f"{results['weighted']['accuracy']:.2f}%")
        st.subheader("📈 Final Accuracy Comparison")
        st.bar_chart(pd.DataFrame({
            "Accuracy (%)": [
                results["original"]["accuracy"],
                results["elite"]["accuracy"],
                results["weighted"]["accuracy"],
            ]
        }, index=["Original Jury", "Elite Jury", "Weighted Jury"]))
        st.subheader("Additional Weighted Jury Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision Coverage", f"{results['weighted']['coverage']:.2f}%")
        c2.metric("Tied Decisions", results["weighted"]["ties"])
        c3.metric("Unique-Decision Accuracy", f"{results['weighted']['decided_accuracy']:.2f}%")
        st.caption("Overall accuracy counts tied decisions as incorrect. Unique-decision accuracy excludes ties.")
        st.info(
            f"Final evaluation used {results['test_size']:,} images. "
            f"Successfully evaluated models: {results['successful_models']}/{len(mnist_models)}."
        )
        st.warning(
            "Methodology note: all subsets are disjoint portions of the same official "
            "10,000-image MNIST test file; the final subset is held out from the "
            "benchmark, Elite evaluation and validation stages, not an external dataset."
        )

    # AI SUCCESS STORIES
    st.divider()
    st.subheader("🔍 AI Success Stories")
    st.write(
        "Discover digits where the Elite Jury made an incorrect or tied decision, "
        "but the Weighted Jury was correct."
    )
    if "independent_weighted_results" not in st.session_state:
        st.info("First complete the Independent Weighted Jury experiment.")
    else:
        results = st.session_state.independent_weighted_results
        final_y = np.asarray(results["final_y"])
        final_predictions = results["final_predictions"]
        weighted_predictions = np.asarray(results["weighted_predictions"])
        elite_names = results["elite_names"]
        final_indices = results["final_test_indices"]
        elite_predictions = calculate_elite_predictions(results)
        success_indices = np.flatnonzero(
            (elite_predictions != final_y) & (weighted_predictions == final_y)
        )
        c1, c2 = st.columns(2)
        c1.metric("Mistakes/ties corrected by Weighted Jury", len(success_indices))
        c2.metric("Final test images", len(final_y))
        st.write(f"The Weighted Jury corrected {len(success_indices)} Elite Jury incorrect or tied decisions.")
        if len(success_indices) == 0:
            st.warning("No correction examples were found in this test set.")
        else:
            selected = st.selectbox(
                "Choose an example:",
                list(range(len(success_indices))),
                format_func=lambda x: f"Success Story {x + 1}",
                key="success_story_example",
            )
            position = int(success_indices[selected])
            actual_digit = int(final_y[position])
            elite_digit = int(elite_predictions[position])
            weighted_digit = int(weighted_predictions[position])
            original_index = int(final_indices[position])
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("**Original handwritten image**")
                st.image(X_test[original_index].reshape(28, 28), width=220, clamp=True)
                st.metric("Actual digit", actual_digit)
            with c2:
                st.write("**Jury decisions**")
                if elite_digit == -1:
                    st.metric("Elite Jury prediction", "TIE", delta="No unique decision", delta_color="inverse")
                else:
                    st.metric("Elite Jury prediction", elite_digit, delta="Incorrect", delta_color="inverse")
                st.metric("Weighted Jury prediction", weighted_digit, delta="Correct")
            st.divider()
            st.subheader("🤖 What Did Each Elite Model Predict?")
            rows = []
            for name in elite_names:
                prediction = int(final_predictions[name][position])
                rows.append({
                    "AI Model": name,
                    "Prediction": prediction,
                    "Result": "Correct" if prediction == actual_digit else "Incorrect",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            correct_count = sum(row["Result"] == "Correct" for row in rows)
            st.subheader("🧠 Why Is This Interesting?")
            st.write(
                f"Out of the four Elite models, {correct_count} correctly predicted "
                f"the digit and {4 - correct_count} made mistakes."
            )
            st.info(
                "The Elite Jury uses equal voting, while the Weighted Jury gives "
                "different voting strengths to models based on validation accuracy."
            )

    # TASK 11: REPRODUCIBILITY AND EXPORTS
    st.divider()
    st.subheader("💾 Reproducibility and Experiment Downloads")
    st.write(
        "The experiment uses fixed random seeds. Re-running a stage with the same "
        "model bundle, dataset and settings selects the same image indices."
    )
    st.caption(
        f"Seeds — benchmark: {BENCHMARK_SEED}, Elite: {ELITE_SEED}, "
        f"weighted split: {WEIGHTED_SEED}."
    )
    export_payload = build_experiment_export()
    st.download_button(
        "⬇️ Download experiment settings and results (JSON)",
        data=json.dumps(export_payload, indent=2),
        file_name="mnist_ai_arena_experiment.json",
        mime="application/json",
        key="download_experiment_json",
    )
    if "benchmark_results" in st.session_state:
        benchmark_csv = pd.DataFrame(st.session_state.benchmark_results["individual"]).to_csv(index=False)
        st.download_button(
            "⬇️ Download benchmark results (CSV)",
            data=benchmark_csv,
            file_name="mnist_benchmark_results.csv",
            mime="text/csv",
            key="download_benchmark_csv",
        )
    if "independent_weighted_results" in st.session_state:
        weighted_results = st.session_state.independent_weighted_results
        weighted_csv = pd.DataFrame([
            {"Jury": "Original Jury", **weighted_results["original"]},
            {"Jury": "Elite Jury", **weighted_results["elite"]},
            {"Jury": "Weighted Jury", **weighted_results["weighted"]},
        ]).to_csv(index=False)
        st.download_button(
            "⬇️ Download final jury comparison (CSV)",
            data=weighted_csv,
            file_name="mnist_jury_comparison.csv",
            mime="text/csv",
            key="download_jury_comparison_csv",
        )


    # ============================================================
    # TASK 17: FINAL MODEL PERFORMANCE COMPARISON
    # ============================================================
    st.divider()
    st.header("🏁 Final Model Performance Comparison")
    if "independent_weighted_results" not in st.session_state:
        st.info(
            "Complete the original benchmark, Elite Jury evaluation and "
            "Independent Weighted Jury experiment to generate the final comparison."
        )
    else:
        final_results = st.session_state.independent_weighted_results
        final_actual = np.asarray(final_results["final_y"], dtype=int)
        final_prediction_dict = final_results["final_predictions"]

        individual_final_rows = []
        for model_name in final_results["original_names"]:
            model_prediction = final_prediction_dict.get(model_name)
            if model_prediction is None:
                continue
            model_prediction = validate_prediction_vector(
                model_prediction, len(final_actual), model_name
            )
            correct_count = int(np.sum(model_prediction == final_actual))
            individual_final_rows.append({
                "Model": model_name,
                "Correct": correct_count,
                "Incorrect": len(final_actual) - correct_count,
                "Final Test Accuracy (%)": round(
                    correct_count / len(final_actual) * 100, 2
                ) if len(final_actual) else 0.0,
            })

        if individual_final_rows:
            individual_final_df = pd.DataFrame(individual_final_rows).sort_values(
                "Final Test Accuracy (%)", ascending=False
            ).reset_index(drop=True)
            individual_final_df.insert(
                0, "Rank", range(1, len(individual_final_df) + 1)
            )
            st.subheader("Individual Model Results on the Final Test Set")
            st.dataframe(
                individual_final_df, hide_index=True, use_container_width=True
            )
            st.bar_chart(
                individual_final_df.set_index("Model")[["Final Test Accuracy (%)"]]
            )
            best_model = individual_final_df.iloc[0]
            st.metric(
                "Best Individual Model",
                best_model["Model"],
                f"{best_model['Final Test Accuracy (%)']:.2f}% final test accuracy",
            )
        else:
            st.warning("No individual model results are available for comparison.")

        jury_metric_rows = []
        for label, key in [
            ("Original 10-Model Jury", "original"),
            ("Elite 4-Model Jury", "elite"),
            ("Weighted 4-Model Jury", "weighted"),
        ]:
            metric = final_results[key]
            jury_metric_rows.append({
                "Jury Strategy": label,
                "Accuracy (%)": round(metric["accuracy"], 2),
                "Decision Coverage (%)": round(metric["coverage"], 2),
                "Unique-Decision Accuracy (%)": round(
                    metric["decided_accuracy"], 2
                ),
                "Ties": metric["ties"],
            })

        jury_comparison_df = pd.DataFrame(jury_metric_rows)
        st.subheader("Final Comparison of the Three Jury Strategies")
        st.dataframe(
            jury_comparison_df, hide_index=True, use_container_width=True
        )
        st.bar_chart(
            jury_comparison_df.set_index("Jury Strategy")[["Accuracy (%)"]]
        )
        best_jury = max(jury_metric_rows, key=lambda row: row["Accuracy (%)"])
        st.metric(
            "Highest Final Jury Accuracy",
            best_jury["Jury Strategy"],
            f"{best_jury['Accuracy (%)']:.2f}%",
        )
        st.caption(
            "The comparison uses the same held-out final images for all three "
            "jury strategies. Tied decisions count as incorrect in overall "
            "accuracy; unique-decision accuracy excludes ties."
        )

    # ============================================================
    # TASK 18: SUMMARY OF THE THREE JURY STRATEGIES
    # ============================================================
    st.divider()
    st.header("⚖️ Summary of the Three Jury Strategies")
    with st.expander("How do the three jury strategies work?", expanded=False):
        st.markdown("""
        **1. Original Jury – Equal Voting**
        - Uses all models that successfully completed final evaluation.
        - Every model contributes one vote.
        - The digit with the highest number of votes is selected.
        - If multiple digits share the highest vote count, the result is a tie.

        **2. Elite Jury – Equal Voting**
        - Uses the four models with the highest individual accuracy in the
          original 500-image benchmark.
        - The four selected models each contribute one vote.
        - The most-voted digit is selected; a tied maximum is recorded as a tie.

        **3. Weighted Elite Jury – Accuracy-Based Voting**
        - Uses the same four Elite models.
        - Each model's voting weight is its accuracy on a separate 500-image
          validation subset.
        - For each digit, the weights of models predicting that digit are added.
        - The digit with the highest total weight is selected. Equal maximum
          weighted totals are recorded as ties.

        **How performance is reported**
        - Overall accuracy counts a tie as an incorrect decision.
        - Decision coverage is the percentage of images with a unique decision.
        - Unique-decision accuracy measures correctness only among images where
          the jury reached a unique decision.
        """)
        if "independent_weighted_results" in st.session_state:
            strategy_results = st.session_state.independent_weighted_results
            strategy_summary = pd.DataFrame([
                {
                    "Strategy": "Original Jury",
                    "Models": len(strategy_results["original_names"]),
                    "Voting": "One vote per model",
                    "Accuracy (%)": round(strategy_results["original"]["accuracy"], 2),
                },
                {
                    "Strategy": "Elite Jury",
                    "Models": len(strategy_results["elite_names"]),
                    "Voting": "One vote per Elite model",
                    "Accuracy (%)": round(strategy_results["elite"]["accuracy"], 2),
                },
                {
                    "Strategy": "Weighted Elite Jury",
                    "Models": len(strategy_results["elite_names"]),
                    "Voting": "Validation-accuracy weights",
                    "Accuracy (%)": round(strategy_results["weighted"]["accuracy"], 2),
                },
            ])
            st.dataframe(
                strategy_summary, hide_index=True, use_container_width=True
            )
        st.info(
            "A more complex voting strategy is not automatically better. "
            "The final held-out results should be used to determine which "
            "strategy performed best in this experiment."
        )

    # ============================================================
    # TASK 19: EXPERIMENTAL METHODOLOGY
    # ============================================================
    st.divider()
    st.header("🧪 Experimental Methodology")
    with st.expander("Dataset, experiment stages and evaluation measures", expanded=False):
        st.markdown("""
        **Dataset and input representation**
        - The project uses the supplied MNIST test file (`mnist_test.npz`).
        - Each image is a 28 × 28 grayscale image represented by 784 features.
        - Pixel values are normalized to the range [0, 1].
        - The model bundle contains ten trained machine-learning models.

        **Experiment stages**
        1. **Original benchmark:** 500 images are used to measure individual
           model performance and identify the four Elite models.
        2. **Elite evaluation:** a separate 500-image subset is used to compare
           the full jury and the four-model Elite Jury.
        3. **Validation:** a separate 500-image subset is used to calculate
           each Elite model's voting weight from its validation accuracy.
        4. **Final evaluation:** a separate 1,000-image subset is used to
           compare the Original, Elite and Weighted juries.

        **Data separation**
        - The four subsets are selected without replacement and are checked
          for overlap.
        - The final evaluation subset is not used for Elite model selection
          or voting-weight calculation.
        - All subsets come from the same supplied 10,000-image MNIST test
          file. The final subset is held out within that file; it is not an
          external dataset.

        **Voting and evaluation**
        - The Original and Elite juries use equal/plurality voting.
        - The Weighted Elite Jury adds validation-accuracy weights for each
          digit predicted by an Elite model.
        - A tied maximum is recorded as no unique decision.
        - Overall accuracy counts ties as incorrect.
        - Decision coverage is the proportion of samples with a unique
          decision.
        - Unique-decision accuracy excludes tied samples.

        **Reproducibility**
        - Fixed random seeds are used for selecting experiment subsets.
        - Reproducibility assumes the same dataset, model bundle and settings.
        """)
        methodology_rows = [
            {"Stage": "Original benchmark", "Images": BENCHMARK_SIZE, "Seed": BENCHMARK_SEED},
            {"Stage": "Elite evaluation", "Images": ELITE_SIZE, "Seed": ELITE_SEED},
            {"Stage": "Weighted validation", "Images": VALIDATION_SIZE, "Seed": WEIGHTED_SEED},
            {"Stage": "Final evaluation", "Images": FINAL_TEST_SIZE, "Seed": WEIGHTED_SEED},
        ]
        st.dataframe(
            pd.DataFrame(methodology_rows),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "The final evaluation indices are drawn from the remaining images "
            "after the benchmark and Elite evaluation subsets have been removed; "
            "validation and final indices are selected together using the weighted "
            "experiment seed."
        )

# ============================================================
# 8. AI JURY BATTLE MODE
# ============================================================
elif st.session_state.selected_mode == "AI Jury Battle":
    st.subheader("⚔️ AI Jury Battle")
    st.write(
        "Draw a handwritten digit and challenge all ten AI models "
        "and three different jury strategies."
    )
    st.info(
        "Draw a digit in the black canvas using your mouse. Use a reasonably "
        "large digit and draw it near the centre."
    )

    if "battle_canvas_key" not in st.session_state:
        st.session_state.battle_canvas_key = 0
    if "battle_result" not in st.session_state:
        st.session_state.battle_result = None
    if "battle_image" not in st.session_state:
        st.session_state.battle_image = None
    if "battle_original_image" not in st.session_state:
        st.session_state.battle_original_image = None

    st.subheader("✍️ Round 1: Draw Your Digit")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key=f"battle_canvas_{st.session_state.battle_canvas_key}",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear Canvas"):
            clear_current_battle()
            st.rerun()
    with c2:
        run_battle = st.button("🚀 Challenge the AI Models", type="primary")

    if run_battle:
        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")
        else:
            original_array, processed_image = preprocess_handwritten_canvas(
                canvas_result.image_data
            )
            if processed_image is None or float(np.max(processed_image)) < 0.03:
                st.warning("The canvas is empty. Please draw a digit.")
            else:
                model_input = processed_image.reshape(1, 784)
                model_predictions, model_details, failed_models = predict_models(model_input)

                if not model_predictions:
                    st.error("All models failed to predict the digit.")
                elif "independent_weighted_results" not in st.session_state:
                    st.warning(
                        "Complete the Independent Weighted Jury experiment "
                        "in AI Jury mode before using all three jury strategies."
                    )
                else:
                    experiment = st.session_state.independent_weighted_results
                    elite_names = experiment["elite_names"]
                    weights = experiment["weights"]
                    available_elite = [
                        name for name in elite_names if name in model_predictions
                    ]
                    if len(available_elite) != 4:
                        st.error("All four Elite models must be available for this battle.")
                    else:
                        original_votes = np.bincount(
                            list(model_predictions.values()), minlength=10
                        )
                        elite_votes = np.bincount(
                            [model_predictions[name] for name in available_elite],
                            minlength=10,
                        )
                        weighted_votes = np.zeros(10, dtype=float)
                        for name in available_elite:
                            weighted_votes[model_predictions[name]] += weights[name]

                        st.session_state.battle_result = {
                            "model_predictions": model_predictions.copy(),
                            "model_details": model_details.copy(),
                            "original_decision": jury_decision(original_votes),
                            "elite_decision": jury_decision(elite_votes),
                            "weighted_decision": jury_decision(weighted_votes),
                            "original_votes": original_votes.copy(),
                            "elite_votes": elite_votes.copy(),
                            "weighted_votes": weighted_votes.copy(),
                            "failed_models": failed_models.copy(),
                        }
                        st.session_state.battle_image = processed_image.copy()
                        st.session_state.battle_original_image = original_array.copy()
                        st.session_state.battle_saved = False
                        st.session_state.pop("battle_revealed", None)
                        st.success("All jury predictions are ready!")

    # DISPLAY BATTLE RESULTS
    if st.session_state.battle_result is not None:
        result = st.session_state.battle_result
        st.divider()
        st.subheader("🤖 Round 2: Individual AI Predictions")
        c1, c2 = st.columns(2)
        with c1:
            st.image(
                st.session_state.battle_original_image,
                caption="Original drawing",
                width=220,
            )
        with c2:
            st.image(
                st.session_state.battle_image,
                caption="Processed 28 × 28 MNIST image",
                width=220,
                clamp=True,
            )

        prediction_rows = []
        for name, details in result["model_details"].items():
            confidence = details["confidence"]
            prediction_rows.append({
                "AI Model": name,
                "Predicted Digit": details["prediction"],
                "Probability Confidence": (
                    f"{confidence:.2f}%" if confidence is not None else "Not available"
                ),
            })
        st.dataframe(pd.DataFrame(prediction_rows), hide_index=True, use_container_width=True)

        st.subheader("📈 Model Confidence Comparison")
        confidence_data = [
            {"AI Model": name, "Confidence (%)": details["confidence"]}
            for name, details in result["model_details"].items()
            if details["confidence"] is not None
        ]
        if confidence_data:
            st.bar_chart(pd.DataFrame(confidence_data).set_index("AI Model"))
            st.caption(
                "Confidence represents the predicted-class probability reported by "
                "each model. It is not a guarantee of accuracy, and values from "
                "different models may not be directly comparable."
            )
        else:
            st.info("Probability confidence is unavailable for these models.")
        if result["failed_models"]:
            st.warning("Some models could not predict: " + ", ".join(result["failed_models"]))

        st.divider()
        st.subheader("🧠 Model Agreement Analysis")
        predictions = result["model_predictions"]
        vote_counts = np.bincount(list(predictions.values()), minlength=10)
        total_models = len(predictions)
        highest_votes = int(vote_counts.max())
        winning_digits = np.flatnonzero(vote_counts == highest_votes)
        agreement_percentage = highest_votes / total_models * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Strongest Agreement", f"{highest_votes}/{total_models}")
        c2.metric("Highest Vote Share", f"{agreement_percentage:.1f}%")
        c3.metric("Different Predictions", len(np.unique(list(predictions.values()))))
        if len(winning_digits) == 1:
            winning_digit = int(winning_digits[0])
            st.success(
                f"Digit {winning_digit} received the highest number of votes "
                f"({highest_votes})."
            )
            agreeing = [name for name, pred in predictions.items() if pred == winning_digit]
            disagreeing = [name for name, pred in predictions.items() if pred != winning_digit]
            st.write(f"**Models supporting digit {winning_digit}:**")
            st.success(", ".join(agreeing))
            st.write("**Models predicting a different digit:**")
            if disagreeing:
                st.warning(", ".join(disagreeing))
            else:
                st.success("All models agree!")
        else:
            st.warning(
                "There is no unique most-supported digit. The top vote is tied between: "
                + ", ".join(map(str, winning_digits))
            )
            for digit in winning_digits:
                supporters = [name for name, pred in predictions.items() if pred == int(digit)]
                st.write(f"**Models voting for {digit}:** " + ", ".join(supporters))

        st.subheader("📊 Model Voting Distribution")
        st.bar_chart(pd.DataFrame({
            "Number of Models": vote_counts
        }, index=pd.Index(range(10), name="Digit")))

        st.divider()
        st.subheader("⚖️ Round 3: Three Jury Strategies")
        c1, c2, c3 = st.columns(3)
        for col, label, key in [
            (c1, "Original Jury", "original_decision"),
            (c2, "Elite Jury", "elite_decision"),
            (c3, "Weighted Jury", "weighted_decision"),
        ]:
            with col:
                st.markdown(f"**{label}**")
                if result[key] == -1:
                    st.warning("TIE")
                else:
                    st.metric("Prediction", result[key])

        st.subheader("📊 Original Jury Vote Distribution")
        st.bar_chart(pd.DataFrame(
            {"Votes": result["original_votes"]},
            index=pd.Index(range(10), name="Digit"),
        ))
        st.subheader("📊 Elite Jury Vote Distribution")
        st.bar_chart(pd.DataFrame(
            {"Votes": result["elite_votes"]},
            index=pd.Index(range(10), name="Digit"),
        ))
        st.subheader("📊 Weighted Jury Vote Distribution")
        st.bar_chart(pd.DataFrame(
            {"Weighted Votes": result["weighted_votes"]},
            index=pd.Index(range(10), name="Digit"),
        ))

        st.divider()
        st.subheader("🏁 Round 4: Reveal the Actual Digit")
        st.write(
            "Enter the digit you intended to draw. The application will compare "
            "all three jury decisions against your answer."
        )
        actual_digit = st.selectbox(
            "What digit did you draw?",
            list(range(10)),
            key="battle_actual_digit",
        )
        if st.button("🎯 Reveal Answer and Declare Winner", type="primary"):
            st.session_state.battle_revealed = int(actual_digit)
            if not st.session_state.battle_saved:
                save_battle_result(
                    st.session_state.battle_result,
                    int(actual_digit),
                    st.session_state.battle_history,
                )
                st.session_state.battle_saved = True
            st.rerun()

        if "battle_revealed" in st.session_state:
            actual = int(st.session_state.battle_revealed)
            st.divider()
            st.subheader("🏆 Battle Results")
            st.metric("Your intended digit", actual)

            st.subheader("🤖 Individual Model Battle Results")
            individual_rows = []
            for name, prediction in result["model_predictions"].items():
                individual_rows.append({
                    "AI Model": name,
                    "Prediction": int(prediction),
                    "Result": "Correct" if int(prediction) == actual else "Incorrect",
                })
            st.dataframe(pd.DataFrame(individual_rows), hide_index=True, use_container_width=True)
            correct_models = sum(
                int(prediction) == actual
                for prediction in result["model_predictions"].values()
            )
            total_models = len(result["model_predictions"])
            accuracy = correct_models / total_models * 100 if total_models else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Correct Models", f"{correct_models}/{total_models}")
            c2.metric("Incorrect Models", total_models - correct_models)
            c3.metric("Individual Model Accuracy", f"{accuracy:.1f}%")
            if correct_models == total_models:
                st.success("All individual AI models predicted correctly!")
            elif correct_models:
                st.info(f"{correct_models} out of {total_models} models predicted correctly.")
            else:
                st.error("None of the individual AI models predicted correctly.")

            strategies = {
                "Original Jury": result["original_decision"],
                "Elite Jury": result["elite_decision"],
                "Weighted Jury": result["weighted_decision"],
            }
            battle_rows = []
            for name, prediction in strategies.items():
                outcome = (
                    "Tie" if prediction == -1
                    else "Correct" if prediction == actual
                    else "Incorrect"
                )
                battle_rows.append({
                    "Jury Strategy": name,
                    "Prediction": "TIE" if prediction == -1 else prediction,
                    "Result": outcome,
                })
            st.dataframe(pd.DataFrame(battle_rows), hide_index=True, use_container_width=True)

            correct_strategies = [name for name, pred in strategies.items() if pred == actual]
            tied_strategies = [name for name, pred in strategies.items() if pred == -1]
            incorrect_strategies = [
                name for name, pred in strategies.items()
                if pred != actual and pred != -1
            ]
            if len(correct_strategies) == 3:
                st.balloons()
                st.success("All three jury strategies predicted correctly!")
            else:
                if correct_strategies:
                    st.success("Correct strategies: " + ", ".join(correct_strategies))
                if tied_strategies:
                    st.warning("Strategies that produced a tie: " + ", ".join(tied_strategies))
                if incorrect_strategies:
                    st.error("Incorrect strategies: " + ", ".join(incorrect_strategies))

            st.caption(
                "Note: The actual digit is self-reported by the person drawing. "
                "The application cannot independently verify the intended digit."
            )

            # TASK 05: SESSION LEADERBOARD AND BATTLE HISTORY
            st.divider()
            st.subheader("🏆 Session Leaderboard")
            history = st.session_state.battle_history
            if history:
                st.dataframe(
                    create_battle_leaderboard(history),
                    hide_index=True,
                    use_container_width=True,
                )
                st.subheader("📜 Battle History")
                history_rows = []
                for battle in history:
                    row = {
                        "Battle": battle["Battle"],
                        "Actual Digit": battle["Actual Digit"],
                        "Correct Models": (
                            f"{battle['Correct Models']}/{battle['Total Models']}"
                        ),
                        "Individual Accuracy (%)": battle["Individual Accuracy (%)"],
                    }
                    for strategy, details in battle["strategies"].items():
                        row[strategy] = (
                            "TIE" if details["result"] == "Tie"
                            else f"{details['prediction']} ({details['result']})"
                        )
                    history_rows.append(row)
                st.dataframe(
                    pd.DataFrame(history_rows),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No completed battles in this session yet.")

            if st.button("🔄 Start Another Battle"):
                clear_current_battle()
                st.rerun()

if st.session_state.selected_mode == "AI Jury":
    # ============================================================
    # 9. AI MODEL DISAGREEMENT EXPLORER
    # ============================================================
    st.markdown("---")
    st.header("🧩 AI Model Disagreement Explorer")
    st.write(
        "Explore handwritten digits where the four Elite AI models produce "
        "different predictions. Compare equal voting with weighted voting."
    )

    if "independent_weighted_results" not in st.session_state:
        st.info("Please complete the Independent Weighted Jury experiment first.")
    else:
        results = st.session_state.independent_weighted_results
        elite_names = results["elite_names"]
        final_predictions = results["final_predictions"]
        final_y = results["final_y"]
        weighted_predictions = results["weighted_predictions"]
        final_indices = results["final_test_indices"]

        elite_matrix = np.column_stack([final_predictions[name] for name in elite_names])
        disagreement_counts = np.array([
            len(np.unique(row)) for row in elite_matrix
        ])
        disagreement_filter = st.selectbox(
            "How many different predictions should the Elite models produce?",
            [
                "All disagreements", "2 different predictions",
                "3 different predictions", "4 different predictions",
            ],
            key="disagreement_filter",
        )
        if disagreement_filter == "All disagreements":
            candidates = np.flatnonzero(disagreement_counts > 1)
        elif disagreement_filter == "2 different predictions":
            candidates = np.flatnonzero(disagreement_counts == 2)
        elif disagreement_filter == "3 different predictions":
            candidates = np.flatnonzero(disagreement_counts == 3)
        else:
            candidates = np.flatnonzero(disagreement_counts == 4)

        st.metric("Images matching this disagreement level", len(candidates))
        if len(candidates) == 0:
            st.warning("No images found for this disagreement level. Please select another level.")
        else:
            selected_position = st.selectbox(
                "Select an image:",
                options=candidates.tolist(),
                format_func=lambda pos: (
                    f"Example {pos + 1} | {disagreement_counts[pos]} different predictions"
                ),
                key="disagreement_example",
            )
            selected_position = int(selected_position)
            original_index = int(final_indices[selected_position])
            image = X_test[original_index].reshape(28, 28)
            actual_digit = int(final_y[selected_position])
            model_votes = elite_matrix[selected_position]
            weighted_decision = int(weighted_predictions[selected_position])
            unique_digits, counts = np.unique(model_votes, return_counts=True)
            top = counts.max()
            equal_winners = unique_digits[counts == top]
            equal_decision = int(equal_winners[0]) if len(equal_winners) == 1 else -1

            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption="Original handwritten image", width=220)
            with c2:
                st.metric("Actual digit", actual_digit)
                st.metric("Different Elite predictions", disagreement_counts[selected_position])

            st.subheader("Individual Elite Model Predictions")
            rows = []
            for name, prediction in zip(elite_names, model_votes):
                prediction = int(prediction)
                rows.append({
                    "Elite AI Model": name,
                    "Prediction": prediction,
                    "Result": "Correct" if prediction == actual_digit else "Incorrect",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.subheader("Equal-Vote Distribution")
            st.bar_chart(pd.DataFrame({"Votes": counts}, index=pd.Index(unique_digits, name="Digit")))

            st.subheader("Equal Voting vs Weighted Voting")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Equal-Vote Elite Jury**")
                if equal_decision == -1:
                    st.warning("TIE – No unique decision")
                else:
                    st.metric("Jury prediction", equal_decision)
                    if equal_decision == actual_digit:
                        st.success("Correct prediction")
                    else:
                        st.error("Incorrect prediction")
            with c2:
                st.markdown("**Weighted Elite Jury**")
                if weighted_decision == -1:
                    st.warning("TIE – No unique decision")
                else:
                    st.metric("Jury prediction", weighted_decision)
                    if weighted_decision == actual_digit:
                        st.success("Correct prediction")
                    else:
                        st.error("Incorrect prediction")

            st.subheader("What Can We Learn?")
            correct_models = int(np.sum(model_votes == actual_digit))
            st.write(f"Out of the four Elite models, {correct_models} predicted the correct digit.")
            if equal_decision == -1 and weighted_decision == actual_digit:
                st.success("Equal voting produced a tie, but weighted voting identified the correct digit.")
            elif equal_decision != actual_digit and weighted_decision == actual_digit:
                st.success("Weighted voting corrected the incorrect equal-vote decision.")
            elif equal_decision == actual_digit and weighted_decision == actual_digit:
                st.info("Both voting strategies reached the correct answer.")
            elif equal_decision == actual_digit and weighted_decision != actual_digit:
                st.warning("Equal voting was correct, but weighted voting was incorrect or tied.")
            else:
                st.warning("Neither voting strategy reached the correct answer.")

    # ============================================================
    # 10. AI CONFUSION DETECTIVE
    # ============================================================
    st.divider()
    st.header("🔍 AI Confusion Detective")
    st.write(
        "Investigate which handwritten digits the Weighted Jury recognizes "
        "correctly and which digits it confuses."
    )

    if "independent_weighted_results" not in st.session_state:
        st.info("Complete the Independent Weighted Jury experiment first.")
    else:
        results = st.session_state.independent_weighted_results
        actual = np.asarray(results["final_y"], dtype=int)
        predicted = np.asarray(results["weighted_predictions"], dtype=int)
        if len(actual) != len(predicted):
            st.error("Analysis error: actual labels and jury predictions have different lengths.")
        else:
            valid = predicted != -1
            actual_valid = actual[valid]
            predicted_valid = predicted[valid]
            total_ties = int(np.sum(predicted == -1))
            st.metric("Tied decisions excluded from confusion matrix", total_ties)

            st.markdown("### 1. Confusion Matrix")
            st.write("Rows represent actual digits. Columns represent digits predicted by the Weighted Jury.")
            confusion = np.zeros((10, 10), dtype=int)
            for true_digit, predicted_digit in zip(actual_valid, predicted_valid):
                confusion[int(true_digit), int(predicted_digit)] += 1
            confusion_df = pd.DataFrame(
                confusion,
                index=[f"Actual {i}" for i in range(10)],
                columns=[f"Predicted {i}" for i in range(10)],
            )
            st.dataframe(confusion_df, use_container_width=True)
            st.caption(
                "Diagonal cells represent correct predictions. Off-diagonal cells "
                "represent misclassifications. Tied decisions are reported separately."
            )

            st.markdown("### 2. Most Common Digit Confusions")
            pairs = []
            for true_digit in range(10):
                for predicted_digit in range(10):
                    if true_digit != predicted_digit:
                        count = int(confusion[true_digit, predicted_digit])
                        if count:
                            pairs.append({
                                "Actual Digit": true_digit,
                                "Predicted Digit": predicted_digit,
                                "Number of Mistakes": count,
                            })
            pairs.sort(key=lambda item: item["Number of Mistakes"], reverse=True)
            if pairs:
                st.write("These are the ten most frequent incorrect predictions.")
                st.dataframe(pd.DataFrame(pairs[:10]), hide_index=True, use_container_width=True)
                labels = [
                    f"{row['Actual Digit']} → {row['Predicted Digit']}"
                    for row in pairs[:10]
                ]
                values = [row["Number of Mistakes"] for row in pairs[:10]]
                st.bar_chart(pd.DataFrame({"Number of Mistakes": values}, index=labels))
            else:
                st.success("No incorrect digit predictions found!")

            st.markdown("### 3. Accuracy for Each Digit")
            digit_results = []
            for digit in range(10):
                mask = actual == digit
                total = int(np.sum(mask))
                correct = int(np.sum((actual == digit) & (predicted == digit)))
                ties = int(np.sum((actual == digit) & (predicted == -1)))
                accuracy = correct / total * 100 if total else 0.0
                digit_results.append({
                    "Digit": digit,
                    "Total Images": total,
                    "Correct Predictions": correct,
                    "Incorrect/Tied": total - correct,
                    "Tied Decisions": ties,
                    "Accuracy (%)": round(accuracy, 2),
                })
            digit_df = pd.DataFrame(digit_results)
            st.dataframe(digit_df, hide_index=True, use_container_width=True)

            st.markdown("### 4. Easiest and Hardest Digits")
            easiest = max(digit_results, key=lambda item: item["Accuracy (%)"])
            hardest = min(digit_results, key=lambda item: item["Accuracy (%)"])
            c1, c2 = st.columns(2)
            c1.metric("Easiest Digit", easiest["Digit"], f"{easiest['Accuracy (%)']}% accuracy")
            c2.metric("Hardest Digit", hardest["Digit"], f"{hardest['Accuracy (%)']}% accuracy")
            st.bar_chart(digit_df.set_index("Digit")[["Accuracy (%)"]])
            st.info(
                "A low accuracy for a particular digit may indicate that its handwritten "
                "patterns are easily confused with other digits. The confusion matrix "
                "helps identify those specific patterns."
            )


# ============================================================
# 11. MODEL INFORMATION
# ============================================================
with st.expander("View AI models"):
    for i, name in enumerate(mnist_models.keys(), start=1):
        st.write(f"{i}. {name}")

# TASKS 15–16: DIAGNOSTICS, TEST GUIDE AND EDGE-CASE VISIBILITY
with st.expander("🧪 System diagnostics and final testing checklist"):
    st.write("**Automatic checks**")
    st.write(f"- Dataset: {len(X_test):,} images, {X_test.shape[1]} features.")
    st.write(f"- Labels: digits {int(np.min(y_test))} to {int(np.max(y_test))}.")
    st.write(f"- Pixel range: {float(np.min(X_test)):.3f} to {float(np.max(X_test)):.3f}.")
    st.write(f"- Loaded models: {len(mnist_models)}.")
    if st.button("▶ Run one-image prediction check for every model", key="run_model_diagnostics"):
        diagnostic_input = X_test[0].reshape(1, -1)
        diagnostic_rows = []
        for name, model in mnist_models.items():
            try:
                prediction = validate_prediction_vector(model.predict(diagnostic_input), 1, name)[0]
                diagnostic_rows.append({"Model": name, "Status": "PASS", "Prediction": int(prediction)})
            except Exception as exc:
                diagnostic_rows.append({"Model": name, "Status": "FAIL", "Prediction": str(exc)[:120]})
        st.dataframe(pd.DataFrame(diagnostic_rows), hide_index=True, use_container_width=True)
        passed = sum(row["Status"] == "PASS" for row in diagnostic_rows)
        if passed == len(mnist_models):
            st.success(f"All {passed} models passed the one-image prediction check.")
        else:
            st.warning(f"{passed}/{len(mnist_models)} models passed. Check failed model details above.")

    st.markdown("**Manual end-to-end checks to complete on your computer**")
    st.markdown("""
    1. Open **AI Arena**, complete all ten rounds, view the final leaderboard, and start a new competition.
    2. Open **AI Jury**, run the 500-image benchmark, Elite evaluation and weighted experiment.
    3. Download the JSON and CSV files and confirm that they open correctly.
    4. Check that the disagreement explorer and confusion detective display only in AI Jury mode.
    5. Open **AI Jury Battle**, draw a digit, reveal predictions, enter the intended digit, and complete two battles.
    6. Check that Clear Canvas, Start Another Battle, ties, and battle history behave correctly.
    7. Test an empty canvas and confirm that the app displays a warning instead of crashing.
    """)

# ============================================================
# TASK 20: FINAL PROJECT DEMONSTRATION GUIDE
# ============================================================
with st.expander("🎤 Final Project Demonstration Guide", expanded=False):
    st.markdown("""
    ## Suggested demonstration sequence

    **1. Introduce the project**
    - Explain that MNIST AI ARENA compares human handwritten-digit recognition
      with ten trained machine-learning models.
    - Introduce the three experiences: AI Arena, AI Jury and AI Jury Battle.

    **2. Demonstrate AI Arena**
    - Start the ten-round competition.
    - Enter a human prediction before revealing the AI predictions.
    - Show the round results, scores and final leaderboard.

    **3. Demonstrate AI Jury**
    - Generate a handwritten MNIST image and show the individual model votes.
    - Run the original 500-image benchmark.
    - Explain how the four Elite models are selected.
    - Run the Elite evaluation and the independent weighted experiment.
    - Present the final model comparison, jury comparison and methodology.
    - Point out the JSON and CSV experiment downloads.

    **4. Demonstrate AI Jury Battle**
    - Draw a digit and challenge the models.
    - Show individual predictions and available confidence values.
    - Compare the Original, Elite and Weighted jury decisions.
    - Enter the intended digit and explain that this is self-reported.
    - Show the battle history and session leaderboard.

    **5. Conclude**
    - Compare the measured performance of individual models and jury strategies.
    - Explain that ties count as incorrect in overall accuracy.
    - State clearly that the final evaluation subset is held out from model
      selection and weight calculation, but comes from the same supplied
      MNIST test file rather than an external dataset.

    **Presentation reminder:** report the actual results produced by the app.
    Do not claim that one jury strategy is always superior based on a single
    experiment.
    """)


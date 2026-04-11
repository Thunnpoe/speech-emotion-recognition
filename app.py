import numpy as np
import streamlit as st
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image
from melspec import plot_colored_polar, plot_melspec

# load only self-trained models
SER_MODEL_PATH = os.getenv("SER_MODEL_PATH", "model3_self_trained.h5")
GENDER_MODEL_PATH = os.getenv("GENDER_MODEL_PATH", "model_mw_self_trained.h5")

if not os.path.exists(SER_MODEL_PATH):
    st.error(
        "Missing self-trained emotion model. "
        "Train it first with `python train_from_scratch.py`."
    )
    st.stop()

model = load_model(SER_MODEL_PATH)

# constants
starttime = datetime.now()

CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {"neutral": "grey",
              "positive": "green",
              "happy": "green",
              "surprise": "orange",
              "fear": "purple",
              "negative": "red",
              "angry": "red",
              "sad": "lightblue",
              "disgust": "brown"}

TEST_CAT = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
TEST_PRED = np.array([.3, .3, .4, .1, .6, .9, .1])
EMOTION_EMOJI = {
    "fear": "😨",
    "angry": "😠",
    "neutral": "😐",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "disgust": "🤢",
}

EMOTION_LABELS_PATH = os.getenv("SER_LABELS_PATH", "model3_self_trained.labels.txt")
EMOTION_ALIASES = {
    "fear": {"fear", "fearful"},
    "angry": {"angry", "negative", "disgust"},
    "neutral": {"neutral", "calm"},
    "happy": {"happy", "positive"},
    "sad": {"sad"},
    "surprise": {"surprise", "surprised"},
}


def load_emotion_model_labels(path=EMOTION_LABELS_PATH):
    if not os.path.exists(path):
        return CAT6
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip().lower() for line in f if line.strip()]
    return labels or CAT6


MODEL_EMOTION_LABELS = load_emotion_model_labels()


def remap_pred_to_ui(predictions):
    label_to_idx = {label: idx for idx, label in enumerate(MODEL_EMOTION_LABELS)}
    ui_pred = np.zeros(len(CAT6), dtype=np.float32)
    for ui_idx, ui_label in enumerate(CAT6):
        aliases = EMOTION_ALIASES.get(ui_label, {ui_label})
        for alias in aliases:
            src_idx = label_to_idx.get(alias)
            if src_idx is not None and src_idx < len(predictions):
                ui_pred[ui_idx] += float(predictions[src_idx])
    return ui_pred


def sentiment_scores_from_ui(ui_pred):
    idx = {name: i for i, name in enumerate(CAT6)}
    pos = ui_pred[idx["happy"]] + ui_pred[idx["surprise"]] * 0.5
    neu = ui_pred[idx["neutral"]] + ui_pred[idx["surprise"]] * 0.5 + ui_pred[idx["sad"]] * 0.5
    neg = ui_pred[idx["fear"]] + ui_pred[idx["angry"]] + ui_pred[idx["sad"]] * 0.5
    return np.array([pos, neu, neg], dtype=np.float32)


# page settings
st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")
# COLOR = "#1f1f2e"
# BACKGROUND_COLOR = "#d1d1e0"


# @st.cache(hash_funcs={tf_agents.utils.object_identity.ObjectIdentityDictionary: load_model})
# def load_model_cache(model):
#     return load_model(model)

# @st.cache
def log_file(txt=None):
    with open("log.txt", "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")


# @st.cache
def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


# @st.cache
def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)


# @st.cache
def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit), dtype=np.float32)
        mfccs[:, :a.shape[1]] = a
    else:
        mfccs = a
    return mfccs


def prepare_mfcc_input(audio_path, keras_model):
    shape = keras_model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) == 4:
        # Expected shape: (None, 40, time_steps, channels)
        limit = int(shape[2])
        mfccs = get_mfccs(audio_path, limit)
        return mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)

    # Backward compatibility with older 3D models: (None, 40, time_steps)
    limit = int(shape[-1])
    mfccs = get_mfccs(audio_path, limit)
    return mfccs.reshape(1, *mfccs.shape)


@st.cache_data
def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title


@st.cache_data
def color_dict(coldict=COLOR_DICT):
    return COLOR_DICT


def plot_polar(fig, predictions=TEST_PRED, categories=TEST_CAT,
               title="TEST", colors=COLOR_DICT):
    # color_sector = "grey"

    N = len(predictions)
    ind = predictions.argmax()

    COLOR = color_sector = colors[categories[ind]]
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = np.zeros_like(predictions)
    radii[predictions.argmax()] = predictions.max() * 10
    width = np.pi / 1.8 * predictions
    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar="True")
    ax.bar(theta, radii, width=width, bottom=0.0, color=color_sector, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.suptitle(title, color="darkblue", size=12)
    plt.title(f"BIG {N}\n", color=COLOR)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)


def apply_ui_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        :root {
            --bg-soft: #eef3ff;
            --bg-cream: #f8fafc;
            --ink-1: #0f172a;
            --ink-2: #334155;
            --line: rgba(30, 58, 138, 0.18);
            --brand-a: #1e3a8a;
            --brand-b: #0284c7;
        }

        html, body, [class*="css"] {
            font-family: "Manrope", "Trebuchet MS", "Segoe UI", sans-serif;
        }

       .stApp {
            background:
              radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.45) 0%, transparent 55%),
              radial-gradient(circle at 90% 25%, rgba(6, 182, 212, 0.35) 0%, transparent 55%),
              radial-gradient(circle at 70% 95%, rgba(16, 185, 129, 0.25) 0%, transparent 55%),
              linear-gradient(180deg, #050814 0%, #0b1220 55%, #060a18 100%);
        }

        .block-container {
            padding-top: 1.15rem !important;
            padding-bottom: 1.8rem !important;
            max-width: 1180px;
        }

        .hero {
            background: linear-gradient(120deg, #0f172a 0%, var(--brand-a) 50%, #0ea5e9 100%);
            color: #ffffff;
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.14);
        }

        .hero h1 {
            margin: 0;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            color: #ffffff;
        }

        .hero p {
            margin: 0.4rem 0 0;
            font-size: 0.95rem;
            opacity: 0.95;
        }

        .panel-title {
            margin: 0.5rem 0 0.6rem;
            color: #ffffff;
            font-size: 1.08rem;
            font-weight: 700;
        }

        .stat-chip {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.52rem 0.78rem;
            margin-bottom: 0.75rem;
            color: var(--ink-2);
            font-size: 0.86rem;
            font-weight: 700;
            text-align: center;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
        }

        .subtle-note {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            color: var(--ink-2);
            border-radius: 12px;
            padding: 0.58rem 0.72rem;
            margin: 0.2rem 0 0.9rem;
            font-size: 0.86rem;
        }

        .section-divider {
            height: 1px;
            width: 100%;
            margin: 0.35rem 0 1rem;
            background: linear-gradient(90deg, transparent 0%, rgba(30, 58, 138, 0.42) 12%, rgba(30, 58, 138, 0.14) 88%, transparent 100%);
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eaf1ff 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stSidebarContent"] {
            padding-top: 0.2rem;
        }

        div[data-testid="stSidebar"] [data-testid="stImage"] {
            margin-top: 0 !important;
            margin-bottom: 1rem;
        }

        div[data-testid="stSidebar"] [data-testid="stImage"] img {
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.45);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.24), 0 0 0 4px rgba(255, 255, 255, 0.12);
            filter: saturate(1.08) contrast(1.03);
            transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease;
        }

        div[data-testid="stSidebar"] [data-testid="stImage"] img:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.3), 0 0 0 4px rgba(255, 255, 255, 0.18);
            filter: saturate(1.14) contrast(1.05);
        }

        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--ink-2);
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed rgba(30, 58, 138, 0.48);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.76);
            transition: all 0.2s ease-in-out;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(2, 132, 199, 0.8);
            box-shadow: 0 8px 18px rgba(2, 132, 199, 0.15);
        }

        div[data-testid="stFileUploaderFile"] * {
            color: #ffffff !important;
        }

        .result-card {
            background: linear-gradient(130deg, rgba(255,255,255,0.95) 0%, rgba(224,242,254,0.95) 100%);
            border: 1px solid rgba(2, 132, 199, 0.35);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(2, 132, 199, 0.12);
            margin: 0.5rem 0 0.8rem;
        }

        .result-label {
            color: #0c4a6e;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0 0 0.25rem 0;
        }

        .result-main {
            color: #0f172a;
            font-size: 1.2rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.25;
        }

        .result-sub {
            color: #334155;
            font-size: 0.9rem;
            font-weight: 700;
            margin: 0.32rem 0 0;
        }

        .stButton > button {
            border: 0;
            border-radius: 11px;
            background: linear-gradient(120deg, var(--brand-a) 0%, var(--brand-b) 100%);
            color: white;
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(2, 132, 199, 0.22);
        }

        .stButton > button:hover {
            filter: brightness(1.05);
        }

        [data-testid="stCheckbox"] label p {
            font-weight: 600;
            color: var(--ink-2);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_ui_theme()
    website_menu = "Emotion Recognition"
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if website_menu == "Emotion Recognition":
        model_type = "mfccs"
        em3 = em6 = gender = True
        audio_file = None
        path = None
        wav = sr = Xdb = mfccs = None
        pred = data3 = None
        top_emotion = sentiment = None
        top_conf = 0.0

        st.markdown(
            """
            <div class="hero">
                <h1>Speech Emotion Recognition</h1>
                <p>Upload a voice sample and get emotion insights with waveform, MFCC, and prediction visuals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="panel-title">Upload Audio</p>', unsafe_allow_html=True)
        with st.container():
            col1, col2 = st.columns(2)
            # audio_file = None
            # path = None
            with col1:
                audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
                if audio_file is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audio_file.name)
                    if_save_audio = save_audio(audio_file)
                    if if_save_audio == 1:
                        st.warning("File size is too large. Try another file.")
                    elif if_save_audio == 0:
                        # extract features
                        try:
                            wav, sr = librosa.load(path, sr=44100)
                            Xdb = get_melspec(path)[1]
                            mfccs = librosa.feature.mfcc(y=wav, sr=sr)
                            model_input = prepare_mfcc_input(path, model)
                            pred = model.predict(model_input, verbose=0)[0]
                            pred = remap_pred_to_ui(pred)
                            pos, neu, neg = sentiment_scores_from_ui(pred)
                            data3 = np.array([pos, neu, neg])
                            top_emotion = CAT6[int(pred.argmax())]
                            top_conf = float(pred.max()) * 100
                            sentiment = CAT3[int(data3.argmax())]
                            st.audio(audio_file, format='audio/wav', start_time=0)
                            # # display audio
                            # st.audio(audio_file, format='audio/wav', start_time=0)
                        except Exception as e:
                            audio_file = None
                            st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
                    else:
                        st.error("Unknown error")
            with col2:
                if audio_file is not None and top_emotion is not None:
                    emoji = EMOTION_EMOJI.get(top_emotion, "")
                    st.markdown("<div style='height: 2.2rem'></div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <p class="result-label">Detected Emotion</p>
                            <p class="result-main">{emoji} {top_emotion.title()} ({top_conf:.1f}%)</p>
                            <p class="result-sub">Overall tone: {sentiment.title()}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    idx = {name: i for i, name in enumerate(CAT6)}
                    st.caption(
                        f"Confidence mix | Happy: {pred[idx['happy']]:.2f}  Sad: {pred[idx['sad']]:.2f}  "
                        f"Angry: {pred[idx['angry']]:.2f}  Fear: {pred[idx['fear']]:.2f}"
                    )
                else:
                    pass
            #     st.write("Record audio file")
            #     if st.button('Record'):
            #         with st.spinner(f'Recording for 5 seconds ....'):
            #             st.write("Recording...")
            #             time.sleep(3)
            #         st.success("Recording completed")
            #         st.write("Error while loading the file")

        # with st.sidebar.expander("Change colors"):
        #     st.sidebar.write("Use this options after you got the plots")
        #     col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        #
        #     with col1:
        #         a = st.color_picker("Angry", value="#FF0000")
        #     with col2:
        #         f = st.color_picker("Fear", value="#800080")
        #     with col3:
        #         d = st.color_picker("Disgust", value="#A52A2A")
        #     with col4:
        #         sd = st.color_picker("Sad", value="#ADD8E6")
        #     with col5:
        #         n = st.color_picker("Neutral", value="#808080")
        #     with col6:
        #         sp = st.color_picker("Surprise", value="#FFA500")
        #     with col7:
        #         h = st.color_picker("Happy", value="#008000")
        #     if st.button("Update colors"):
        #         global COLOR_DICT
        #         COLOR_DICT = {"neutral": n,
        #                       "positive": h,
        #                       "happy": h,
        #                       "surprise": sp,
        #                       "fear": f,
        #                       "negative": a,
        #                       "angry": a,
        #                       "sad": sd,
        #                       "disgust": d}
        #         st.success(COLOR_DICT)

        if audio_file is not None:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<p class="panel-title">Analyzing Audio</p>', unsafe_allow_html=True)

            with st.container():
                fig = plt.figure(figsize=(10, 1.35))
                fig.set_facecolor('#d1d1e0')
                plt.title("Wave-form")
                librosa.display.waveshow(wav, sr=44100)
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.spines["right"].set_visible(False)
                plt.gca().axes.spines["left"].set_visible(False)
                plt.gca().axes.spines["top"].set_visible(False)
                plt.gca().axes.spines["bottom"].set_visible(False)
                plt.gca().axes.set_facecolor('#d1d1e0')
                st.write(fig)
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = plt.figure(figsize=(10, 2.5))
                    fig2.set_facecolor('#d1d1e0')
                    plt.title("MFCCs")
                    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig2)
                with col2:
                    fig3 = plt.figure(figsize=(10, 2.5))
                    fig3.set_facecolor('#d1d1e0')
                    plt.title("Mel-log-spectrogram")
                    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig3)

            if model_type == "mfccs":
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown('<p class="panel-title">Predictions</p>', unsafe_allow_html=True)
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    if pred is None:
                        mfccs_input = prepare_mfcc_input(path, model)
                        pred = model.predict(mfccs_input, verbose=0)[0]
                        pred = remap_pred_to_ui(pred)

                    with col1:
                        if em3:
                            pos, neu, neg = sentiment_scores_from_ui(pred)
                            data3 = np.array([pos, neu, neg])
                            txt = "MFCCs\n" + get_title(data3, CAT3)
                            fig = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig, predictions=data3, categories=CAT3,
                                               title=txt, colors=COLORS)
                            # plot_polar(fig, predictions=data3, categories=CAT3,
                            # title=txt, colors=COLORS)
                            st.write(fig)
                    with col2:
                        if em6:
                            txt = "MFCCs\n" + get_title(pred, CAT6)
                            fig2 = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig2, predictions=pred, categories=CAT6,
                                               title=txt, colors=COLORS)
                            # plot_polar(fig2, predictions=pred, categories=CAT6,
                            #            title=txt, colors=COLORS)
                            st.write(fig2)
                    with col3:
                        if gender:
                            with st.spinner('Wait for it...'):
                                if not os.path.exists(GENDER_MODEL_PATH):
                                    st.warning(
                                        "Gender model not found. "
                                        "Train it with `python train_from_scratch.py`."
                                    )
                                    gmodel = None
                                else:
                                    gmodel = load_model(GENDER_MODEL_PATH)
                                if gmodel is not None:
                                    gmfccs = prepare_mfcc_input(path, gmodel)
                                    gpred = gmodel.predict(gmfccs)[0]
                                    gdict = [["female", "woman.png"], ["male", "man.png"]]
                                    ind = gpred.argmax()
                                    txt = "Predicted gender: " + gdict[ind][0]
                                    img = Image.open("images/" + gdict[ind][1])

                                    fig4 = plt.figure(figsize=(5, 5))
                                    fig4.set_facecolor('#d1d1e0')
                                    plt.title(txt)
                                    plt.imshow(img)
                                    plt.axis("off")
                                    st.write(fig4)

            # if model_type == "mel-specs":
            # st.markdown("## Predictions")
            # st.warning("The model in test mode. It may not be working properly.")
            # if st.checkbox("I'm OK with it"):
            #     try:
            #         with st.spinner("Wait... It can take some time"):
            #             global tmodel
            #             tmodel = load_model_cache("tmodel_all.h5")
            #             fig, tpred = plot_melspec(path, tmodel)
            #         col1, col2, col3 = st.columns(3)
            #         with col1:
            #             st.markdown("### Emotional spectrum")
            #             dimg = Image.open("images/spectrum.png")
            #             st.image(dimg, use_column_width=True)
            #         with col2:
            #             fig_, tpred_ = plot_melspec(path=path,
            #                                         tmodel=tmodel,
            #                                         three=True)
            #             st.write(fig_, use_column_width=True)
            #         with col3:
            #             st.write(fig, use_column_width=True)
            #     except Exception as e:
            #         st.error(f"Error {e}, model is not loaded")


    elif website_menu == "Project description":
        import pandas as pd
        import plotly.express as px
        st.title("Project description")
        st.subheader("GitHub")
        link = '[GitHub repository of the web-application]' \
               '(https://github.com/CyberMaryVer/speech-emotion-webapp)'
        st.markdown(link, unsafe_allow_html=True)

        st.subheader("Theory")
        link = '[Theory behind - Medium article]' \
               '(https://talbaram3192.medium.com/classifying-emotions-using-audio-recordings-and-python-434e748a95eb)'
        st.markdown(link + ":clap::clap::clap: Tal!", unsafe_allow_html=True)
        with st.expander("See Wikipedia definition"):
            components.iframe("https://en.wikipedia.org/wiki/Emotion_recognition",
                              height=320, scrolling=True)

        st.subheader("Dataset")
        txt = """
            This web-application is a part of the final **Data Mining** project for **ITC Fellow Program 2020**. 

            Datasets used in this project
            * Crowd-sourced Emotional Mutimodal Actors Dataset (**Crema-D**)
            * Ryerson Audio-Visual Database of Emotional Speech and Song (**Ravdess**)
            * Surrey Audio-Visual Expressed Emotion (**Savee**)
            * Toronto emotional speech set (**Tess**)    
            """
        st.markdown(txt, unsafe_allow_html=True)

        df = pd.read_csv("df_audio.csv")
        fig = px.violin(df, y="source", x="emotion4", color="actors", box=True, points="all", hover_data=df.columns)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("FYI")
        st.write("Since we are currently using a free tier instance of AWS, "
                 "we disabled mel-spec and ensemble models.\n\n"
                 "If you want to try them we recommend to clone our GitHub repo")
        st.code("git clone https://github.com/CyberMaryVer/speech-emotion-webapp.git", language='bash')

        st.write("After that, just uncomment the relevant sections in the app.py file "
                 "to use these models:")

    elif website_menu == "Our team":
        st.subheader("Our team")
        st.balloons()
        col1, col2 = st.columns([3, 2])
        with col1:
            st.info("maria.s.startseva@gmail.com")
            st.info("talbaram3192@gmail.com")
            st.info("asherholder123@gmail.com")
        with col2:
            liimg = Image.open("images/LI-Logo.png")
            st.image(liimg)
            st.markdown(f""":speech_balloon: [Maria Startseva](https://www.linkedin.com/in/maria-startseva)""",
                        unsafe_allow_html=True)
            st.markdown(f""":speech_balloon: [Tal Baram](https://www.linkedin.com/in/tal-baram-b00b66180)""",
                        unsafe_allow_html=True)
            st.markdown(f""":speech_balloon: [Asher Holder](https://www.linkedin.com/in/asher-holder-526a05173)""",
                        unsafe_allow_html=True)

    elif website_menu == "Leave feedback":
        st.subheader("Leave feedback")
        user_input = st.text_area("Your feedback is greatly appreciated")
        user_name = st.selectbox("Choose your personality", ["checker1", "checker2", "checker3", "checker4"])

        if st.button("Submit"):
            st.success(f"Message\n\"\"\"{user_input}\"\"\"\nwas sent")

            if user_input == "log123456" and user_name == "checker4":
                with open("log0.txt", "r", encoding="utf8") as f:
                    st.text(f.read())
            elif user_input == "feedback123456" and user_name == "checker4":
                with open("log.txt", "r", encoding="utf8") as f:
                    st.text(f.read())
            else:
                log_file(user_name + " " + user_input)
                thankimg = Image.open("images/sticky.png")
                st.image(thankimg)

    else:
        import requests
        import json

        url = 'http://api.quotable.io/random'
        if st.button("get random mood"):
            with st.container():
                col1, col2 = st.columns(2)
                n = np.random.randint(1, 1000, 1)[0]
                with col1:
                    quotes = {"Good job and almost done": "checker1",
                              "Great start!!": "checker2",
                              "Please make corrections base on the following observation": "checker3",
                              "DO NOT train with test data": "folk wisdom",
                              "good work, but no docstrings": "checker4",
                              "Well done!": "checker3",
                              "For the sake of reproducibility, I recommend setting the random seed": "checker1"}
                    if n % 5 == 0:
                        a = np.random.choice(list(quotes.keys()), 1)[0]
                        quote, author = a, quotes[a]
                    else:
                        try:
                            r = requests.get(url=url)
                            text = json.loads(r.text)
                            quote, author = text['content'], text['author']
                        except Exception as e:
                            a = np.random.choice(list(quotes.keys()), 1)[0]
                            quote, author = a, quotes[a]
                    st.markdown(f"## *{quote}*")
                    st.markdown(f"### ***{author}***")
                with col2:
                    st.image(image=f"https://picsum.photos/800/600?random={n}")


if __name__ == '__main__':
    main()

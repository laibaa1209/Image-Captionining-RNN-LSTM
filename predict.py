import os
# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import warnings
import numpy as np
import tensorflow as tf

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as eff_preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Load Config & Data ──────────────────────────────────────────
MODELS_DIR = "Models"
config = pickle.load(open(os.path.join(MODELS_DIR, "config_v4.pkl"), "rb"))
word_to_idx = pickle.load(open(os.path.join(MODELS_DIR, "word_to_idx.pkl"), "rb"))
idx_to_word = pickle.load(open(os.path.join(MODELS_DIR, "idx_to_word.pkl"), "rb"))

MAX_LEN = config['max_len']
FEAT_DIM = config['feat_dim']
CHAN_DIM = config['chan_dim']

# ── Custom Layers ──────────────────────────────────────────────
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units   = units
        self.W_feat  = tf.keras.layers.Dense(units)
        self.W_query = tf.keras.layers.Dense(units)
        self.V       = tf.keras.layers.Dense(1)

    def call(self, img_proj, query):
        # Explicitly cast to the layer's compute dtype to avoid float16/32 mismatch
        dtype  = self.compute_dtype
        img_proj = tf.cast(img_proj, dtype)
        query    = tf.cast(query, dtype)

        q_exp  = tf.expand_dims(self.W_query(query), 2)
        f_proj = tf.expand_dims(self.W_feat(img_proj), 1)
        score  = tf.squeeze(self.V(tf.nn.tanh(q_exp + f_proj)), -1)
        alpha  = tf.nn.softmax(score, axis=-1)
        
        # Ensure alpha and img_proj match before matmul
        alpha = tf.cast(alpha, dtype)
        context = tf.matmul(alpha, img_proj)
        return context, alpha

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

def masked_loss(y_true, y_pred): return 0.0 

print("Loading Captioner Model...")
model = load_model(os.path.join(MODELS_DIR, "model_v4_final.keras"), 
                   custom_objects={
                       'SpatialAttention': SpatialAttention,
                       'masked_loss': masked_loss
                   })

print("Loading CNN Encoder...")
base = EfficientNetB0(weights='imagenet', include_top=False)
cnn_model = tf.keras.Model(base.input, base.output)

# ── Prediction Logic ────────────────────────────────────────────
def extract_feature(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = eff_preprocess(img)
    img = np.expand_dims(img, 0)
    feat = cnn_model.predict(img, verbose=0)
    return feat.reshape(1, FEAT_DIM, CHAN_DIM)

def generate_caption(img_path, beam_width=3):
    feat = extract_feature(img_path)
    start = word_to_idx.get('startseq')
    end = word_to_idx.get('endseq')
    
    beams = [(0.0, [start])]
    done = []

    for _ in range(MAX_LEN - 1):
        cands = []
        for lp, seq in beams:
            if seq[-1] == end:
                done.append((lp, seq))
                continue
            
            inp = pad_sequences([seq], maxlen=MAX_LEN, padding='post')
            # The model outputs raw logits, so we apply softmax here
            res = model.predict([feat, inp], verbose=0)
            t = len(seq) - 1
            probs = tf.nn.softmax(res[0, t]).numpy()
            
            top_ids = np.argsort(probs)[-beam_width:]
            
            for next_id in top_ids:
                p = probs[next_id]
                new_lp = lp + np.log(p + 1e-12)
                cands.append((new_lp, seq + [next_id]))
        
        if not cands: break
        beams = sorted(cands, key=lambda x: x[0], reverse=True)[:beam_width]

    all_final = done + [(lp, seq) for lp, seq in beams]
    _, best = max(all_final, key=lambda x: x[0])
    
    caption = [idx_to_word[i] for i in best if i not in [start, end, 0]]
    return ' '.join(caption)

# ── Main Execution ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # ANSI Color Codes
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    GOLD  = "\033[93m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    if len(sys.argv) < 2:
        print(f"{BOLD}{GOLD}Usage:{RESET} python predict.py <image_path>")
    else:
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            print("\n" + "═"*50)
            print(f"{BOLD}{CYAN}         IMAGE CAPTIONING         {RESET}")
            print("═"*50)
            
            print(f" Analyzing image...")
            
            # Generate caption
            result = generate_caption(img_path)
            
            # Formatting the final string
            final_caption = result.strip().capitalize() + "."
            
            print("\n" + "─"*50)
            print(f" {BOLD}{GREEN}GENERATED CAPTION:{RESET}")
            print(f" {BOLD} {GOLD}{final_caption}{RESET}")
            print("─"*50 + "\n")
        else:
            print(f"\nError: Image file '{img_path}' not found.\n")

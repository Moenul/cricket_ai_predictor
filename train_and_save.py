import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ... after training ...

try:
    model_path = os.path.join(BASE_DIR, "model.pkl")
    enc_path = os.path.join(BASE_DIR, "encoders.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(enc_path, "wb") as f:
        pickle.dump({
            "bat": le_bat,
            "bowl": le_bowl,
            "venue": le_venue
        }, f)

    print(f"✅ Successfully saved model.pkl and encoders.pkl at {BASE_DIR}")
except Exception as e:
    print(f"\n❌ ERROR saving files: {e}")

import random
import numpy as np

# ── Speed Perturbation ────────────────────────────────────────────────────────

def speed_perturb(audio: np.ndarray, sr: int, rate: float | None = None) -> np.ndarray:
    """Time-stretch audio without changing pitch.

    rate > 1 → faster (compressed); rate < 1 → slower (stretched).
    Kept separate from pitch_shift so each effect is independently controllable.
    """
    import librosa
    if rate is None:
        rate = random.choice([0.90, 0.95, 1.05, 1.10])
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)


# ── Pitch Shift ───────────────────────────────────────────────────────────────

def pitch_shift(audio: np.ndarray, sr: int, n_steps: float | None = None) -> np.ndarray:
    """Shift pitch by n semitones without changing duration."""
    import librosa
    if n_steps is None:
        n_steps = random.choice([-2.0, -1.0, 1.0, 2.0])
    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=n_steps)


# ── Additive Gaussian Noise ───────────────────────────────────────────────────

def add_noise(audio: np.ndarray, snr_db: float | None = None) -> np.ndarray:
    """Add white Gaussian noise at a target SNR (dB).

    SNR range 15–30 dB keeps speech intelligible while adding realistic
    background noise at the recording-room level.
    """
    if snr_db is None:
        snr_db = random.uniform(15.0, 30.0)
    rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = rms_signal / (10 ** (snr_db / 20.0))
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_rms
    return audio + noise


# ── Main Entry Point ──────────────────────────────────────────────────────────

def augment(audio: np.ndarray, sr: int, p: float = 0.8) -> np.ndarray:
    """Randomly apply one or more augmentations to a 16 kHz float32 array.

    Each augmentation is applied with its own independent probability so
    combinations are possible (e.g. speed + noise). At least one op is
    guaranteed when p-gate is passed.

    Args:
        audio: 1-D float32 numpy array at sr Hz.
        sr:    Sampling rate (expected 16000).
        p:     Probability of applying any augmentation at all.

    Returns:
        Augmented audio clipped to [-1, 1], same dtype as input.
    """
    if random.random() > p:
        return audio

    ops = []
    if random.random() < 0.60:
        ops.append(lambda a: speed_perturb(a, sr))
    if random.random() < 0.40:
        ops.append(lambda a: pitch_shift(a, sr))
    if random.random() < 0.50:
        ops.append(lambda a: add_noise(a))

    if not ops:
        ops.append(lambda a: speed_perturb(a, sr))

    for op in ops:
        audio = op(audio)

    return np.clip(audio, -1.0, 1.0).astype(np.float32)

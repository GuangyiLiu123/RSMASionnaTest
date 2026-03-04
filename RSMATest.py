import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi, UMa, RMa
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
import sionna.phy

# ---- Basic helpers ----
def db2lin(x_db):
    return 10**(x_db/10)

def normalize_precoder(p):
    norm = tf.sqrt(tf.reduce_sum(tf.abs(p)**2, axis=-1, keepdims=True) + 1e-12)
    norm = tf.cast(norm, p.dtype)
    return p / norm

def apply_channel_estimation_error(h1, h2, nmse, seed=None):
    """Add estimation error: h_est = h + e, e ~ CN(0, nmse/2) per element."""
    if seed is not None:
        tf.random.set_seed(seed)
    sigma = tf.sqrt(nmse / 2.0)
    e1_re = tf.random.normal(tf.shape(h1), 0.0, sigma, dtype=tf.float32)
    e1_im = tf.random.normal(tf.shape(h1), 0.0, sigma, dtype=tf.float32)
    e2_re = tf.random.normal(tf.shape(h2), 0.0, sigma, dtype=tf.float32)
    e2_im = tf.random.normal(tf.shape(h2), 0.0, sigma, dtype=tf.float32)
    h1_est = h1 + tf.complex(e1_re, e1_im)
    h2_est = h2 + tf.complex(e2_re, e2_im)
    return h1_est, h2_est

def sionna_h_batch(batch_size, M, scenario="umi",
                   carrier_frequency=3.5e9,
                   fft_size=128, subcarrier_spacing=30e3,
                   pick="dc",
                   direction="downlink",
                   enable_pathloss=False,
                   enable_shadow_fading=False,
                   seed=42):

    sionna.phy.config.seed = seed

    ut_array = Antenna(polarization="single",
                       polarization_type="V",
                       antenna_pattern="omni",
                       carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=M,
                            polarization="single",
                            polarization_type="V",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    Model = {"umi": UMi, "uma": UMa, "rma": RMa}[scenario]
    channel_model = Model(carrier_frequency=carrier_frequency,
                          o2i_model="low" if scenario in ["umi","uma"] else None,
                          ut_array=ut_array,
                          bs_array=bs_array,
                          direction=direction,
                          enable_pathloss=enable_pathloss,
                          enable_shadow_fading=enable_shadow_fading)

    topology = gen_topology(batch_size, 2, scenario)
    channel_model.set_topology(*topology)

    freqs = subcarrier_frequencies(fft_size, subcarrier_spacing)
    a, tau = channel_model(num_time_samples=1, sampling_frequency=1.0)
    h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=True)

    h = tf.squeeze(h_freq)

    if len(h.shape) != 4:
        raise ValueError(f"Unexpected channel tensor shape after squeeze: {h.shape}. "
                         f"Print h_freq.shape and h.shape and adapt indexing.")

    if pick == "avg":
        h_nb = tf.reduce_mean(h, axis=-1)      # [B, 2, M]
    else:
        k0 = fft_size // 2
        h_nb = h[..., k0]                      # [B, 2, M]

    h1 = tf.cast(h_nb[:, 0, :], tf.complex64)
    h2 = tf.cast(h_nb[:, 1, :], tf.complex64)
    return h1, h2

# ---- SDMA: private streams only, 50-50 power ----
# Precoders from h1_est, h2_est; gains use true h1, h2
@tf.function(jit_compile=True)
def sdma_sum_rate(h1, h2, h1_est, h2_est, snr_db=10.0):
    snr_lin = tf.cast(db2lin(snr_db), tf.float32)
    noise_var = 1.0 / snr_lin
    p1 = normalize_precoder(h1_est)
    p2 = normalize_precoder(h2_est)
    P1 = tf.constant(0.5, tf.float32)
    P2 = tf.constant(0.5, tf.float32)

    def gain(h, p):
        hp = tf.reduce_sum(tf.math.conj(h) * p, axis=-1)
        return tf.abs(hp)**2

    g11 = gain(h1, p1); g12 = gain(h1, p2)
    g21 = gain(h2, p1); g22 = gain(h2, p2)
    sinr1 = (P1*g11) / (P2*g12 + noise_var)
    sinr2 = (P2*g22) / (P1*g21 + noise_var)
    R1 = tf.math.log(1.0 + sinr1) / tf.math.log(2.0)
    R2 = tf.math.log(1.0 + sinr2) / tf.math.log(2.0)
    return R1 + R2

# ---- Core RSMA sim ----
# Precoders from h1_est, h2_est; gains use true h1, h2
@tf.function(jit_compile=True)
def rsma_sum_rate(h1, h2, h1_est, h2_est, snr_db=10.0, Pc_frac=0.2, P1_frac=0.4, P2_frac=0.4):
    snr_lin = tf.cast(db2lin(snr_db), tf.float32)
    noise_var = 1.0 / snr_lin

    pc = normalize_precoder(h1_est + h2_est)
    p1 = normalize_precoder(h1_est)
    p2 = normalize_precoder(h2_est)

    Pc = tf.cast(Pc_frac, tf.float32)
    P1 = tf.cast(P1_frac, tf.float32)
    P2 = tf.cast(P2_frac, tf.float32)

    def gain(h, p):
        hp = tf.reduce_sum(tf.math.conj(h) * p, axis=-1)
        return tf.abs(hp)**2

    g1c = gain(h1, pc); g2c = gain(h2, pc)
    g11 = gain(h1, p1); g12 = gain(h1, p2)
    g21 = gain(h2, p1); g22 = gain(h2, p2)

    sinr_c1 = (Pc*g1c) / (P1*g11 + P2*g12 + noise_var)
    sinr_c2 = (Pc*g2c) / (P1*g21 + P2*g22 + noise_var)
    Rc1 = tf.math.log(1.0 + sinr_c1) / tf.math.log(2.0)
    Rc2 = tf.math.log(1.0 + sinr_c2) / tf.math.log(2.0)
    Rc = tf.minimum(Rc1, Rc2)

    sinr_p1 = (P1*g11) / (P2*g12 + noise_var)
    sinr_p2 = (P2*g22) / (P1*g21 + noise_var)
    Rp1 = tf.math.log(1.0 + sinr_p1) / tf.math.log(2.0)
    Rp2 = tf.math.log(1.0 + sinr_p2) / tf.math.log(2.0)
    sum_rate = Rc + Rp1 + Rp2
    return sum_rate
def main():
    batch = 10000
    M = 4
    snr_dbs = np.linspace(0, 20, 11)
    nmse = 0.1  # normalized MSE of channel estimate (10%)

    h1, h2 = sionna_h_batch(batch, M, scenario="umi", pick="dc", seed=1)
    h1_est, h2_est = apply_channel_estimation_error(h1, h2, nmse, seed=42)

    sdma_ideal = []
    sdma_ce = []
    rsma_ideal = []
    rsma_ce = []

    for snr_db in snr_dbs:
        sdma_ideal.append(sdma_sum_rate(h1, h2, h1, h2, snr_db=float(snr_db)).numpy().mean())
        sdma_ce.append(sdma_sum_rate(h1, h2, h1_est, h2_est, snr_db=float(snr_db)).numpy().mean())
        rsma_ideal.append(rsma_sum_rate(h1, h2, h1, h2, snr_db=float(snr_db), Pc_frac=0.2, P1_frac=0.4, P2_frac=0.4).numpy().mean())
        rsma_ce.append(rsma_sum_rate(h1, h2, h1_est, h2_est, snr_db=float(snr_db), Pc_frac=0.2, P1_frac=0.4, P2_frac=0.4).numpy().mean())

    plt.figure(figsize=(7, 5))
    plt.plot(snr_dbs, sdma_ideal, "o-", label="SDMA (perfect CSI)")
    plt.plot(snr_dbs, sdma_ce, "s-", label="SDMA (channel est. errors)")
    plt.plot(snr_dbs, rsma_ideal, "^-", label="RSMA (perfect CSI)")
    plt.plot(snr_dbs, rsma_ce, "d-", label="RSMA (channel est. errors)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sum rate (bits/s/Hz)")
    plt.legend()
    plt.title(f"SDMA vs RSMA: effect of channel estimation (NMSE = {nmse})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rsma_sdma_channel_estimation.png", dpi=200)
    plt.close()
    print("Wrote rsma_sdma_channel_estimation.png")

if __name__ == "__main__":
    main()
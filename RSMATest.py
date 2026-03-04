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

# ---- Core RSMA sim ----
@tf.function(jit_compile=True)
def rsma_rates_mc_from_h(h1, h2, snr_db=10.0, Pc_frac=0.2, P1_frac=0.4, P2_frac=0.4):
    snr_lin = tf.cast(db2lin(snr_db), tf.float32)
    noise_var = 1.0 / snr_lin

    pc = normalize_precoder(h1 + h2)
    p1 = normalize_precoder(h1)
    p2 = normalize_precoder(h2)

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

    return Rc, Rp1, Rp2, sinr_c1, sinr_c2, sinr_p1, sinr_p2
def main():
    batch = 20000
    M = 4
    snr_db = 10.0

    h1, h2 = sionna_h_batch(batch, M, scenario="umi", pick="dc", seed=1)
    print("h1 shape:", h1.shape, "dtype:", h1.dtype)
    print("h2 shape:", h2.shape, "dtype:", h2.dtype)

    Rc, Rp1, Rp2, sc1, sc2, sp1, sp2 = rsma_rates_mc_from_h(
        h1, h2, snr_db=snr_db, Pc_frac=0.2, P1_frac=0.4, P2_frac=0.4
    )

    Rc_np = Rc.numpy()
    Rp1_np = Rp1.numpy()
    Rp2_np = Rp2.numpy()

    print("Mean Rc:", Rc_np.mean())
    print("Mean Rp1:", Rp1_np.mean())
    print("Mean Rp2:", Rp2_np.mean())
    print("Mean sum-rate:", (Rc_np + Rp1_np + Rp2_np).mean())

    # quick histogram
    plt.figure()
    plt.hist(Rc_np, bins=80, alpha=0.6, label="Rc")
    plt.hist(Rp1_np, bins=80, alpha=0.6, label="Rp1")
    plt.hist(Rp2_np, bins=80, alpha=0.6, label="Rp2")
    plt.legend()
    plt.title(f"Rate samples @ SNR={snr_db} dB")
    plt.tight_layout()
    plt.savefig("rsma_quick_check.png", dpi=200)
    plt.close()
    

    print("Wrote rsma_quick_check.png")

if __name__ == "__main__":
    main()
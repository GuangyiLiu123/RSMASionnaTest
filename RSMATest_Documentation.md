# RSMATest.py — Technical Documentation

**SDMA vs RSMA Downlink Simulation with Channel Estimation Errors**

---

## 1. Overview

This script simulates a two-user downlink MIMO system comparing **Spatial Division Multiple Access (SDMA)** and **Rate-Splitting Multiple Access (RSMA)**. It evaluates sum rates under perfect channel state information (CSI) and under channel estimation errors, using realistic channel models from Sionna.

**Output:** A single PNG plot with four curves: SDMA (perfect CSI), SDMA (channel est. errors), RSMA (perfect CSI), RSMA (channel est. errors).

---

## 2. Architecture and Code Flow

### 2.1 High-Level Flow

```
main()
    │
    ├── sionna_h_batch()          → Generate true channels h1, h2
    ├── apply_channel_estimation_error() → Corrupt to get h1_est, h2_est
    │
    └── For each SNR point:
            ├── sdma_sum_rate(h1, h2, h1, h2)       → SDMA, perfect CSI
            ├── sdma_sum_rate(h1, h2, h1_est, h2_est) → SDMA, with CE errors
            ├── rsma_sum_rate(h1, h2, h1, h2, ...)  → RSMA, perfect CSI
            └── rsma_sum_rate(h1, h2, h1_est, h2_est, ...) → RSMA, with CE errors
```

### 2.2 Key Design Choice: Mismatched Precoding

- **Precoders** are designed from estimated channels (h1_est, h2_est).
- **Gains** (and thus SINRs/rates) are computed using the true channels (h1, h2).
- This models the real scenario: the BS uses imperfect estimates to design beamformers, but the actual received signal depends on the true channel.

---

## 3. Component-by-Component Breakdown

### 3.1 Helper Functions

| Function | Purpose |
|----------|---------|
| `db2lin(x_db)` | Converts SNR from dB to linear: 10^(x_db/10) |
| `normalize_precoder(p)` | Ensures precoder has unit norm: p / ||p|| |
| `apply_channel_estimation_error(h1, h2, nmse, seed)` | Adds additive error: ĥ = h + e, e ~ CN(0, NMSE/2) per element |

### 3.2 Channel Generation: `sionna_h_batch()`

**Role of Sionna:** This function relies entirely on Sionna for channel generation.

**What it does:**
1. Configures UE and BS antenna arrays (UE: omnidirectional; BS: 1×M uniform linear array).
2. Instantiates a 3GPP TR 38.901 channel model (UMi, UMa, or RMa).
3. Generates random UE positions and sets topology.
4. Runs the channel model to get channel impulse response (CIR) coefficients (a, τ).
5. Converts CIR to OFDM frequency-domain channel via `cir_to_ofdm_channel()`.
6. Extracts narrowband channels at DC subcarrier (or averages) for both users.
7. Returns h1, h2 as complex tensors of shape [batch, M].

**Sionna imports used:**
- `Antenna`, `AntennaArray` — antenna configuration
- `UMi`, `UMa`, `RMa` — 3GPP TR 38.901 scenarios
- `gen_single_sector_topology` — UE placement
- `subcarrier_frequencies`, `cir_to_ofdm_channel` — OFDM channel conversion

### 3.3 SDMA: `sdma_sum_rate(h1, h2, h1_est, h2_est, snr_db)`

**Model:** Each user gets only a private stream. No common message.

- **Precoders:** p1 = h1_est / ||h1_est||, p2 = h2_est / ||h2_est|| (match-filter / conjugate beamforming)
- **Power split:** P1 = P2 = 0.5
- **Rates:** R1 = log2(1 + SINR1), R2 = log2(1 + SINR2), where each user treats the other’s stream as interference.

### 3.4 RSMA: `rsma_sum_rate(h1, h2, h1_est, h2_est, snr_db, Pc_frac, P1_frac, P2_frac)`

**Model:** Message splitting with a common and two private streams.

- **Streams:**
  - **Common:** sc, precoder pc = (h1_est + h2_est) / ||h1_est + h2_est||
  - **Private:** s1, s2 with p1, p2 as in SDMA
- **Power split:** Pc (default 0.2), P1 (0.4), P2 (0.4)
- **Decoding:** Both users decode sc first, then subtract and decode their private stream.
- **Common rate:** Rc = min(Rc1, Rc2) (limited by the weaker user).
- **Sum rate:** Rc + Rp1 + Rp2.

---

## 4. Role of Sionna

Sionna provides:

1. **Realistic channel models** — 3GPP TR 38.901 UMi/UMa/RMa with proper path loss, delay spread, and spatial correlation.
2. **OFDM support** — Conversion from CIR to frequency-domain channels.
3. **Reproducibility** — Seeded random generation for channels and topology.
4. **Flexibility** — Different carrier frequencies, antenna patterns, scenarios.

**What this script does not use from Sionna:** Full link-level simulation, LDPC/polar coding, modulation, detectors, etc. It uses Sionna only for channel generation and computes rates analytically via the Shannon formula.

---

## 5. What Can Be Tweaked or Extended

### 5.1 Channel and Scenario

| Parameter | Location | Effect |
|-----------|----------|--------|
| `scenario` | `sionna_h_batch()` | "umi", "uma", "rma" — changes propagation model |
| `carrier_frequency` | `sionna_h_batch()` | Default 3.5 GHz |
| `M` (antennas) | `main()` | Number of BS antennas (default 4) |
| `batch` | `main()` | Number of channel realizations (default 10000) |
| `pick` | `sionna_h_batch()` | "dc" (center subcarrier) or "avg" (average over subcarriers) |
| `enable_pathloss`, `enable_shadow_fading` | `sionna_h_batch()` | Turn path loss / shadow fading on/off |

### 5.2 Channel Estimation

| Parameter | Location | Effect |
|-----------|----------|--------|
| `nmse` | `main()`, `apply_channel_estimation_error()` | Normalized MSE of estimate (default 0.1 = 10%) |
| Error model | `apply_channel_estimation_error()` | Currently additive CN(0, NMSE/2). Could be replaced with pilot-based or SNR-dependent NMSE. |

### 5.3 RSMA Power Allocation

| Parameter | Location | Effect |
|-----------|----------|--------|
| `Pc_frac`, `P1_frac`, `P2_frac` | `rsma_sum_rate()`, `main()` | Power split across common and private streams. Optimizing these can improve sum rate. |

### 5.4 SNR and Plotting

| Parameter | Location | Effect |
|-----------|----------|--------|
| `snr_dbs` | `main()` | SNR range (default 0–20 dB) and resolution |

### 5.5 Extensions to Consider

1. **More users (K > 2)** — Generalize SDMA/RSMA to K users; RSMA common stream structure becomes more complex.
2. **Different precoding** — ZF, MMSE, or optimal RSMA precoders instead of match-filter.
3. **Joint power and precoding optimization** — WMMSE or similar for RSMA.
4. **Phase noise** — Reintroduce per-antenna phase noise as an impairment.
5. **Non-Gaussian noise** — Impulsive or heavy-tailed noise models (requires Monte Carlo or different capacity bounds).
6. **Correlated channels** — Use Sionna’s capability for antenna correlation.
7. **Pilot-based CE model** — NMSE ∝ 1/(SNR_pilot × num_pilots) for a more physical CE model.
8. **Different metrics** — Per-user rates, fairness (e.g., max-min), or outage probability.

---

## 6. Dependencies

- **NumPy** — Array operations, linspace
- **TensorFlow** — Channel tensors, rate computation, JIT compilation
- **Matplotlib** — Plotting
- **Sionna** — Channel models (TR 38.901, OFDM, topology)

---

## 7. Running the Script

```
python RSMATest.py
```

**Output:** `rsma_sdma_channel_estimation.png` — Sum rate vs SNR for four configurations.

---

## 8. File Summary

| File | Purpose |
|------|---------|
| `RSMATest.py` | Main simulation script |
| `RSMATest_Documentation.md` | This documentation (Markdown source) |
| `rsma_sdma_channel_estimation.png` | Generated comparison plot |

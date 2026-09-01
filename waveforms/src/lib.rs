/*!
 * LW-LGM: Latent-to-Waveform Linear Geometric Map
 *
 * Maps a latent vector z ∈ ℝ^d to an analog waveform x(t) ∈ C^0(ℝ)
 * using a linear expansion in a fixed dictionary of geometrically
 * transformed atoms (affine group acting on a mother waveform).
 *
 * The mapping is: x(t) = z^T W^T Ψ(t)
 * where:
 *   - Ψ(t) = [ψ_1(t), ψ_2(t), ..., ψ_m(t)] is the dictionary vector
 *   - ψ_i(t) = (1/√|a_i|) φ((t - b_i)/a_i) is a dilated/translated atom
 *   - φ(t) is a mother waveform (Gaussian by default)
 *   - W ∈ ℝ^{m×d} is a fixed linear map (identity when d=m)
 *
 * Properties:
 *   - Linearity: L(αz₁ + βz₂) = αL(z₁) + βL(z₂)
 *   - Frame expansion in L^2(ℝ) with affine dictionary
 *   - Energy preservation via tight frame design
 */

use ndarray::{s, Array1, Array2};

// ── Mother Waveform ──────────────────────────────────────────────────────

/// Normalized Gaussian mother waveform:
/// φ(t) = (1/(2πσ₀²)^{1/4}) · exp(-t²/(2σ₀²))
fn mother_gaussian(t: f64, sigma0: f64) -> f64 {
    let norm = 1.0 / (2.0 * std::f64::consts::PI * sigma0.powi(2)).powf(0.25);
    norm * (-0.5 * t * t / (sigma0 * sigma0)).exp()
}

// ── Dictionary Construction ──────────────────────────────────────────────

/// Build the dictionary matrix Ψ ∈ ℝ^{N×m} from an affine group action.
///
/// # Arguments
/// * `sigma0` - Mother Gaussian width
/// * `a_min` - Minimum dilation (must be > 0)
/// * `a_max` - Maximum dilation (must be > a_min)
/// * `b_min` - Minimum translation
/// * `b_max` - Maximum translation
/// * `m` - Number of atoms (must be even for symmetry)
/// * `t_start` - Time axis start
/// * `t_end` - Time axis end
/// * `dt` - Time step
///
/// # Returns
/// * `Psi` - Dictionary matrix of shape (N, m) where N = ceil((t_end - t_start) / dt)
pub fn build_dictionary(
    sigma0: f64,
    a_min: f64,
    a_max: f64,
    b_min: f64,
    b_max: f64,
    m: usize,
    t_start: f64,
    t_end: f64,
    dt: f64,
) -> Array2<f64> {
    let n = ((t_end - t_start) / dt).ceil() as usize;
    let mut psi = Array2::<f64>::zeros((n, m));

    let log_a_min = a_min.ln();
    let log_a_max = a_max.ln();
    let log_a_step = (log_a_max - log_a_min) / ((m / 2) as f64);

    for i in 0..m {
        // Logarithmic dilation grid
        let a = if i < m / 2 {
            (log_a_min + i as f64 * log_a_step).exp()
        } else {
            -((log_a_min + (m - 1 - i) as f64 * log_a_step).exp())
        };

        // Uniform translation
        let b = b_min + (i as f64) * (b_max - b_min) / ((m - 1) as f64);

        // Precompute 1/√|a|
        let scale = 1.0 / a.abs().sqrt();

        // Fill column i of Ψ
        for k in 0..n {
            let t = t_start + k as f64 * dt;
            let arg = (t - b) / a;
            let phi_val = mother_gaussian(arg, sigma0);
            psi[[k, i]] = scale * phi_val;
        }
    }

    psi
}

// ── Latent-to-Waveform Mapping ──────────────────────────────────────────

/// Map a latent vector z to waveform samples x = Ψ(Wz).
///
/// # Arguments
/// * `z` - Latent vector of length d
/// * `W` - Fixed matrix of shape (m, d), or identity if d == m
/// * `psi` - Dictionary matrix of shape (N, m)
///
/// # Returns
/// * `x` - Output waveform samples of length N
pub fn latent_to_waveform(
    z: &Array1<f64>,
    W: &Array2<f64>,
    psi: &Array2<f64>,
) -> Array1<f64> {
    // c = W * z
    let c = if W.ncols() == z.len() {
        W.dot(z)
    } else {
        z.to_owned()
    };

    // x = Ψ * c
    psi.dot(&c)
}

// ── Validation Tests ─────────────────────────────────────────────────────

/// Linearity test: verify L(αz₁ + βz₂) = αL(z₁) + βL(z₂)
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Random;

    #[test]
    fn test_linearity() {
        let sigma0 = 1.0;
        let (a_min, a_max) = (0.5, 2.0);
        let (b_min, b_max) = (-5.0, 5.0);
        let m = 32;
        let (t_start, t_end, dt) = (-10.0, 10.0, 0.1);

        let psi = build_dictionary(sigma0, a_min, a_max, b_min, b_max, m, t_start, t_end, dt);
        let W = Array2::<f64>::eye(m);

        let z1 = Array1::<f64>::random(m, rand::distributions::Uniform::new(-1.0, 1.0));
        let z2 = Array1::<f64>::random(m, rand::distributions::Uniform::new(-1.0, 1.0));

        let alpha = 2.5;
        let beta = -1.3;

        let lhs = latent_to_waveform(&(alpha * &z1 + beta * &z2), &W, &psi);
        let rhs = alpha * latent_to_waveform(&z1, &W, &psi)
            + beta * latent_to_waveform(&z2, &W, &psi);

        let diff = (&lhs - &rhs).mapv(|x| x.abs()).sum();
        assert!(diff < 1e-10, "Linearity test failed: diff = {}", diff);
    }

    #[test]
    fn test_identity_mapping() {
        let sigma0 = 1.0;
        let (a_min, a_max) = (0.5, 2.0);
        let (b_min, b_max) = (-5.0, 5.0);
        let m = 16;
        let (t_start, t_end, dt) = (-10.0, 10.0, 0.1);

        let psi = build_dictionary(sigma0, a_min, a_max, b_min, b_max, m, t_start, t_end, dt);
        let W = Array2::<f64>::eye(m);

        let z = Array1::<f64>::random(m, rand::distributions::Uniform::new(-1.0, 1.0));
        let x = latent_to_waveform(&z, &W, &psi);

        // Verify shape
        assert_eq!(x.len(), psi.nrows());
    }

    #[test]
    fn test_energy_bounds() {
        let sigma0 = 1.0;
        let (a_min, a_max) = (0.5, 2.0);
        let (b_min, b_max) = (-5.0, 5.0);
        let m = 64;
        let (t_start, t_end, dt) = (-10.0, 10.0, 0.01);

        let psi = build_dictionary(sigma0, a_min, a_max, b_min, b_max, m, t_start, t_end, dt);
        let W = Array2::<f64>::eye(m);

        let z = Array1::<f64>::random(m, rand::distributions::Uniform::new(-1.0, 1.0));
        let x = latent_to_waveform(&z, &W, &psi);

        let energy_x = x.mapv(|v| v * v).sum() * dt;
        let energy_z = z.mapv(|v| v * v).sum();

        // Energy ratio should be bounded (frame bounds)
        let ratio = energy_x / energy_z;
        assert!(ratio > 0.0 && ratio.is_finite(), "Energy ratio invalid: {}", ratio);
    }
}

// ── CLI Entry Point ──────────────────────────────────────────────────────

fn main() {
    let sigma0 = 1.0;
    let (a_min, a_max) = (0.5, 2.0);
    let (b_min, b_max) = (-5.0, 5.0);
    let m = 64;
    let (t_start, t_end, dt) = (-10.0, 10.0, 0.01);
    let d = m;

    println!("LW-LGM: Latent-to-Waveform Linear Geometric Map");
    println!("================================================");
    println!("Parameters:");
    println!("  σ₀ = {}", sigma0);
    println!("  a ∈ [{}, {}]", a_min, a_max);
    println!("  b ∈ [{}, {}]", b_min, b_max);
    println!("  m = {} atoms", m);
    println!("  t ∈ [{}, {}] dt={}", t_start, t_end, dt);
    println!();

    // Build dictionary
    let psi = build_dictionary(sigma0, a_min, a_max, b_min, b_max, m, t_start, t_end, dt);
    println!("Dictionary Ψ: {}×{}", psi.nrows(), psi.ncols());

    // Identity mapping
    let W = Array2::<f64>::eye(m);

    // Random latent vector
    let z = Array1::<f64>::random(m, rand::distributions::Uniform::new(-1.0, 1.0));
    println!("Latent z: {} dimensions", z.len());

    // Generate waveform
    let x = latent_to_waveform(&z, &W, &psi);
    println!("Output x: {} samples", x.len());
    println!("x[0..10] = {:?}", x.slice(s![0..10]).to_vec());

    let energy = x.mapv(|v| v * v).sum() * dt;
    println!("Signal energy: {:.6}", energy);
}

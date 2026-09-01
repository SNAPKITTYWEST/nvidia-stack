use lw_lgm::{build_dictionary, latent_to_waveform};
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Random;

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
    println!("x[0..10] = {:?}", x.slice(ndarray::s![0..10]).to_vec());

    let energy = x.mapv(|v| v * v).sum() * dt;
    println!("Signal energy: {:.6}", energy);
}

use ndarray::*;
use ndarray_linalg::*;

use rand::Rng;

const NUM_PLAYERS: usize = 35;
const NOISE_RANGE: f64 = 0.2;
const CHANCE_SKIP_ROW: f64 = 0.;

// Generate a vector of randomized rankings to seed the matrix with instead of asking questions
// Results are normalized to the last entry, so the last entry will always be 1
fn gen_seed_rankings() -> Vec<f64>{
    let mut seed_rankings: Vec<f64> = Vec::new();
    for _ in 0..NUM_PLAYERS{
        seed_rankings.push(rand::random::<f64>());
    }

    // Normalize seed rankings
    for i in 0..NUM_PLAYERS{
        seed_rankings[i] = seed_rankings[i] / seed_rankings[NUM_PLAYERS - 1];
    }
    seed_rankings
}

// Given a seed vector of rankings of players, generate a test linear system
fn gen_lin_sys_from_seed_rankings(seed_rankings: &Vec<f64>) -> (Array2<f64>, Array1<f64>){
    // Generate A matrix
    let mut a: Array2<f64> = arr2(&[[]]);
    // First row
    for _ in 0..NUM_PLAYERS{
        a.push_column(ArrayView::from(&[0.])).unwrap();
    }
    a[[0, NUM_PLAYERS - 1]] = 1.;

    // Remaining rows
    for p1 in 0..NUM_PLAYERS{
        for p2 in (p1 + 1)..NUM_PLAYERS{
            // Skip rows occasionally
            if rand::random::<f64>() < CHANCE_SKIP_ROW{
                continue;
            }

            let mut next_row: [f64; NUM_PLAYERS] = [0.; NUM_PLAYERS];
            // Generate random noise on the accuracy of the rankings
            let noise: f64 = rand::thread_rng().gen_range((1. - NOISE_RANGE/2.)..(1. + NOISE_RANGE/2.));
            next_row[p1] = 1.;
            next_row[p2] = -seed_rankings[p1]/seed_rankings[p2] * noise; // noisy seed ranking
            a.push_row(ArrayView::from(&next_row)).unwrap();
        }
    }

    // Generate b vector
    let mut b: Vec<f64> = vec![0.; a.dim().0];
    b[0] = seed_rankings[NUM_PLAYERS - 1];
    let b: Array1<f64> = arr1(&b);

    (a, b)
}

// Solve a linear system using a least squares regression
fn least_squares_regression(a: Array2<f64>, b: Array1<f64>) -> Vec<f64>{
    // Perform linear algebra to do least squares regression
    let ata = a.t().dot(&a);
    let atb = a.t().dot(&b);
    let mut sol = ata.solve(&atb).unwrap().to_vec();

    // Normalize solution vector
    for i in 0..NUM_PLAYERS{
        sol[i] = sol[i] / sol[NUM_PLAYERS - 1];
    }
    sol
}

// Given the seed ranks and the computed result ranks, compute a vector or %error on the computation
fn compute_err_from_seed(seed_rankings: &Vec<f64>, sol: &Vec<f64>) -> Vec<f64>{
    let mut err: Vec<f64> = Vec::new();
    for i in 0..NUM_PLAYERS{
        err.push(((seed_rankings[i] - sol[i])/seed_rankings[i]).abs());
    }
    err
}

// Compute and log max and mean errors on all the players
fn log_err_stats(err: &Vec<f64>){
    let mut max_err = err[0];
    let mut sum = 0.;
    for er in err{
        sum += er;
        if *er > max_err {
            max_err = *er;
        }
    }
    let mean = sum / err.len() as f64;
    println!("Max: {:.3} Mean: {:.3}", max_err, mean);
}

fn main() {
    let seed_rankings = gen_seed_rankings();
    let (a, b) = gen_lin_sys_from_seed_rankings(&seed_rankings);
    let sol = least_squares_regression(a, b);
    let err = compute_err_from_seed(&seed_rankings, &sol);
    log_err_stats(&err);
}

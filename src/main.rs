use ndarray::*;
use ndarray_linalg::*;

use rand::Rng;

const NUM_PLAYERS: usize = 35;
const NOISE_RANGE: f64 = 0.05;

fn main() {
    // Generate a series of players with random rating in range [0,1)
    let mut seed_rankings: Vec<f64> = Vec::new();
    for _ in 0..NUM_PLAYERS{
        seed_rankings.push(rand::random::<f64>());
    }
    println!("{:?}", seed_rankings);

    // Generate A matrix
    let mut a: Array2<f64> = arr2(&[[]]);
    for _ in 0..NUM_PLAYERS{
        a.push_column(ArrayView::from(&[0.])).unwrap();
    }
    a[[0, NUM_PLAYERS - 1]] = 1.;

    for p1 in 0..NUM_PLAYERS{
        for p2 in (p1 + 1)..NUM_PLAYERS{
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

    // Perform linear algebra to do least squares regression
    let ata = a.t().dot(&a);
    let atb = a.t().dot(&b);
    let _x = ata.solve(&atb).unwrap().to_vec();

    println!("Solution: {:?}", _x);
}

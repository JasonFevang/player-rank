use ndarray::*;
use ndarray_linalg::*;

fn main() {
    // A = 5.9D
    // A = 3.1C
    // A = 1.55B
    // B = 2.1C
    // B = 3.9D
    // C = 1.9D
    // D = 1

    let a: Array2<f64> = arr2(&[
                   [1., 0., 0., -5.9],
                   [1., 0., -3.1, 0.],
                   [1., -1.55, 0., 0.],
                   [0., 1., -2.1, 0.],
                   [0., 1., 0., -3.9],
                   [0., 0., 1., -1.9],
                   [0., 0., 0., 1.],
    ]);
    let b: Array1<f64> = arr1(&[0., 0., 0., 0., 0., 0., 1.]);

    let ata = a.t().dot(&a);
    let atb = a.t().dot(&b);

    let _x = ata.solve(&atb).unwrap();
    println!("Solution: {}", _x);
}

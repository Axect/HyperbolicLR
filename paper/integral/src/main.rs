use rugfield::{grf, Kernel};
use peroxide::fuga::*;
use peroxide::fuga::anyhow::Result;
use rayon::prelude::*;
use indicatif::{ProgressBar, ParallelProgressIterator};

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let n = 10000;

    println!("Generate dataset...");
    let ds = Dataset::generate(n, 0.8)?;
    ds.write_parquet()?;
    println!("Generate dataset complete");

    Ok(())
}

// ┌─────────────────────────────────────────────────────────┐
//  Dataset
// └─────────────────────────────────────────────────────────┘
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Dataset {
    pub train_u: Matrix,
    pub train_y: Matrix,
    pub train_Gu: Matrix,
    pub val_u: Matrix,
    pub val_y: Matrix,
    pub val_Gu: Matrix,
}

impl Dataset {
    #[allow(non_snake_case)]
    pub fn generate(n: usize, f_train: f64) -> Result<Self> {
        // Gaussian Random Field generation
        let x_min = 0f64;
        let x_max = 1f64;
        let x_len = 1000;
        let u_l = Uniform(0.1, 0.4);
        let l_samples = u_l.sample(n);

        let grf_vec = l_samples
            .par_iter()
            .progress_with(ProgressBar::new(n as u64))
            .map(|&l| grf(x_len, Kernel::SquaredExponential(l)))
            .collect::<Vec<_>>();

        // Normalize
        let grf_max_vec = grf_vec
            .iter()
            .map(|grf| grf.max())
            .collect::<Vec<_>>();

        let grf_min_vec = grf_vec
            .iter()
            .map(|grf| grf.min())
            .collect::<Vec<_>>();

        let grf_max_mean = grf_max_vec.mean();
        let grf_max_std  = grf_max_vec.sd();
        let grf_max_sigma = grf_max_mean + 1.0 * grf_max_std;

        let grf_min_mean = grf_min_vec.mean();
        let grf_min_std  = grf_min_vec.sd();
        let grf_min_sigma = grf_min_mean - 1.0 * grf_min_std;

        let u_vec = grf_vec.iter()
            .map(|grf| {
                grf.fmap(|x| (x - grf_min_sigma) / (grf_max_sigma - grf_min_sigma) * 2f64 - 1f64)
            }).collect::<Vec<_>>();

        let x = linspace_with_precision(x_min, x_max, x_len, 3);
        let y_range = linspace_with_precision(0, 1, 100, 3);
        
        // Integration
        let Gu_vec = u_vec
            .par_iter()
            .progress_with(ProgressBar::new(n as u64))
            .map(|grf| {
                let cs = cubic_hermite_spline(&x, grf, Quadratic).unwrap();
                y_range.fmap(|y| cs.integrate((x_min, y)))
            })
            .collect::<Vec<_>>();

        // choose 100 points from 0 to 1
        let u_vec = u_vec.iter().map(|x| x.iter().step_by(10).cloned().collect::<Vec<_>>()).collect::<Vec<_>>();
        let y_vec = vec![y_range.clone(); n];

        let n_train = (n as f64 * f_train).round() as usize;
        let n_val = n - n_train;

        println!("n_train: {}", n_train);
        println!("n_val: {}", n_val);

        let train_u = u_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_y = y_vec.iter().take(n_train).cloned().collect::<Vec<_>>();
        let train_Gu = Gu_vec.iter().take(n_train).cloned().collect::<Vec<_>>();

        let val_u = u_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();
        let val_y = y_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();
        let val_Gu = Gu_vec.iter().skip(n_train).cloned().collect::<Vec<_>>();

        Ok(Self {
            train_u: py_matrix(train_u),
            train_y: py_matrix(train_y),
            train_Gu: py_matrix(train_Gu),
            val_u: py_matrix(val_u),
            val_y: py_matrix(val_y),
            val_Gu: py_matrix(val_Gu),
        })
    }

    #[allow(non_snake_case)]
    pub fn train_set(&self) -> (Matrix, Matrix, Matrix) {
        (
            self.train_u.clone(),
            self.train_y.clone(),
            self.train_Gu.clone()
        )
    }

    #[allow(non_snake_case)]
    pub fn val_set(&self) -> (Matrix, Matrix, Matrix) {
        (
            self.val_u.clone(),
            self.val_y.clone(),
            self.val_Gu.clone()
        )
    }

    #[allow(non_snake_case)]
    pub fn write_parquet(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data_folder = "../data";
        if !std::path::Path::new(data_folder).exists() {
            std::fs::create_dir(data_folder)?;
        }

        let (train_u, train_y, train_Gu) = self.train_set();
        let (val_u, val_y, val_Gu) = self.val_set();

        let mut df = DataFrame::new(vec![]);
        df.push("train_u", Series::new(train_u.data));
        df.push("train_y", Series::new(train_y.data));
        df.push("train_Gu", Series::new(train_Gu.data));

        
        let train_path = format!("{}/train.parquet", data_folder);
        df.write_parquet(&train_path, UNCOMPRESSED)?;

        let mut df = DataFrame::new(vec![]);
        df.push("val_u", Series::new(val_u.data));
        df.push("val_y", Series::new(val_y.data));
        df.push("val_Gu", Series::new(val_Gu.data));

        let val_path = format!("{}/val.parquet", data_folder);
        df.write_parquet(&val_path, UNCOMPRESSED)?;

        Ok(())
    }
}

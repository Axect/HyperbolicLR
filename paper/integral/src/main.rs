use indicatif::{ParallelProgressIterator, ProgressBar};
use peroxide::fuga::*;
use rand_08::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use rugfield::{grf_with_rng, Kernel};
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Config {
    out: PathBuf,
    train_count: usize,
    validation_count: usize,
    test_count: usize,
    train_seed: u64,
    validation_seed: u64,
    test_seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            out: PathBuf::from("../data"),
            train_count: 8_000,
            validation_count: 1_000,
            test_count: 1_000,
            train_seed: 1,
            validation_seed: 2,
            test_seed: 3,
        }
    }
}

fn parse_args() -> std::result::Result<Config, Box<dyn Error>> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--out" => config.out = PathBuf::from(value),
            "--train-count" => config.train_count = value.parse()?,
            "--validation-count" => config.validation_count = value.parse()?,
            "--test-count" => config.test_count = value.parse()?,
            "--train-seed" => config.train_seed = value.parse()?,
            "--validation-seed" => config.validation_seed = value.parse()?,
            "--test-seed" => config.test_seed = value.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if config.train_count == 0 || config.validation_count == 0 || config.test_count == 0 {
        return Err("all split counts must be positive".into());
    }
    if [config.train_seed, config.validation_seed, config.test_seed]
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != 3
    {
        return Err("train, validation, and test seeds must be distinct".into());
    }
    Ok(config)
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    println!("Generate deterministic operator dataset: {config:?}");
    let dataset = Dataset::generate(&config);
    dataset.write_parquet(&config.out)?;
    dataset.write_generation_manifest(&config.out, &config)?;
    println!("Generate dataset complete");
    Ok(())
}

#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Split {
    pub u: Matrix,
    pub y: Matrix,
    pub Gu: Matrix,
}

#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Dataset {
    pub train: Split,
    pub validation: Split,
    pub test: Split,
    pub train_min_sigma: f64,
    pub train_max_sigma: f64,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn sample_seed(base_seed: u64, index: usize, stream: u64) -> u64 {
    splitmix64(base_seed ^ (index as u64).wrapping_mul(0xD6E8FEB86659FD93) ^ stream)
}

fn unit_interval(seed: u64) -> f64 {
    ((seed >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn generate_raw_grfs(count: usize, seed: u64) -> Vec<Vec<f64>> {
    (0..count)
        .into_par_iter()
        .progress_with(ProgressBar::new(count as u64))
        .map(|index| {
            let length_seed = sample_seed(seed, index, 0x4C454E475448);
            let length = 0.1 + 0.3 * unit_interval(length_seed);
            let mut grf_rng = StdRng::seed_from_u64(sample_seed(seed, index, 0x475246));
            grf_with_rng(&mut grf_rng, 1_000, Kernel::SquaredExponential(length))
        })
        .collect()
}

fn training_scale(grfs: &[Vec<f64>]) -> (f64, f64) {
    let maxima = grfs.iter().map(|grf| grf.max()).collect::<Vec<_>>();
    let minima = grfs.iter().map(|grf| grf.min()).collect::<Vec<_>>();
    let max_sigma = maxima.mean() + maxima.sd();
    let min_sigma = minima.mean() - minima.sd();
    if !min_sigma.is_finite() || !max_sigma.is_finite() || min_sigma >= max_sigma {
        panic!("invalid training normalization range: [{min_sigma}, {max_sigma}]");
    }
    (min_sigma, max_sigma)
}

#[allow(non_snake_case)]
fn build_split(raw_grfs: Vec<Vec<f64>>, min_sigma: f64, max_sigma: f64) -> Split {
    let count = raw_grfs.len();
    let normalized = raw_grfs
        .into_par_iter()
        .map(|grf| grf.fmap(|value| (value - min_sigma) / (max_sigma - min_sigma) * 2.0 - 1.0))
        .collect::<Vec<_>>();

    let x = linspace_with_precision(0.0, 1.0, 1_000, 3);
    let y_range = linspace_with_precision(0.0, 1.0, 100, 3);
    let Gu = normalized
        .par_iter()
        .progress_with(ProgressBar::new(count as u64))
        .map(|grf| {
            let spline = cubic_hermite_spline(&x, grf, Quadratic).unwrap();
            y_range.fmap(|y| spline.integrate((0.0, y)))
        })
        .collect::<Vec<_>>();
    let u = normalized
        .iter()
        .map(|values| values.iter().step_by(10).copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let y = vec![y_range; count];

    Split {
        u: py_matrix(u),
        y: py_matrix(y),
        Gu: py_matrix(Gu),
    }
}

impl Dataset {
    fn generate(config: &Config) -> Self {
        let train_raw = generate_raw_grfs(config.train_count, config.train_seed);
        let (train_min_sigma, train_max_sigma) = training_scale(&train_raw);
        let train = build_split(train_raw, train_min_sigma, train_max_sigma);
        let validation = build_split(
            generate_raw_grfs(config.validation_count, config.validation_seed),
            train_min_sigma,
            train_max_sigma,
        );
        let test = build_split(
            generate_raw_grfs(config.test_count, config.test_seed),
            train_min_sigma,
            train_max_sigma,
        );
        Self {
            train,
            validation,
            test,
            train_min_sigma,
            train_max_sigma,
        }
    }

    fn write_split(
        split: &Split,
        name: &str,
        data_folder: &Path,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let mut frame = DataFrame::new(vec![]);
        frame.push("u", Series::new(split.u.data.clone()));
        frame.push("y", Series::new(split.y.data.clone()));
        frame.push("Gu", Series::new(split.Gu.data.clone()));
        let path = data_folder.join(format!("{name}.parquet"));
        frame.write_parquet(path.to_str().unwrap(), UNCOMPRESSED)?;
        Ok(())
    }

    pub fn write_parquet(&self, data_folder: &Path) -> std::result::Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(data_folder)?;
        Self::write_split(&self.train, "train", data_folder)?;
        Self::write_split(&self.validation, "validation", data_folder)?;
        Self::write_split(&self.test, "test", data_folder)?;
        Ok(())
    }

    fn write_generation_manifest(
        &self,
        data_folder: &Path,
        config: &Config,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let payload = format!(
            concat!(
                "{{\n",
                "  \"generator\": \"paper/integral/src/main.rs\",\n",
                "  \"train_count\": {},\n",
                "  \"validation_count\": {},\n",
                "  \"test_count\": {},\n",
                "  \"train_seed\": {},\n",
                "  \"validation_seed\": {},\n",
                "  \"test_seed\": {},\n",
                "  \"train_min_sigma\": {:.17e},\n",
                "  \"train_max_sigma\": {:.17e}\n",
                "}}\n"
            ),
            config.train_count,
            config.validation_count,
            config.test_count,
            config.train_seed,
            config.validation_seed,
            config.test_seed,
            self.train_min_sigma,
            self.train_max_sigma,
        );
        std::fs::write(data_folder.join("generation.json"), payload)?;
        Ok(())
    }
}

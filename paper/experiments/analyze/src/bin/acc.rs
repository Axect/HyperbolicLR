use peroxide::fuga::*;
use dialoguer::{theme::ColorfulTheme, Select};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = vec!["CNN", "ViT"];
    let select = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Model")
        .items(&models)
        .default(0)
        .interact()
        .unwrap();
    let model = models[select];

    let schedulers = vec!["N", "P", "C", "E", "H", "EH"];
    let select = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Scheduler")
        .items(&schedulers)
        .default(0)
        .interact()
        .unwrap();
    let scheduler = schedulers[select];

    let file = format!("data/Result_CIFAR10_{}_ACC-{}.csv", model, scheduler);
    let mut df = DataFrame::read_csv(&file, ',')?;
    df.as_types(vec![F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64]);

    let epochs = [50, 100, 150, 200];
    let val_losses = epochs.iter().map(|epoch| {
        let loss: Vec<f64> = df[format!("{}{}_val_loss", scheduler, epoch).as_str()].to_vec();
        loss
    })
    .collect::<Vec<Vec<f64>>>();

    let mut diffs = zeros(4, 4);
    for i in 0 .. epochs.len() {
        for j in i+1 .. epochs.len() {
            let diff = curve_diff(&val_losses[i], &val_losses[j])?;
            diffs[(i, j)] = diff;
        }
    }

    let mut df = DataFrame::new(vec![]);
    for i in 0 .. epochs.len() {
        df.push(&format!("{}", epochs[i]), Series::new(diffs.row(i)));
        println!("{:?}", diffs.row(i));
    }

    df.print();

    Ok(())
}

fn curve_diff(x: &[f64], y: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    assert!(x.len() <= y.len());

    let epoch_x = linspace(1, x.len() as f64, x.len());
    let epoch_y = linspace(1, y.len() as f64, y.len());
    let x = savitzky_golay_filter_9(&x);
    let y = savitzky_golay_filter_9(&y);
    let cx = cubic_hermite_spline(&epoch_x, &x, Quadratic)?;
    let cy = cubic_hermite_spline(&epoch_y, &y, Quadratic)?;
    let f = |t: f64| (cx.eval(t) - cy.eval(t)).abs();

    let epoch_init = 10f64;
    let epoch_final = x.len() as f64 * 0.8;
    
    let diff = integrate(f, (epoch_init, epoch_final), G7K15R(1e-5, 20));
    Ok(diff / (x.len() as f64))
}

/// Savitzky-Golay Filter for smoothing (5-point quadratic)
pub fn savitzky_golay_filter_5(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut y_smooth = vec![0f64; n];
    for i in 0..n {
        if i < 2 || i > n - 3 {
            y_smooth[i] = y[i];
        } else {
            y_smooth[i] = (-3f64 * (y[i - 2] + y[i + 2])
                + 12f64 * (y[i - 1] + y[i + 1])
                + 17f64 * y[i])
                / 35f64;
        }
    }
    y_smooth
}

/// Savitzky-Golay Filter for smoothing (9-point quadratic)
pub fn savitzky_golay_filter_9(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut y_smooth = vec![0f64; n];
    for i in 0..n {
        if i < 4 || i > n - 5 {
            y_smooth[i] = y[i];
        } else {
            y_smooth[i] = (-21f64 * (y[i - 4] + y[i + 4])
                + 14f64 * (y[i - 3] + y[i + 3])
                + 39f64 * (y[i - 2] + y[i + 2])
                + 54f64 * (y[i - 1] + y[i + 1])
                + 59f64 * y[i])
                / 231f64;
        }
    }
    y_smooth
}

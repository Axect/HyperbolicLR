use peroxide::fuga::*;
use dialoguer::{theme::ColorfulTheme, Select};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = vec!["CIFAR10", "OSC", "Integral"];
    let select = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Dataset")
        .items(&dataset)
        .default(0)
        .interact()?;
    let dataset = dataset[select];
    let models = match dataset {
        "CIFAR10" => vec!["CNN", "ResNet", "ViT"],
        "OSC" => vec!["LSTM"],
        "Integral" => vec!["TF", "MLP"],
        _ => panic!("Invalid dataset"),
    };
    let select = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Model")
        .items(&models)
        .default(0)
        .interact()
        .unwrap();
    let model = models[select];

    let schedulers = vec!["N", "P", "C", "E", "H", "EH", "L", "S", "OC", "CY", "WH", "WEH", "WC"];
    let select = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Scheduler")
        .items(&schedulers)
        .default(0)
        .interact()
        .unwrap();
    let scheduler = schedulers[select];

    let file = format!("data/Result_{}_{}-{}.csv", dataset, model, scheduler);
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
    let x = savitzky_golay_filter_9(x);
    let y = savitzky_golay_filter_9(y);
    Ok(
        zip_with(|x, y| (x - y).abs() / (x + y), &x, &y)
            .sum() / x.len() as f64
    )
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

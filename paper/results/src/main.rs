use peroxide::fuga::*;
use dialoguer::{theme::ColorfulTheme, Select};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schedulers = ["N", "P", "C", "E", "H", "EH"];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select scheduler")
        .items(&schedulers)
        .default(0)
        .interact()?;
    let selected_scheduler = schedulers[selection];
    println!("Selected scheduler: {}", selected_scheduler);

    let mut df = DataFrame::read_csv(&format!("data/Result_Integral_TF-{}.csv", selected_scheduler), ',')?;
    df.as_types(vec![F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64]);

    let val_loss_50: Vec<f64> = df[format!("{}50_val_loss", selected_scheduler).as_str()].to_vec();
    let val_loss_100: Vec<f64> = df[format!("{}100_val_loss", selected_scheduler).as_str()].to_vec();
    let val_loss_150: Vec<f64> = df[format!("{}150_val_loss", selected_scheduler).as_str()].to_vec();
    let val_loss_200: Vec<f64> = df[format!("{}200_val_loss", selected_scheduler).as_str()].to_vec();
    let val_losses = [val_loss_50, val_loss_100, val_loss_150, val_loss_200];

    let mut result_matrix = zeros_shape(4, 4, Row);
    for i in 0..4 {
        for j in i+1 .. 4 {
            result_matrix[(i, j)] = measure_diff_learning_curve(&val_losses[i], &val_losses[j])?;
        }
    }

    let mut df = DataFrame::new(vec![]);

    df.push("50", Series::new(result_matrix.row(0)));
    df.push("100", Series::new(result_matrix.row(1)));
    df.push("150", Series::new(result_matrix.row(2)));

    df.print();

    Ok(())
}

fn measure_diff_learning_curve(x: &[f64], y: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    assert!(x.len() < y.len(), "X must be smaller than Y");

    let epoch_x = linspace(1, x.len() as f64, x.len());
    let epoch_y = linspace(1, y.len() as f64, y.len());

    let ln_x = x.iter().map(|x| x.ln()).collect::<Vec<f64>>();
    let ln_y = y.iter().map(|x| x.ln()).collect::<Vec<f64>>();

    let cs_x = cubic_hermite_spline(&epoch_x, &ln_x, Quadratic)?;
    let cs_y = cubic_hermite_spline(&epoch_y, &ln_y, Quadratic)?;

    let f = |epoch: f64| {
        (cs_x.eval(epoch).exp() - cs_y.eval(epoch).exp()).abs()
    };

    Ok(integrate(f, (1.0, x.len() as f64), G7K15R(1e-4, 20)) / x.len() as f64)
}

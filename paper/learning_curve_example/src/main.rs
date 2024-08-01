use peroxide::fuga::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut df_c = DataFrame::read_csv("./data/Result_OSC_LSTM-C.csv", ',')?;
    let mut df_eh = DataFrame::read_csv("./data/Result_OSC_LSTM-EH.csv", ',')?;
    df_c.as_types(vec![F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64]);
    df_eh.as_types(vec![F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64]);

    let c50: Vec<f64> = df_c["C50_val_loss"].to_vec();
    let c100: Vec<f64> = df_c["C100_val_loss"].to_vec();
    let eh50: Vec<f64> = df_eh["EH50_val_loss"].to_vec();
    let eh100: Vec<f64> = df_eh["EH100_val_loss"].to_vec();

    let step_50 = linspace(1, c50.len() as f64, c50.len());
    let step_100 = linspace(1, c100.len() as f64, c100.len());

    // Measure
    let diff_poly = curve_diff(&c50, &c100)?;
    let diff_eh = curve_diff(&eh50, &eh100)?;

    println!("CosineAnneling: {:.4e}", diff_poly);
    println!("ExpHyperbolic: {:.4e}", diff_eh);

    // Plotting
    // CosineAnneling
    let mut plt = Plot2D::new();
    plt.insert_pair((step_50.clone(), c50.clone()))
        .insert_pair((step_100.clone(), c100.clone()))
        .set_legend(vec!["CosineAnnelingLR(50)", "CosineAnnelingLR(100)"])
        .set_xlabel("Epoch")
        .set_ylabel("Validation Loss")
        .set_yscale(PlotScale::Log)
        .set_line_style(vec![(0, LineStyle::Solid), (1, LineStyle::Dashed)])
        .set_color(vec![(0, "darkblue"), (1, "red")])
        .set_style(PlotStyle::Nature)
        .set_dpi(600)
        .set_path("../figs/osc_learning_curve_cos.png")
        .savefig()?;

    // ExpHyperbolic
    let mut plt = Plot2D::new();
    plt.insert_pair((step_50.clone(), eh50.clone()))
        .insert_pair((step_100.clone(), eh100.clone()))
        .set_legend(vec!["ExpHyperbolicLR(50)", "ExpHyperbolicLR(100)"])
        .set_xlabel("Epoch")
        .set_ylabel("Validation Loss")
        .set_yscale(PlotScale::Log)
        .set_line_style(vec![(0, LineStyle::Solid), (1, LineStyle::Dashed)])
        .set_color(vec![(0, "darkblue"), (1, "red")])
        .set_style(PlotStyle::Nature)
        .set_dpi(600)
        .set_path("../figs/osc_learning_curve_eh.png")
        .savefig()?;

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

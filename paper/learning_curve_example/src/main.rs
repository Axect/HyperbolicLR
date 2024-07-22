use peroxide::fuga::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = DataFrame::read_csv("data/deeponet_learning_curve.csv", ',')?;
    df.as_types(vec![F64, F64, F64, F64, F64]);
    df.print();

    let p50: Vec<f64> = df["p50"].to_vec();
    let p100: Vec<f64> = df["p100"].to_vec();
    let eh50: Vec<f64> = df["eh50"].to_vec();
    let eh100: Vec<f64> = df["eh100"].to_vec();

    let step_50 = linspace(1, p50.len() as f64, p50.len());
    let step_100 = linspace(1, p100.len() as f64, p100.len());

    // Measure
    let diff_poly = curve_diff(&p50, &p100)?;
    let diff_eh = curve_diff(&eh50, &eh100)?;

    println!("Polynomial: {:.4e}", diff_poly);
    println!("ExpHyperbolic: {:.4e}", diff_eh);

    // Plotting
    // Polynomial
    let mut plt = Plot2D::new();
    plt.insert_pair((step_50.clone(), p50.clone()))
        .insert_pair((step_100.clone(), p100.clone()))
        .set_legend(vec!["PolynomialLR(50)", "PolynomialLR(100)"])
        .set_xlabel("Epoch")
        .set_ylabel("Validation Loss")
        .set_yscale(PlotScale::Log)
        .set_line_style(vec![(0, LineStyle::Solid), (1, LineStyle::Dashed)])
        .set_color(vec![(0, "darkblue"), (1, "red")])
        .set_style(PlotStyle::Nature)
        .set_dpi(600)
        .set_path("../figs/deeponet_learning_curve_poly.png")
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
        .set_path("../figs/deeponet_learning_curve_eh.png")
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

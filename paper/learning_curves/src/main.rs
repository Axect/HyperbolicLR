use peroxide::fuga::*;
use dialoguer::{theme::ColorfulTheme, Select};

const SCHEDULERS: [&str; 13] = ["N", "P", "C", "E", "H", "EH", "L", "S", "OC", "CY", "WH", "WEH", "WC"];
const TASKMODELS: [&str; 9] = [
    "CIFAR10_CNN",
    "CIFAR10_CNN_ACC",
    "CIFAR10_ResNet",
    "CIFAR10_ResNet_ACC",
    "CIFAR10_ViT",
    "CIFAR10_ViT_ACC",
    "OSC_LSTM",
    "Integral_MLP",
    "Integral_TF",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let task_model = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Task Model")
        .items(&TASKMODELS)
        .default(0)
        .interact()?;

    let task_model = TASKMODELS[task_model];

    for scheduler in SCHEDULERS {
        let data_path = format!("./data/Result_{}-{}.csv", task_model, scheduler);
        let mut df = DataFrame::read_csv(&data_path, ',')?;
        df.as_types(vec![F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64]);

        let prefix = scheduler;
        let val_50: Vec<f64>  = df[format!("{}50_val_loss", prefix).as_str()].to_vec();
        let val_100: Vec<f64> = df[format!("{}100_val_loss", prefix).as_str()].to_vec();
        let val_150: Vec<f64> = df[format!("{}150_val_loss", prefix).as_str()].to_vec();
        let val_200: Vec<f64> = df[format!("{}200_val_loss", prefix).as_str()].to_vec();

        let min = val_50.min().min(val_100.min().min(val_150.min().min(val_200.min())));
        let max = val_50.max().max(val_100.max().max(val_150.max().max(val_200.max())));

        let epoch_50 = linspace(0, 49, 50);
        let epoch_100 = linspace(0, 99, 100);
        let epoch_150 = linspace(0, 149, 150);
        let epoch_200 = linspace(0, 199, 200);

        let is_acc = task_model.contains("ACC");
        let ylabel = if is_acc {
            "Accuracy"
        } else {
            "Validation Loss"
        };

        let mut plt = Plot2D::new();

        if !is_acc {
            plt.set_yscale(PlotScale::Log)
                .set_ylim((min * 0.99, max));
        } else {
            plt.set_ylim((0.5, 0.95));
        }

        let full_name = match scheduler {
            "N" => "No Scheduler",
            "P" => "PolynomialLR",
            "C" => "CosineAnnealingLR",
            "E" => "ExponentialLR",
            "H" => "HyperbolicLR",
            "EH" => "ExpHyperbolicLR",
            "L" => "LinearLR",
            "S" => "StepLR",
            "OC" => "OneCycleLR",
            "CY" => "CyclicLR",
            "WH" => "Warmup+HyperbolicLR",
            "WEH" => "Warmup+ExpHyperbolicLR",
            "WC" => "Warmup+CosineAnnealingLR",
            _ => unreachable!(),
        };

        plt
            .insert_pair((epoch_200, val_200))
            .insert_pair((epoch_150, val_150))
            .insert_pair((epoch_100, val_100))
            .insert_pair((epoch_50, val_50))
            .set_title(full_name)
            .set_legend(vec!["Epoch 200", "Epoch 150", "Epoch 100", "Epoch 50"])
            .set_color(vec![(0, "darkblue"), (1, "red"), (2, "darkgreen"), (3, "orange")])
            .set_xlabel("Epoch")
            .set_ylabel(ylabel)
            .set_xlim((0f64, 200f64))
            .set_style(PlotStyle::Nature)
            .set_dpi(600)
            .tight_layout()
            .set_path(&format!("../figs/learning_curve_{}-{}.png", task_model, scheduler))
            .savefig()?;
    }

    Ok(())
}

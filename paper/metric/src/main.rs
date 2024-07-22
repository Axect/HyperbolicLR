use peroxide::fuga::*;
use std::f64::consts::PI;

fn main() -> anyhow::Result<()> {
    let poly_lrs = [
        PolynomialLR { init_lr: 1f64, max_epoch: 250.0, power: 0.5 },
        PolynomialLR { init_lr: 1f64, max_epoch: 500.0, power: 0.5 },
        PolynomialLR { init_lr: 1f64, max_epoch: 750.0, power: 0.5 },
        PolynomialLR { init_lr: 1f64, max_epoch: 1000.0, power: 0.5 },
    ];

    let cos_lrs = [
        CosineAnnealingLR { min_lr: 1e-4, max_lr: 1f64, max_epoch: 250.0 },
        CosineAnnealingLR { min_lr: 1e-4, max_lr: 1f64, max_epoch: 500.0 },
        CosineAnnealingLR { min_lr: 1e-4, max_lr: 1f64, max_epoch: 750.0 },
        CosineAnnealingLR { min_lr: 1e-4, max_lr: 1f64, max_epoch: 1000.0 },
    ];

    let hyp_lrs = [
        HyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 250.0, upper_bound: 1000.0 },
        HyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 500.0, upper_bound: 1000.0 },
        HyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 750.0, upper_bound: 1000.0 },
        HyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 1000.0, upper_bound: 1000.0 },
    ];

    let exp_hyp_lrs = [
        ExpHyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 250.0, upper_bound: 1000.0 },
        ExpHyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 500.0, upper_bound: 1000.0 },
        ExpHyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 750.0, upper_bound: 1000.0 },
        ExpHyperbolicLR { init_lr: 1f64, infimum_lr: 1e-3, max_epoch: 1000.0, upper_bound: 1000.0 },
    ];

    evaluate_lrs(&poly_lrs)?;
    evaluate_lrs(&cos_lrs)?;
    evaluate_lrs(&hyp_lrs)?;
    evaluate_lrs(&exp_hyp_lrs)?;

    plot_lrs(&poly_lrs, "figs/poly")?;
    plot_lrs(&cos_lrs, "figs/cos")?;
    plot_lrs(&hyp_lrs, "figs/hyp")?;
    plot_lrs(&exp_hyp_lrs, "figs/exp_hyp")?;

    Ok(())
}

fn evaluate_lrs<T: LRScheduler>(lrs: &[T]) -> anyhow::Result<()> {
    let metrics: Vec<f64> = lrs.iter().map(|lr| lr.eval_metric().unwrap()).collect();
    let last_metric = metrics.last().unwrap();
    for metric in &metrics[0..metrics.len()-1] {
        print!("{:.2}, ", eval_error(*metric, *last_metric));
    }
    println!("{:.2}", eval_error(*metrics.last().unwrap(), *last_metric));
    Ok(())
}

fn plot_lrs<T: LRScheduler>(lrs: &[T], name: &str) -> anyhow::Result<()> {
    let mut plt = Plot2D::new();
    let legends = vec![r"$N=250$", r"$N=500$", r"$N=750$", r"$N=1000$"];

    lrs.iter().for_each(|lr| {
        let max_epoch = lr.max_epoch();
        let epochs = seq(0.0, max_epoch, 1.0);
        let lrs = lr.get_lr_vec(&epochs);

        plt.insert_pair((epochs, lrs));
    });

    if name.contains("exp") {
        plt.set_yscale(PlotScale::Log);
    }

    plt.set_xlabel("Epoch")
        .set_ylabel("Learning Rate")
        .set_legend(legends)
        .set_line_style(vec![(0, LineStyle::Solid), (1, LineStyle::Dashed), (2, LineStyle::Dotted), (3, LineStyle::DashDot)])
        .set_color(vec![(0, "darkblue"), (1, "red"), (2, "darkgreen"), (3, "black")])
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path(&format!("{}.png", name))
        .savefig()?;

    Ok(())
}

trait LRScheduler: RootFindingProblem<1, 1, (f64, f64)> + Sized {
    fn get_lr(&self, epoch: f64) -> f64;

    fn max_epoch(&self) -> f64 {
        self.initial_guess().1
    }

    #[allow(dead_code)]
    fn get_lr_vec(&self, epochs: &[f64]) -> Vec<f64> {
        epochs
            .iter()
            .map(|epoch| self.get_lr(*epoch))
            .collect::<Vec<f64>>()
    }

    fn eval_metric(&self) -> anyhow::Result<f64> {
        let bisect = BisectionMethod { max_iter: 100, tol: 1e-4 };
        let crit_epoch = bisect.find(self)?;

        let f = |epoch: f64| self.function([epoch]).unwrap()[0];
        Ok(integrate(f, (0f64, crit_epoch[0]), G7K15R(1e-5, 20)))
    }
}

struct PolynomialLR {
    pub init_lr: f64,
    pub max_epoch: f64,
    pub power: f64,
}

impl RootFindingProblem<1, 1, (f64, f64)> for PolynomialLR {
    fn initial_guess(&self) -> (f64, f64) {
        (0f64, self.max_epoch)
    }

    fn function(&self, x: Pt<1>) -> anyhow::Result<Pt<1>> {
        Ok([self.get_lr(x[0]) - 0.8 * self.init_lr])
    }
}

impl LRScheduler for PolynomialLR {
    fn get_lr(&self, epoch: f64) -> f64 {
        self.init_lr * (1.0 - epoch / self.max_epoch).powf(self.power)
    }
}

struct CosineAnnealingLR {
    pub min_lr: f64,
    pub max_lr: f64,
    pub max_epoch: f64,
}

impl RootFindingProblem<1, 1, (f64, f64)> for CosineAnnealingLR {
    fn initial_guess(&self) -> (f64, f64) {
        (0f64, self.max_epoch)
    }

    fn function(&self, x: Pt<1>) -> anyhow::Result<Pt<1>> {
        Ok([self.get_lr(x[0]) - 0.8 * self.min_lr])
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, epoch: f64) -> f64 {
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * epoch / self.max_epoch).cos())
    }
}

struct HyperbolicLR {
    pub init_lr: f64,
    pub infimum_lr: f64,
    pub max_epoch: f64,
    pub upper_bound: f64,
}

impl RootFindingProblem<1, 1, (f64, f64)> for HyperbolicLR {
    fn initial_guess(&self) -> (f64, f64) {
        (0f64, self.max_epoch)
    }

    fn function(&self, x: Pt<1>) -> anyhow::Result<Pt<1>> {
        Ok([self.get_lr(x[0]) - 0.8 * self.init_lr])
    }
}

impl LRScheduler for HyperbolicLR {
    #[allow(non_snake_case)]
    fn get_lr(&self, epoch: f64) -> f64 {
        let delta_lr = self.init_lr - self.infimum_lr;
        let N = self.max_epoch;
        let U = self.upper_bound;

        self.init_lr + delta_lr * (
            ((N- epoch) / U * (2f64 - (N + epoch) / U)).sqrt()
            - (N / U * (2f64 - N / U)).sqrt()
        )
    }
}

struct ExpHyperbolicLR {
    pub init_lr: f64,
    pub infimum_lr: f64,
    pub max_epoch: f64,
    pub upper_bound: f64,
}

impl RootFindingProblem<1, 1, (f64, f64)> for ExpHyperbolicLR {
    fn initial_guess(&self) -> (f64, f64) {
        (0f64, self.max_epoch)
    }

    fn function(&self, x: Pt<1>) -> anyhow::Result<Pt<1>> {
        Ok([self.get_lr(x[0]) - 0.8 * self.init_lr])
    }
}

impl LRScheduler for ExpHyperbolicLR {
    #[allow(non_snake_case)]
    fn get_lr(&self, epoch: f64) -> f64 {
        let delta_lr = self.init_lr / self.infimum_lr;
        let N = self.max_epoch;
        let U = self.upper_bound;

        self.init_lr * delta_lr.powf(
            ((N- epoch) / U * (2f64 - (N + epoch) / U)).sqrt()
            - (N / U * (2f64 - N / U)).sqrt()
        )
    }
}

fn eval_error(f1: f64, f2: f64) -> f64 {
    (f1 - f2).abs() / f2 * 100f64
}

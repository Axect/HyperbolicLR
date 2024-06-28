use peroxide::fuga::*;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = DataFrame::read_csv("../data/ETTh1.csv", ',')?;
    df.as_types(vec![Str, F64, F64, F64, F64, F64, F64, F64]);

    let HUFL: Vec<f64> = df["HUFL"].to_vec();
    let HULL: Vec<f64> = df["HULL"].to_vec();
    let MUFL: Vec<f64> = df["MUFL"].to_vec();
    let MULL: Vec<f64> = df["MULL"].to_vec();
    let LUFL: Vec<f64> = df["LUFL"].to_vec();
    let LULL: Vec<f64> = df["LULL"].to_vec();
    let OT: Vec<f64> = df["OT"].to_vec();

    // Normalize
    let HUFL_stat = vec![HUFL.mean(), HUFL.sd()];
    let HULL_stat = vec![HULL.mean(), HULL.sd()];
    let MUFL_stat = vec![MUFL.mean(), MUFL.sd()];
    let MULL_stat = vec![MULL.mean(), MULL.sd()];
    let LUFL_stat = vec![LUFL.mean(), LUFL.sd()];
    let LULL_stat = vec![LULL.mean(), LULL.sd()];
    let OT_stat = vec![OT.mean(), OT.sd()];

    let HUFL = HUFL.into_iter().map(|x| ((x - HUFL_stat[0]) / HUFL_stat[1]) as f32).collect();
    let HULL = HULL.into_iter().map(|x| ((x - HULL_stat[0]) / HULL_stat[1]) as f32).collect();
    let MUFL = MUFL.into_iter().map(|x| ((x - MUFL_stat[0]) / MUFL_stat[1]) as f32).collect();
    let MULL = MULL.into_iter().map(|x| ((x - MULL_stat[0]) / MULL_stat[1]) as f32).collect();
    let LUFL = LUFL.into_iter().map(|x| ((x - LUFL_stat[0]) / LUFL_stat[1]) as f32).collect();
    let LULL = LULL.into_iter().map(|x| ((x - LULL_stat[0]) / LULL_stat[1]) as f32).collect();
    let OT = OT.into_iter().map(|x| ((x - OT_stat[0]) / OT_stat[1]) as f32).collect();

    // Save to parquet
    let mut dg = DataFrame::new(vec![]);
    dg.push("HUFL", Series::new(HUFL));
    dg.push("HULL", Series::new(HULL));
    dg.push("MUFL", Series::new(MUFL));
    dg.push("MULL", Series::new(MULL));
    dg.push("LUFL", Series::new(LUFL));
    dg.push("LULL", Series::new(LULL));
    dg.push("OT", Series::new(OT));
    dg.print();
    dg.write_parquet("../data/ETTh1.parquet", CompressionOptions::Snappy)?;

    // Save statistics to parquet
    let mut dh = DataFrame::new(vec![]);
    dh.push("HUFL", Series::new(HUFL_stat));
    dh.push("HULL", Series::new(HULL_stat));
    dh.push("MUFL", Series::new(MUFL_stat));
    dh.push("MULL", Series::new(MULL_stat));
    dh.push("LUFL", Series::new(LUFL_stat));
    dh.push("LULL", Series::new(LULL_stat));
    dh.push("OT", Series::new(OT_stat));
    dh.print();
    dh.write_parquet("../data/ETTh1_stat.parquet", CompressionOptions::Snappy)?;

    Ok(())
}

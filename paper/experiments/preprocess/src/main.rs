use peroxide::{cbind, fuga::*};

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

    let HUFL = HUFL.into_iter().map(|x| (x - HUFL_stat[0]) / HUFL_stat[1]).collect::<Vec<_>>();
    let HULL = HULL.into_iter().map(|x| (x - HULL_stat[0]) / HULL_stat[1]).collect::<Vec<_>>();
    let MUFL = MUFL.into_iter().map(|x| (x - MUFL_stat[0]) / MUFL_stat[1]).collect::<Vec<_>>();
    let MULL = MULL.into_iter().map(|x| (x - MULL_stat[0]) / MULL_stat[1]).collect::<Vec<_>>();
    let LUFL = LUFL.into_iter().map(|x| (x - LUFL_stat[0]) / LUFL_stat[1]).collect::<Vec<_>>();
    let LULL = LULL.into_iter().map(|x| (x - LULL_stat[0]) / LULL_stat[1]).collect::<Vec<_>>();
    let OT = OT.into_iter().map(|x| (x - OT_stat[0]) / OT_stat[1]).collect::<Vec<_>>();

    let data = vec![
        HUFL.clone(),
        HULL.clone(),
        MUFL.clone(),
        MULL.clone(),
        LUFL.clone(),
        LULL.clone(),
        OT.clone(),
    ];
    let data = matrix(data.into_iter().flatten().collect(), HUFL.len(), 7, Col);
    println!("Normalize Complete!");

    // Train Test Split
    //
    // - Total: 2016-07-01 ~ 2018-06-25 (725 days * 24 hours)
    // - Task: From 7days data, predict 1day data
    // - Split: Randomly select 2016-07-01 ~ 2018-06-25 (5days input + 1day label)
    // - Split ratio: 0.8
    let total_len = data.row;
    let days = total_len / 24;
    let dataset_len = (days - 8) * 24;
    let input_amount = 7 * 24;
    let label_amount = 24;
    let (input, label): (Vec<Matrix>, Vec<Matrix>) = (0 .. dataset_len)
        .map(|i| {
            let mut input = zeros(input_amount, 7);
            let mut label = zeros(label_amount, 7);

            for j in 0 .. input_amount {
                input.subs_row(j, &data.row(i+j));
            }
            for j in 0 .. label_amount {
                label.subs_row(j, &data.row(i+j+input_amount));
            }
            (input, label)
        })
        .unzip();
    println!("Sliding Window Complete!");

    let mut rng = stdrng_from_seed(42);
    let mut total_ics = (0 .. input.len()).collect::<Vec<_>>();
    total_ics.shuffle(&mut rng);
    let train_ics = total_ics[0 .. (input.len() as f64 * 0.8) as usize].to_vec();
    let test_ics = total_ics[(input.len() as f64 * 0.8) as usize ..].to_vec();
    let train_input = train_ics.iter().map(|i| input[*i].clone()).collect::<Vec<_>>();
    let train_label = train_ics.iter().map(|i| label[*i].clone()).collect::<Vec<_>>();
    let test_input = test_ics.iter().map(|i| input[*i].clone()).collect::<Vec<_>>();
    let test_label = test_ics.iter().map(|i| label[*i].clone()).collect::<Vec<_>>();
    
    let train_size = train_input.len();
    let test_size  = test_input.len();

    println!("Train Size: {}, Test Size: {}", train_size, test_size);

    let train_data = train_input
        .into_iter()
        .zip(train_label)
        .enumerate()
        .flat_map(|(i, (x, y))| {
            let z = rbind(x, y).unwrap();

            let group = vec![i as f64; z.row].to_col();
            let data_type = concat(&vec![0f64; input_amount], &vec![1f64; label_amount]).to_col();

            let result = cbind!(group, data_type, z);
            match result.shape {
                Row => result.data,
                Col => result.change_shape().data,
            }
        })
        .collect::<Vec<_>>();
    let train_data_len = train_data.len();
    let train_data = matrix(train_data, train_data_len / 9, 9, Row).change_shape();

    let test_data = test_input
        .into_iter()
        .zip(test_label)
        .enumerate()
        .flat_map(|(i, (x, y))| {
            let z = rbind(x, y).unwrap();

            let group = vec![i as f64; z.row].to_col();
            let data_type = concat(&vec![0f64; input_amount], &vec![1f64; label_amount]).to_col();

            let result = cbind!(group, data_type, z);
            match result.shape {
                Row => result.data,
                Col => result.change_shape().data,
            }
        })
        .collect::<Vec<_>>();
    let test_data_len = test_data.len();
    let test_data = matrix(test_data, test_data_len / 9, 9, Row).change_shape();

    // Save train data to parquet
    let mut df = DataFrame::new(vec![]);
    df.push("group", Series::new(v64_to_v32(train_data.col(0))));
    df.push("type", Series::new(v64_to_v32(train_data.col(1))));
    df.push("HUFL", Series::new(v64_to_v32(train_data.col(2))));
    df.push("HULL", Series::new(v64_to_v32(train_data.col(3))));
    df.push("MUFL", Series::new(v64_to_v32(train_data.col(4))));
    df.push("MULL", Series::new(v64_to_v32(train_data.col(5))));
    df.push("LUFL", Series::new(v64_to_v32(train_data.col(6))));
    df.push("LULL", Series::new(v64_to_v32(train_data.col(7))));
    df.push("OT", Series::new(v64_to_v32(train_data.col(8))));
    df.write_parquet("../data/ETTh1_train.parquet", CompressionOptions::Snappy)?;
    println!("Train Data Saved");
    df.print();

    // Save test data to parquet
    let mut df = DataFrame::new(vec![]);
    df.push("group", Series::new(v64_to_v32(test_data.col(0))));
    df.push("type", Series::new(v64_to_v32(test_data.col(1))));
    df.push("HUFL", Series::new(v64_to_v32(test_data.col(2))));
    df.push("HULL", Series::new(v64_to_v32(test_data.col(3))));
    df.push("MUFL", Series::new(v64_to_v32(test_data.col(4))));
    df.push("MULL", Series::new(v64_to_v32(test_data.col(5))));
    df.push("LUFL", Series::new(v64_to_v32(test_data.col(6))));
    df.push("LULL", Series::new(v64_to_v32(test_data.col(7))));
    df.push("OT", Series::new(v64_to_v32(test_data.col(8))));
    df.write_parquet("../data/ETTh1_test.parquet", CompressionOptions::Snappy)?;
    println!("Test Data Saved");
    df.print();

    // Save statistics to parquet
    let mut df = DataFrame::new(vec![]);
    df.push("HUFL", Series::new(v64_to_v32(HUFL_stat)));
    df.push("HULL", Series::new(v64_to_v32(HULL_stat)));
    df.push("MUFL", Series::new(v64_to_v32(MUFL_stat)));
    df.push("MULL", Series::new(v64_to_v32(MULL_stat)));
    df.push("LUFL", Series::new(v64_to_v32(LUFL_stat)));
    df.push("LULL", Series::new(v64_to_v32(LULL_stat)));
    df.push("OT", Series::new(v64_to_v32(OT_stat)));
    df.write_parquet("../data/ETTh1_stat.parquet", CompressionOptions::Snappy)?;
    println!("Statistics Saved");
    df.print();

    Ok(())
}

fn v64_to_v32(v: Vec<f64>) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect::<Vec<_>>()
}

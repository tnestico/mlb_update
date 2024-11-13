import polars as pl
import joblib

model = joblib.load('stuff_model/lgbm_model_2020_2023.joblib')
# Read the values from the text file
with open('stuff_model/target_stats.txt', 'r') as file:
    lines = file.readlines()
    target_mean = float(lines[0].strip())
    target_std = float(lines[1].strip())
    
# Define the features to be used for training
features = ['start_speed',
            'spin_rate',
            'extension',
            'az',
            'ax',
            'x0',
            'z0',
            'speed_diff',
            'az_diff',
            'ax_diff']


def stuff_apply(df:pl.DataFrame) -> pl.DataFrame:
    # Filter the dataframe to include only the rows for the year 2024 and drop rows with null values in the specified features and target column
    # df_test = df.drop_nulls(subset=features)
    df_test = df.clone()

    # Predict the target values for the 2024 data using the trained model
    df_test = df_test.with_columns(
        pl.Series(name="target", values=model.predict(df_test[features].to_numpy()))
    )
    # Standardize the target column to create a z-score
    df_test = df_test.with_columns(
        ((pl.col('target') - target_mean) / target_std).alias('target_zscore')
    )

    # Convert the z-score to tj_stuff_plus
    df_test = df_test.with_columns(
        (100 - (pl.col('target_zscore') * 10)).alias('tj_stuff_plus')
    )

    df_pitch_types = pl.read_csv('stuff_model/tj_stuff_plus_pitch.csv')

    # Join the pitch type statistics with the main DataFrame based on pitch_type
    df_pitch_all = df_test.join(df_pitch_types, left_on='pitch_type', right_on='pitch_type')

    # Normalize pitch_grade values to a range between -0.5 and 0.5 based on the percentiles
    df_pitch_all = df_pitch_all.with_columns(
        ((pl.col('tj_stuff_plus') - pl.col('mean')) / pl.col('std')).alias('pitch_grade')
    )

    # Scale the pitch_grade values to a range between 20 and 80
    df_pitch_all = df_pitch_all.with_columns(
        (pl.col('pitch_grade') * 10 + 50).clip(20, 80)
    )
    return df_pitch_all
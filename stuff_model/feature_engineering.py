import polars as pl
import numpy as np

def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    # Extract the year from the game_date column
    df = df.with_columns(
        pl.col('game_date').str.slice(0, 4).alias('year')
    )

    df = df.with_columns([
        
        (-(pl.col('vy0')**2 - (2 * pl.col('ay') * (pl.col('y0') - 17/12)))**0.5).alias('vy_f'),
        ])

    df = df.with_columns([
        ((pl.col('vy_f') - pl.col('vy0')) / pl.col('ay')).alias('t'),
        ])

    df = df.with_columns([
        (pl.col('vz0') + (pl.col('az') * pl.col('t'))).alias('vz_f'),
        (pl.col('vx0') + (pl.col('ax') * pl.col('t'))).alias('vx_f')
        ])

    df = df.with_columns([
            (-np.arctan(pl.col('vz_f') / pl.col('vy_f')) * (180 / np.pi)).alias('vaa'),
            (-np.arctan(pl.col('vx_f') / pl.col('vy_f')) * (180 / np.pi)).alias('haa')
        ])

    # Mirror horizontal break for left-handed pitchers
    df = df.with_columns(
        pl.when(pl.col('pitcher_hand') == 'L')
        .then(-pl.col('ax'))
        .otherwise(pl.col('ax'))
        .alias('ax')
    )

    # Mirror horizontal break for left-handed pitchers
    df = df.with_columns(
        pl.when(pl.col('pitcher_hand') == 'L')
        .then(-pl.col('hb'))
        .otherwise(pl.col('hb'))
        .alias('hb')
    )

    # Mirror horizontal release point for left-handed pitchers
    df = df.with_columns(
        pl.when(pl.col('pitcher_hand') == 'L')
        .then(pl.col('x0'))
        .otherwise(-pl.col('x0'))
        .alias('x0')
    )

    # Define the pitch types to be considered
    pitch_types = ['SI', 'FF', 'FC']

    # Filter the DataFrame to include only the specified pitch types
    df_filtered = df.filter(pl.col('pitch_type').is_in(pitch_types))

    # Group by pitcher_id and year, then aggregate to calculate average speed and usage percentage
    df_agg = df_filtered.group_by(['pitcher_id', 'year', 'pitch_type']).agg([
        pl.col('start_speed').mean().alias('avg_fastball_speed'),
        pl.col('az').mean().alias('avg_fastball_az'),
        pl.col('ax').mean().alias('avg_fastball_ax'),
        pl.len().alias('count')
    ])

    # Sort the aggregated data by count and average fastball speed
    df_agg = df_agg.sort(['count', 'avg_fastball_speed'], descending=[True, True])
    df_agg = df_agg.unique(subset=['pitcher_id', 'year'], keep='first')

    # Join the aggregated data with the main DataFrame
    df = df.join(df_agg, on=['pitcher_id', 'year'])

    # If no fastball, use the fastest pitch for avg_fastball_speed
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_speed').is_null())
        .then(pl.col('start_speed').max().over('pitcher_id'))
        .otherwise(pl.col('avg_fastball_speed'))
        .alias('avg_fastball_speed')
    )

    # If no fastball, use the fastest pitch for avg_fastball_az
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_az').is_null())
        .then(pl.col('az').max().over('pitcher_id'))
        .otherwise(pl.col('avg_fastball_az'))
        .alias('avg_fastball_az')
    )

    # If no fastball, use the fastest pitch for avg_fastball_ax
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_ax').is_null())
        .then(pl.col('ax').max().over('ax'))
        .otherwise(pl.col('avg_fastball_ax'))
        .alias('avg_fastball_ax')
    )

    # Calculate pitch differentials
    df = df.with_columns(
        (pl.col('start_speed') - pl.col('avg_fastball_speed')).alias('speed_diff'),
        (pl.col('az') - pl.col('avg_fastball_az')).alias('az_diff'),
        (pl.col('ax') - pl.col('avg_fastball_ax')).abs().alias('ax_diff')
    )

    # Cast the year column to integer type
    df = df.with_columns(
        pl.col('year').cast(pl.Int64)
    )


    
    df = df.with_columns([
        pl.lit('All').alias('all')
    ])


    
    return df
import polars as pl
import numpy as np
import joblib

loaded_model = joblib.load('joblib_model/barrel_model.joblib')
in_zone_model = joblib.load('joblib_model/in_zone_model_knn_20240410.joblib')
attack_zone_model = joblib.load('joblib_model/model_attack_zone.joblib')
xwoba_model = joblib.load('joblib_model/xwoba_model.joblib')
px_model = joblib.load('joblib_model/linear_reg_model_x.joblib')
pz_model = joblib.load('joblib_model/linear_reg_model_z.joblib')


class df_update:
    def __init__(self):
        pass

    def update(self, df_clone: pl.DataFrame):

        df = df_clone.clone()
        # Assuming px_model is defined and df is your DataFrame
        hit_codes = ['single',
            'double','home_run', 'triple']

        ab_codes = ['single', 'strikeout', 'field_out',
            'grounded_into_double_play', 'fielders_choice', 'force_out',
            'double', 'field_error', 'home_run', 'triple',
            'double_play',
            'fielders_choice_out', 'strikeout_double_play',
            'other_out','triple_play']


        obp_true_codes = ['single', 'walk',
            'double','home_run', 'triple',
            'hit_by_pitch', 'intent_walk']

        obp_codes = ['single', 'strikeout', 'walk', 'field_out',
            'grounded_into_double_play', 'fielders_choice', 'force_out',
            'double', 'sac_fly', 'field_error', 'home_run', 'triple',
            'hit_by_pitch', 'double_play', 'intent_walk',
            'fielders_choice_out', 'strikeout_double_play',
            'sac_fly_double_play',
            'other_out','triple_play']


        contact_codes = ['In play, no out',
                'Foul', 'In play, out(s)',
            'In play, run(s)',
            'Foul Bunt']

        bip_codes = ['In play, no out', 'In play, run(s)','In play, out(s)']


        conditions_barrel = [
            df['launch_speed'].is_null(),
            (df['launch_speed'] * 1.5 - df['launch_angle'] >= 117) & 
            (df['launch_speed'] + df['launch_angle'] >= 124) & 
            (df['launch_speed'] >= 98) & 
            (df['launch_angle'] >= 4) & (df['launch_angle'] <= 50)
        ]
        choices_barrel = [False, True]

        conditions_tb = [
            (df['event_type'] == 'single'),
            (df['event_type'] == 'double'),
            (df['event_type'] == 'triple'),
            (df['event_type'] == 'home_run')
        ]
        choices_tb = [1, 2, 3, 4]


        conditions_woba = [
            df['event_type'].is_in(['strikeout', 'field_out', 'sac_fly', 'force_out', 'grounded_into_double_play', 'fielders_choice', 'field_error', 'sac_bunt', 'double_play', 'fielders_choice_out', 'strikeout_double_play', 'sac_fly_double_play', 'other_out']),
            df['event_type'] == 'walk',
            df['event_type'] == 'hit_by_pitch',
            df['event_type'] == 'single',
            df['event_type'] == 'double',
            df['event_type'] == 'triple',
            df['event_type'] == 'home_run'
        ]
        choices_woba = [0, 0.689, 0.720, 0.881, 1.254, 1.589, 2.048]

        woba_codes = ['strikeout', 'field_out', 'single', 'walk', 'hit_by_pitch', 'double', 'sac_fly', 'force_out', 'home_run', 'grounded_into_double_play', 'fielders_choice', 'field_error', 'triple', 'sac_bunt', 'double_play', 'fielders_choice_out', 'strikeout_double_play', 'sac_fly_double_play', 'other_out']

        pitch_cat = {'FA': 'Fastball',
                    'FF': 'Fastball',
                    'FT': 'Fastball',
                    'FC': 'Fastball',
                    'FS': 'Off-Speed',
                    'FO': 'Off-Speed',
                    'SI': 'Fastball',
                    'ST': 'Breaking',
                    'SL': 'Breaking',
                    'CU': 'Breaking',
                    'KC': 'Breaking',
                    'SC': 'Off-Speed',
                    'GY': 'Off-Speed',
                    'SV': 'Breaking',
                    'CS': 'Breaking',
                    'CH': 'Off-Speed',
                    'KN': 'Off-Speed',
                    'EP': 'Breaking',
                    'UN': None,
                    'IN': None,
                    'PO': None,
                    'AB': None,
                    'AS': None,
                    'NP': None}


        df = df.with_columns([
            pl.when(df['type_ab'].is_not_null()).then(1).otherwise(0).alias('pa'),
            pl.when(df['is_pitch']).then(1).otherwise(0).alias('pitches'),
            pl.when(df['sz_top'] == 0).then(None).otherwise(df['sz_top']).alias('sz_top'),
            pl.when(df['sz_bot'] == 0).then(None).otherwise(df['sz_bot']).alias('sz_bot'),
            pl.when(df['zone'] > 0).then(df['zone'] < 10).otherwise(None).alias('in_zone'),
            pl.Series(px_model.predict(df[['x']].fill_null(0).to_numpy())[:, 0]).alias('px_predict'),
            pl.Series(pz_model.predict(df[['y']].fill_null(0).to_numpy())[:, 0] + 3.2).alias('pz_predict'),
            pl.Series(in_zone_model.predict(df[['px','pz','sz_top','sz_bot']].fill_null(0).to_numpy())[:]).alias('in_zone_predict'),
            pl.Series(attack_zone_model.predict(df[['px','pz','sz_top','sz_bot']].fill_null(0).to_numpy())[:]).alias('attack_zone_predict'),
            pl.when(df['event_type'].is_in(hit_codes)).then(True).otherwise(False).alias('hits'),
            pl.when(df['event_type'].is_in(ab_codes)).then(True).otherwise(False).alias('ab'),
            pl.when(df['event_type'].is_in(obp_true_codes)).then(True).otherwise(False).alias('on_base'),
            pl.when(df['event_type'].is_in(obp_codes)).then(True).otherwise(False).alias('obp'),
            pl.when(df['play_description'].is_in(bip_codes)).then(True).otherwise(False).alias('bip'),
            pl.when(conditions_barrel[0]).then(choices_barrel[0]).when(conditions_barrel[1]).then(choices_barrel[1]).otherwise(None).alias('barrel'),
            pl.when(df['launch_angle'].is_null()).then(False).when((df['launch_angle'] >= 8) & (df['launch_angle'] <= 32)).then(True).otherwise(None).alias('sweet_spot'),
            pl.when(df['launch_speed'].is_null()).then(False).when(df['launch_speed'] >= 94.5).then(True).otherwise(None).alias('hard_hit'),
            pl.when(conditions_tb[0]).then(choices_tb[0]).when(conditions_tb[1]).then(choices_tb[1]).when(conditions_tb[2]).then(choices_tb[2]).when(conditions_tb[3]).then(choices_tb[3]).otherwise(None).alias('tb'),
            pl.when(conditions_woba[0]).then(choices_woba[0]).when(conditions_woba[1]).then(choices_woba[1]).when(conditions_woba[2]).then(choices_woba[2]).when(conditions_woba[3]).then(choices_woba[3]).when(conditions_woba[4]).then(choices_woba[4]).when(conditions_woba[5]).then(choices_woba[5]).when(conditions_woba[6]).then(choices_woba[6]).otherwise(None).alias('woba'),
            pl.when((df['play_code'] == 'S') | (df['play_code'] == 'W') | (df['play_code'] == 'T')).then(1).otherwise(0).alias('whiffs'),
            pl.when((df['play_code'] == 'S') | (df['play_code'] == 'W') | (df['play_code'] == 'T') | (df['play_code'] == 'C')).then(1).otherwise(0).alias('csw'),
            pl.when(pl.col('is_swing').cast(pl.Boolean)).then(1).otherwise(0).alias('swings'),
            pl.col('event_type').is_in(['strikeout','strikeout_double_play']).alias('k'),
            pl.col('event_type').is_in(['walk', 'intent_walk']).alias('bb'),
            pl.lit(None).alias('attack_zone'),
            pl.lit(None).alias('woba_pred'),
            pl.lit(None).alias('woba_pred_contact')

        ])

        df = df.with_columns([
            pl.when(df['event_type'].is_in(woba_codes)).then(1).otherwise(None).alias('woba_codes'),
            pl.when(df['event_type'].is_in(woba_codes)).then(1).otherwise(None).alias('xwoba_codes'),
            pl.when((pl.col('tb') >= 0)).then(df['woba']).otherwise(None).alias('woba_contact'),
            pl.when(pl.col('px').is_null()).then(pl.col('px_predict')).otherwise(pl.col('px')).alias('px'),
            pl.when(pl.col('pz').is_null()).then(pl.col('pz_predict')).otherwise(pl.col('pz')).alias('pz'),
            pl.when(pl.col('in_zone').is_null()).then(pl.col('in_zone_predict')).otherwise(pl.col('in_zone')).alias('in_zone'),
            pl.when(df['launch_speed'].is_null()).then(None).otherwise(df['barrel']).alias('barrel'),
            pl.lit('average').alias('average'),
                pl.when(pl.col('in_zone') == False).then(True).otherwise(False).alias('out_zone'),
            pl.when((pl.col('in_zone') == True) & (pl.col('swings') == 1)).then(True).otherwise(False).alias('zone_swing'),
            pl.when((pl.col('in_zone') == True) & (pl.col('swings') == 1) & (pl.col('whiffs') == 0)).then(True).otherwise(False).alias('zone_contact'),
            pl.when((pl.col('in_zone') == False) & (pl.col('swings') == 1)).then(True).otherwise(False).alias('ozone_swing'),
            pl.when((pl.col('in_zone') == False) & (pl.col('swings') == 1) & (pl.col('whiffs') == 0)).then(True).otherwise(False).alias('ozone_contact'),
            pl.when(pl.col('event_type').str.contains('strikeout')).then(True).otherwise(False).alias('k'),
            pl.when(pl.col('event_type').is_in(['walk', 'intent_walk'])).then(True).otherwise(False).alias('bb'),
            pl.when(pl.col('attack_zone').is_null()).then(pl.col('attack_zone_predict')).otherwise(pl.col('attack_zone')).alias('attack_zone'),
            

        ])

        df = df.with_columns([
            (df['k'].cast(pl.Float32) - df['bb'].cast(pl.Float32)).alias('k_minus_bb'),
            (df['bb'].cast(pl.Float32) - df['k'].cast(pl.Float32)).alias('bb_minus_k'),
            (df['launch_speed'] > 0).alias('bip_div'),
            (df['attack_zone'] == 0).alias('heart'),
            (df['attack_zone'] == 1).alias('shadow'),
            (df['attack_zone'] == 2).alias('chase'),
            (df['attack_zone'] == 3).alias('waste'),
            ((df['attack_zone'] == 0) & (df['swings'] == 1)).alias('heart_swing'),
            ((df['attack_zone'] == 1) & (df['swings'] == 1)).alias('shadow_swing'),
            ((df['attack_zone'] == 2) & (df['swings'] == 1)).alias('chase_swing'),
            ((df['attack_zone'] == 3) & (df['swings'] == 1)).alias('waste_swing'),
            ((df['attack_zone'] == 0) & (df['whiffs'] == 1)).alias('heart_whiff'),
            ((df['attack_zone'] == 1) & (df['whiffs'] == 1)).alias('shadow_whiff'),
            ((df['attack_zone'] == 2) & (df['whiffs'] == 1)).alias('chase_whiff'),
            ((df['attack_zone'] == 3) & (df['whiffs'] == 1)).alias('waste_whiff')
        ])


        [0, 0.689, 0.720, 0.881, 1.254, 1.589, 2.048]

        df = df.with_columns([
                pl.Series(
                    [sum(x) for x in xwoba_model.predict_proba(df[['launch_angle', 'launch_speed']].fill_null(0).to_numpy()[:]) * ([0, 0.881, 1.254, 1.589, 2.048])]
                ).alias('woba_pred_predict')
            ])

        df = df.with_columns([
            pl.when(pl.col('event_type').is_in(['walk'])).then(0.689)
            .when(pl.col('event_type').is_in(['hit_by_pitch'])).then(0.720)
            .when(pl.col('event_type').is_in(['strikeout', 'strikeout_double_play'])).then(0)
            .otherwise(pl.col('woba_pred_predict')).alias('woba_pred_predict')
        ])

        df = df.with_columns([
            pl.when(pl.col('woba_codes').is_null()).then(None).otherwise(pl.col('woba_pred_predict')).alias('woba_pred'),
            pl.when(pl.col('bip')!=1).then(None).otherwise(pl.col('woba_pred_predict')).alias('woba_pred_contact'),
        ])

        df = df.with_columns([
            pl.when(pl.col('trajectory').is_in(['bunt_popup'])).then(pl.lit('popup'))
            .when(pl.col('trajectory').is_in(['bunt_grounder'])).then(pl.lit('ground_ball'))
            .when(pl.col('trajectory').is_in(['bunt_line_drive'])).then(pl.lit('line_drive'))
            .when(pl.col('trajectory').is_in([''])).then(pl.lit(None))
            .otherwise(pl.col('trajectory')).alias('trajectory')
        ])


        # Create one-hot encoded columns for the trajectory column
        dummy_df = df.select(pl.col('trajectory')).to_dummies()

        # Rename the one-hot encoded columns
        dummy_df = dummy_df.rename({
            'trajectory_fly_ball': 'trajectory_fly_ball',
            'trajectory_ground_ball': 'trajectory_ground_ball',
            'trajectory_line_drive': 'trajectory_line_drive',
            'trajectory_popup': 'trajectory_popup'
        })

        # Ensure the columns are present in the DataFrame
        for col in ['trajectory_fly_ball', 'trajectory_ground_ball', 'trajectory_line_drive', 'trajectory_popup']:
            if col not in dummy_df.columns:
                dummy_df = dummy_df.with_columns(pl.lit(0).alias(col))

        # Join the one-hot encoded columns back to the original DataFrame
        df = df.hstack(dummy_df)

        # Check if 'trajectory_null' column exists and drop it
        if 'trajectory_null' in df.columns:
            df = df.drop('trajectory_null')
            
        return df

    # Assuming df is your Polars DataFrame
    def update_summary(self, df: pl.DataFrame, pitcher: bool = True) -> pl.DataFrame:
        """
        Update summary statistics for pitchers or batters.

        Parameters:
        df (pl.DataFrame): The input Polars DataFrame containing player statistics.
        pitcher (bool): A flag indicating whether to calculate statistics for pitchers (True) or batters (False).

        Returns:
        pl.DataFrame: A Polars DataFrame with aggregated and calculated summary statistics.
        """

        # Determine the position based on the pitcher flag
        if pitcher:
            position = 'pitcher'
        else:
            position = 'batter'

        # Group by position_id and position_name, then aggregate various statistics
        df_summ = df.group_by([f'{position}_id', f'{position}_name']).agg([
            pl.col('pa').sum().alias('pa'),
            pl.col('ab').sum().alias('ab'),
            pl.col('obp').sum().alias('obp_pa'),
            pl.col('hits').sum().alias('hits'),
            pl.col('on_base').sum().alias('on_base'),
            pl.col('k').sum().alias('k'),
            pl.col('bb').sum().alias('bb'),
            pl.col('bb_minus_k').sum().alias('bb_minus_k'),
            pl.col('csw').sum().alias('csw'),
            pl.col('bip').sum().alias('bip'),
            pl.col('bip_div').sum().alias('bip_div'),
            pl.col('tb').sum().alias('tb'),
            pl.col('woba').sum().alias('woba'),
            pl.col('woba_contact').sum().alias('woba_contact'),
            pl.col('woba_pred').sum().alias('xwoba'),
            pl.col('woba_pred_contact').sum().alias('xwoba_contact'),
            pl.col('woba_codes').sum().alias('woba_codes'),
            pl.col('xwoba_codes').sum().alias('xwoba_codes'),
            pl.col('hard_hit').sum().alias('hard_hit'),
            pl.col('barrel').sum().alias('barrel'),
            pl.col('sweet_spot').sum().alias('sweet_spot'),
            pl.col('launch_speed').max().alias('max_launch_speed'),
            pl.col('launch_speed').quantile(0.90).alias('launch_speed_90'),
            pl.col('launch_speed').mean().alias('launch_speed'),
            pl.col('launch_angle').mean().alias('launch_angle'),
            pl.col('is_pitch').sum().alias('pitches'),
            pl.col('swings').sum().alias('swings'),
            pl.col('in_zone').sum().alias('in_zone'),
            pl.col('out_zone').sum().alias('out_zone'),
            pl.col('whiffs').sum().alias('whiffs'),
            pl.col('zone_swing').sum().alias('zone_swing'),
            pl.col('zone_contact').sum().alias('zone_contact'),
            pl.col('ozone_swing').sum().alias('ozone_swing'),
            pl.col('ozone_contact').sum().alias('ozone_contact'),
            pl.col('trajectory_ground_ball').sum().alias('ground_ball'),
            pl.col('trajectory_line_drive').sum().alias('line_drive'),
            pl.col('trajectory_fly_ball').sum().alias('fly_ball'),
            pl.col('trajectory_popup').sum().alias('pop_up'),
            pl.col('attack_zone').count().alias('attack_zone'),
            pl.col('heart').sum().alias('heart'),
            pl.col('shadow').sum().alias('shadow'),
            pl.col('chase').sum().alias('chase'),
            pl.col('waste').sum().alias('waste'),
            pl.col('heart_swing').sum().alias('heart_swing'),
            pl.col('shadow_swing').sum().alias('shadow_swing'),
            pl.col('chase_swing').sum().alias('chase_swing'),
            pl.col('waste_swing').sum().alias('waste_swing'),
            pl.col('heart_whiff').sum().alias('heart_whiff'),
            pl.col('shadow_whiff').sum().alias('shadow_whiff'),
            pl.col('chase_whiff').sum().alias('chase_whiff'),
            pl.col('waste_whiff').sum().alias('waste_whiff')
        ])

        # Add calculated columns to the summary DataFrame
        df_summ = df_summ.with_columns([
            (pl.col('hits') / pl.col('ab')).alias('avg'),
            (pl.col('on_base') / pl.col('obp_pa')).alias('obp'),
            (pl.col('tb') / pl.col('ab')).alias('slg'),
            (pl.col('on_base') / pl.col('obp_pa') + pl.col('tb') / pl.col('ab')).alias('ops'),
            (pl.col('k') / pl.col('pa')).alias('k_percent'),
            (pl.col('bb') / pl.col('pa')).alias('bb_percent'),
            (pl.col('bb_minus_k') / pl.col('pa')).alias('bb_minus_k_percent'),
            (pl.col('bb') / pl.col('k')).alias('bb_over_k_percent'),
            (pl.col('csw') / pl.col('pitches')).alias('csw_percent'),
            (pl.col('sweet_spot') / pl.col('bip_div')).alias('sweet_spot_percent'),
            (pl.col('woba') / pl.col('woba_codes')).alias('woba_percent'),
            (pl.col('woba_contact') / pl.col('bip')).alias('woba_percent_contact'),
            (pl.col('hard_hit') / pl.col('bip_div')).alias('hard_hit_percent'),
            (pl.col('barrel') / pl.col('bip_div')).alias('barrel_percent'),
            (pl.col('zone_contact') / pl.col('zone_swing')).alias('zone_contact_percent'),
            (pl.col('zone_swing') / pl.col('in_zone')).alias('zone_swing_percent'),
            (pl.col('in_zone') / pl.col('pitches')).alias('zone_percent'),
            (pl.col('ozone_swing') / (pl.col('pitches') - pl.col('in_zone'))).alias('chase_percent'),
            (pl.col('ozone_contact') / pl.col('ozone_swing')).alias('chase_contact'),
            (pl.col('swings') / pl.col('pitches')).alias('swing_percent'),
            (pl.col('whiffs') / pl.col('swings')).alias('whiff_rate'),
            (pl.col('whiffs') / pl.col('pitches')).alias('swstr_rate'),
            (pl.col('ground_ball') / pl.col('bip')).alias('ground_ball_percent'),
            (pl.col('line_drive') / pl.col('bip')).alias('line_drive_percent'),
            (pl.col('fly_ball') / pl.col('bip')).alias('fly_ball_percent'),
            (pl.col('pop_up') / pl.col('bip')).alias('pop_up_percent'),
            (pl.col('heart') / pl.col('attack_zone')).alias('heart_zone_percent'),
            (pl.col('shadow') / pl.col('attack_zone')).alias('shadow_zone_percent'),
            (pl.col('chase') / pl.col('attack_zone')).alias('chase_zone_percent'),
            (pl.col('waste') / pl.col('attack_zone')).alias('waste_zone_percent'),
            (pl.col('heart_swing') / pl.col('heart')).alias('heart_zone_swing_percent'),
            (pl.col('shadow_swing') / pl.col('shadow')).alias('shadow_zone_swing_percent'),
            (pl.col('chase_swing') / pl.col('chase')).alias('chase_zone_swing_percent'),
            (pl.col('waste_swing') / pl.col('waste')).alias('waste_zone_swing_percent'),
            (pl.col('heart_whiff') / pl.col('heart_swing')).alias('heart_zone_whiff_percent'),
            (pl.col('shadow_whiff') / pl.col('shadow_swing')).alias('shadow_zone_whiff_percent'),
            (pl.col('chase_whiff') / pl.col('chase_swing')).alias('chase_zone_whiff_percent'),
            (pl.col('waste_whiff') / pl.col('waste_swing')).alias('waste_zone_whiff_percent'),
            (pl.col('xwoba') / pl.col('xwoba_codes')).alias('xwoba_percent'),
            (pl.col('xwoba_contact') / pl.col('bip')).alias('xwoba_percent_contact')
        ])

        return df_summ
    




    
    # Assuming df is your Polars DataFrame
    def update_summary_select(self, df: pl.DataFrame, selection: list) -> pl.DataFrame:
        """
        Update summary statistics for pitchers or batters.

        Parameters:
        df (pl.DataFrame): The input Polars DataFrame containing player statistics.
        pitcher (bool): A flag indicating whether to calculate statistics for pitchers (True) or batters (False).

        Returns:
        pl.DataFrame: A Polars DataFrame with aggregated and calculated summary statistics.
        """

        # Group by position_id and position_name, then aggregate various statistics
        df_summ = df.group_by(selection).agg([
            pl.col('pa').sum().alias('pa'),
            pl.col('ab').sum().alias('ab'),
            pl.col('obp').sum().alias('obp_pa'),
            pl.col('hits').sum().alias('hits'),
            pl.col('on_base').sum().alias('on_base'),
            pl.col('k').sum().alias('k'),
            pl.col('bb').sum().alias('bb'),
            pl.col('bb_minus_k').sum().alias('bb_minus_k'),
            pl.col('csw').sum().alias('csw'),
            pl.col('bip').sum().alias('bip'),
            pl.col('bip_div').sum().alias('bip_div'),
            pl.col('tb').sum().alias('tb'),
            pl.col('woba').sum().alias('woba'),
            pl.col('woba_contact').sum().alias('woba_contact'),
            pl.col('woba_pred').sum().alias('xwoba'),
            pl.col('woba_pred_contact').sum().alias('xwoba_contact'),
            pl.col('woba_codes').sum().alias('woba_codes'),
            pl.col('xwoba_codes').sum().alias('xwoba_codes'),
            pl.col('hard_hit').sum().alias('hard_hit'),
            pl.col('barrel').sum().alias('barrel'),
            pl.col('sweet_spot').sum().alias('sweet_spot'),
            pl.col('launch_speed').max().alias('max_launch_speed'),
            pl.col('launch_speed').quantile(0.90).alias('launch_speed_90'),
            pl.col('launch_speed').mean().alias('launch_speed'),
            pl.col('launch_angle').mean().alias('launch_angle'),
            pl.col('is_pitch').sum().alias('pitches'),
            pl.col('swings').sum().alias('swings'),
            pl.col('in_zone').sum().alias('in_zone'),
            pl.col('out_zone').sum().alias('out_zone'),
            pl.col('whiffs').sum().alias('whiffs'),
            pl.col('zone_swing').sum().alias('zone_swing'),
            pl.col('zone_contact').sum().alias('zone_contact'),
            pl.col('ozone_swing').sum().alias('ozone_swing'),
            pl.col('ozone_contact').sum().alias('ozone_contact'),
            pl.col('trajectory_ground_ball').sum().alias('ground_ball'),
            pl.col('trajectory_line_drive').sum().alias('line_drive'),
            pl.col('trajectory_fly_ball').sum().alias('fly_ball'),
            pl.col('trajectory_popup').sum().alias('pop_up'),
            pl.col('attack_zone').count().alias('attack_zone'),
            pl.col('heart').sum().alias('heart'),
            pl.col('shadow').sum().alias('shadow'),
            pl.col('chase').sum().alias('chase'),
            pl.col('waste').sum().alias('waste'),
            pl.col('heart_swing').sum().alias('heart_swing'),
            pl.col('shadow_swing').sum().alias('shadow_swing'),
            pl.col('chase_swing').sum().alias('chase_swing'),
            pl.col('waste_swing').sum().alias('waste_swing'),
            pl.col('heart_whiff').sum().alias('heart_whiff'),
            pl.col('shadow_whiff').sum().alias('shadow_whiff'),
            pl.col('chase_whiff').sum().alias('chase_whiff'),
            pl.col('waste_whiff').sum().alias('waste_whiff')
        ])

        # Add calculated columns to the summary DataFrame
        df_summ = df_summ.with_columns([
            (pl.col('hits') / pl.col('ab')).alias('avg'),
            (pl.col('on_base') / pl.col('obp_pa')).alias('obp'),
            (pl.col('tb') / pl.col('ab')).alias('slg'),
            (pl.col('on_base') / pl.col('obp_pa') + pl.col('tb') / pl.col('ab')).alias('ops'),
            (pl.col('k') / pl.col('pa')).alias('k_percent'),
            (pl.col('bb') / pl.col('pa')).alias('bb_percent'),
            (pl.col('bb_minus_k') / pl.col('pa')).alias('bb_minus_k_percent'),
            (pl.col('bb') / pl.col('k')).alias('bb_over_k_percent'),
            (pl.col('csw') / pl.col('pitches')).alias('csw_percent'),
            (pl.col('sweet_spot') / pl.col('bip_div')).alias('sweet_spot_percent'),
            (pl.col('woba') / pl.col('woba_codes')).alias('woba_percent'),
            (pl.col('woba_contact') / pl.col('bip')).alias('woba_percent_contact'),
            (pl.col('hard_hit') / pl.col('bip_div')).alias('hard_hit_percent'),
            (pl.col('barrel') / pl.col('bip_div')).alias('barrel_percent'),
            (pl.col('zone_contact') / pl.col('zone_swing')).alias('zone_contact_percent'),
            (pl.col('zone_swing') / pl.col('in_zone')).alias('zone_swing_percent'),
            (pl.col('in_zone') / pl.col('pitches')).alias('zone_percent'),
            (pl.col('ozone_swing') / (pl.col('pitches') - pl.col('in_zone'))).alias('chase_percent'),
            (pl.col('ozone_contact') / pl.col('ozone_swing')).alias('chase_contact'),
            (pl.col('swings') / pl.col('pitches')).alias('swing_percent'),
            (pl.col('whiffs') / pl.col('swings')).alias('whiff_rate'),
            (pl.col('whiffs') / pl.col('pitches')).alias('swstr_rate'),
            (pl.col('ground_ball') / pl.col('bip')).alias('ground_ball_percent'),
            (pl.col('line_drive') / pl.col('bip')).alias('line_drive_percent'),
            (pl.col('fly_ball') / pl.col('bip')).alias('fly_ball_percent'),
            (pl.col('pop_up') / pl.col('bip')).alias('pop_up_percent'),
            (pl.col('heart') / pl.col('attack_zone')).alias('heart_zone_percent'),
            (pl.col('shadow') / pl.col('attack_zone')).alias('shadow_zone_percent'),
            (pl.col('chase') / pl.col('attack_zone')).alias('chase_zone_percent'),
            (pl.col('waste') / pl.col('attack_zone')).alias('waste_zone_percent'),
            (pl.col('heart_swing') / pl.col('heart')).alias('heart_zone_swing_percent'),
            (pl.col('shadow_swing') / pl.col('shadow')).alias('shadow_zone_swing_percent'),
            (pl.col('chase_swing') / pl.col('chase')).alias('chase_zone_swing_percent'),
            (pl.col('waste_swing') / pl.col('waste')).alias('waste_zone_swing_percent'),
            (pl.col('heart_whiff') / pl.col('heart_swing')).alias('heart_zone_whiff_percent'),
            (pl.col('shadow_whiff') / pl.col('shadow_swing')).alias('shadow_zone_whiff_percent'),
            (pl.col('chase_whiff') / pl.col('chase_swing')).alias('chase_zone_whiff_percent'),
            (pl.col('waste_whiff') / pl.col('waste_swing')).alias('waste_zone_whiff_percent'),
            (pl.col('xwoba') / pl.col('xwoba_codes')).alias('xwoba_percent'),
            (pl.col('xwoba_contact') / pl.col('bip')).alias('xwoba_percent_contact')
        ])

        return df_summ
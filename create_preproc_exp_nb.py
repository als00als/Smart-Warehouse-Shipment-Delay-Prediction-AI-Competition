import copy
import glob
import json
from pathlib import Path


def _to_source_lines(text: str):
    lines = text.split("\n")
    if not lines:
        return []
    out = [f"{line}\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def main():
    matches = glob.glob("*0415.ipynb")
    if not matches:
        raise FileNotFoundError("Could not find *0415.ipynb in workspace root")

    src = matches[0]
    with open(src, "r", encoding="utf-8") as f:
        nb = json.load(f)

    new_nb = copy.deepcopy(nb)

    cell2 = "".join(new_nb["cells"][2].get("source", []))
    if "PREPROC_ONLY_EXPERIMENT" not in cell2:
        cfg_add = """

PREPROC_ONLY_EXPERIMENT = True
ENABLE_DOMAIN_CLIP = True
ENABLE_MISSING_MASK = True

# ratio feature/column name hints
RATIO_COL_KEYWORDS = ['ratio', 'utilization', 'efficiency', 'pct', 'percent', 'share']

# non-negative metric name hints
NONNEG_COL_KEYWORDS = [
    'count', 'cnt', 'num', 'minutes', 'min', 'hour', 'distance',
    'delay', 'inflow', 'order', 'robot', 'battery', 'density',
    'area', 'weight', 'pressure', 'risk', 'wait', 'trip'
]
"""
        cell2 = cell2 + cfg_add
        new_nb["cells"][2]["source"] = _to_source_lines(cell2)

    preproc_cell = """def add_missing_indicators(df):
    \"\"\"Create binary mask columns for missing values (excluding IDs/target).\"\"\"
    miss_cols = [c for c in df.columns if c not in ID_COLS + [TARGET] and df[c].isnull().any()]
    for c in miss_cols:
        df[f'{c}_isna'] = df[c].isnull().astype(np.int8)
    return df


def handle_missing_values(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    cols = [c for c in df.columns if df[c].isnull().any() and c not in ID_COLS + [TARGET]]
    if cols:
        # scenario-local directional fill first
        df[cols] = df.groupby('scenario_id')[cols].ffill()
        df[cols] = df.groupby('scenario_id')[cols].bfill()
        # robust fallback
        df[cols] = df[cols].fillna(df[cols].median())
    return df


def clip_domain_values(df):
    \"\"\"
    Apply simple physical constraints:
      - ratio-like columns -> [0, 1]
      - non-negative metrics -> [0, +inf)
    \"\"\"
    for c in df.columns:
        if c in ID_COLS + [TARGET]:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        lc = c.lower()
        if any(k in lc for k in RATIO_COL_KEYWORDS):
            df[c] = df[c].clip(0.0, 1.0)
            continue

        if any(k in lc for k in NONNEG_COL_KEYWORDS):
            df[c] = df[c].clip(lower=0.0)
    return df


def add_basic_features(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    df['timeslot'] = df.groupby('scenario_id').cumcount()
    df['robot_efficiency'] = df['robot_active'] / (df['robot_total'] + 1e-6)
    df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1e-6)
    df['order_per_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1e-6)
    df['robot_per_station'] = df['robot_active'] / (df['pack_station_count'] + 1e-6)
    df['cumulative_orders'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum()
    df['order_pressure'] = df['cumulative_orders'] / (df['pack_station_count'] + 1e-6)
    if 'congestion_score' in df.columns:
        df['risk_index'] = df['congestion_score'] * (1 - df['robot_efficiency'])
        df['bottle_neck'] = df['order_per_station'] * df['congestion_score']
    if 'low_battery_ratio' in df.columns:
        df['battery_risk'] = df['low_battery_ratio'] * df['robot_total']
    if 'battery_mean' in df.columns and 'battery_std' in df.columns:
        df['battery_cv'] = df['battery_std'] / (df['battery_mean'] + 1e-6)
    return df


TS_COLS = [
    'order_inflow_15m', 'robot_active', 'robot_idle', 'robot_total',
    'pack_utilization', 'congestion_score', 'avg_trip_distance',
    'low_battery_ratio', 'outbound_truck_wait_min',
    'order_per_station', 'robot_efficiency', 'order_pressure',
    'max_zone_density', 'sku_concentration',
    'battery_risk', 'battery_cv', 'risk_index', 'bottle_neck',
]


def add_timeseries_features(df):
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    for col in TS_COLS:
        if col not in df.columns:
            continue
        g = df.groupby('scenario_id')[col]
        for lag_n in (1, 2, 3, 4, 5):
            df[f'{col}_lag{lag_n}'] = g.shift(lag_n)
        df[f'{col}_diff1'] = g.shift(1) - g.shift(2)
        df[f'{col}_diff2'] = g.shift(2) - g.shift(3)
        for w in (3, 5, 10):
            df[f'{col}_roll{w}_mean'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
            df[f'{col}_roll{w}_std'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
            df[f'{col}_roll{w}_max'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).max())
            df[f'{col}_roll{w}_min'] = g.transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).min())
        df[f'{col}_exp_mean'] = g.transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        df[f'{col}_ewm_mean'] = g.transform(lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean())

    lag_cols = [c for c in df.columns if ('_lag' in c or '_diff' in c) and c not in ID_COLS]
    if lag_cols:
        df[lag_cols] = df.groupby('scenario_id')[lag_cols].ffill()
        for c in lag_cols:
            base_col = c.split('_lag')[0].split('_diff')[0]
            if base_col in df.columns:
                scen_mean = df.groupby('scenario_id')[base_col].transform('mean')
                df[c] = df[c].fillna(scen_mean)
        df[lag_cols] = df[lag_cols].fillna(df[lag_cols].median())
    return df


def add_interaction_features(df):
    if 'congestion_score' in df.columns and 'pack_utilization' in df.columns:
        df['cong_x_pack'] = df['congestion_score'] * df['pack_utilization']
    if 'congestion_score' in df.columns and 'avg_trip_distance' in df.columns:
        df['cong_x_trip'] = df['congestion_score'] * df['avg_trip_distance']
    if 'low_battery_ratio' in df.columns and 'robot_efficiency' in df.columns:
        df['lowbat_x_eff'] = df['low_battery_ratio'] * (1 - df['robot_efficiency'])
    if 'order_per_station' in df.columns and 'congestion_score' in df.columns:
        df['ops_x_cong'] = df['order_per_station'] * df['congestion_score']
    if 'order_per_station' in df.columns and 'pack_utilization' in df.columns:
        df['ops_x_pack'] = df['order_per_station'] * df['pack_utilization']
    if 'timeslot' in df.columns:
        for col in ['congestion_score', 'pack_utilization', 'order_per_station', 'low_battery_ratio']:
            if col in df.columns:
                df[f'ts_x_{col}'] = df['timeslot'] * df[col]

    scen_agg_cols = [
        'congestion_score', 'order_inflow_15m', 'battery_mean', 'pack_utilization',
        'avg_trip_distance', 'low_battery_ratio', 'max_zone_density', 'sku_concentration',
        'robot_idle', 'outbound_truck_wait_min', 'order_per_station', 'robot_efficiency',
        'order_pressure', 'risk_index', 'battery_risk', 'battery_cv',
    ]
    for col in scen_agg_cols:
        if col not in df.columns:
            continue
        stats = df.groupby('scenario_id')[col].agg(['mean', 'max', 'min', 'std']).reset_index()
        stats.columns = ['scenario_id'] + [f'{col}_scen_{f}' for f in ['mean', 'max', 'min', 'std']]
        df = df.merge(stats, on='scenario_id', how='left')

    for col in ['congestion_score', 'order_per_station', 'pack_utilization', 'avg_trip_distance']:
        sm = f'{col}_scen_mean'
        if col in df.columns and sm in df.columns:
            df[f'{col}_rel_to_scen'] = df[col] / (df[sm] + 1e-6)

    for col in ['congestion_score', 'order_per_station', 'pack_utilization']:
        if col in df.columns:
            df[f'{col}_scen_rank'] = df.groupby('scenario_id')[col].rank(pct=True)
    return df


def add_extra_features_v13(df):
    \"\"\"Scenario progress + periodic features (train/test shared).\"\"\"
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    gsz = df.groupby('scenario_id')['scenario_id'].transform('size')
    gpos = df.groupby('scenario_id').cumcount()
    df['v13_row_frac_in_scen'] = (gpos + 1) / (gsz + 1e-6)
    if 'order_inflow_15m' in df.columns:
        df['v13_sqrt_order_inflow'] = np.sqrt(np.maximum(df['order_inflow_15m'].astype(np.float64), 0.0))
    if 'robot_total' in df.columns:
        df['v13_sqrt_robot_total'] = np.sqrt(np.maximum(df['robot_total'].astype(np.float64), 0.0))
    if 'timeslot' in df.columns:
        ang = 2 * np.pi * df['timeslot'].astype(np.float64) / 24.0
        df['v13_ts_sin'] = np.sin(ang)
        df['v13_ts_cos'] = np.cos(ang)
    return df


def preprocess_all(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')

    if ENABLE_MISSING_MASK:
        df = add_missing_indicators(df)

    df = handle_missing_values(df)
    df = add_basic_features(df)
    df = add_timeseries_features(df)
    df = add_interaction_features(df)
    df = add_extra_features_v13(df)

    if ENABLE_DOMAIN_CLIP:
        df = clip_domain_values(df)

    if 'layout_type' in df.columns:
        df['layout_type'] = pd.factorize(df['layout_type'])[0]
    return df


section('Preprocess (preproc-only experiment: clip + missing mask)')
t0 = time.time()
train = preprocess_all(train, layout)
test = preprocess_all(test, layout)
print(f"▶ preprocess done ({elapsed(t0)})")
if PREPROC_ONLY_EXPERIMENT:
    print('▶ preproc-only experiment mode: model/CV/ensemble logic unchanged')
"""
    new_nb["cells"][12]["source"] = _to_source_lines(preproc_cell)

    src_path = Path(src)
    dst_name = src_path.name.replace("_0415.ipynb", "_0415_preproc_only_exp.ipynb")
    dst = src_path.with_name(dst_name)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, ensure_ascii=False, indent=1)

    print(str(dst))


if __name__ == "__main__":
    main()

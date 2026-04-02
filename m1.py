import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SUPABASE_CONN_STR = "postgresql://postgres.pawoxtsikeepvgiemush:7hyuKbHqNRsxadPt@aws-1-us-east-2.pooler.supabase.com:5432/postgres"

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
	{
		"model_version": "weather_temp_auto_20260402_115700",
		"data_start": "2026-03-01 00:00:00+00",
		"data_end_exclusive": "2026-03-03 00:00:00+00",
		"label": "2 dias"
	},
	{
		"model_version": "weather_temp_auto_20260402_115800",
		"data_start": "2026-03-01 00:00:00+00",
		"data_end_exclusive": "2026-03-04 00:00:00+00",
		"label": "3 dias"
	},
	{
		"model_version": "weather_temp_auto_20260402_115900",
		"data_start": "2026-03-01 00:00:00+00",
		"data_end_exclusive": "2026-03-05 00:00:00+00",
		"label": "4 dias"
	},
]


def load_data(data_start, data_end_exclusive):
	query = """
		select
			observation_time,
			temperature_2m,
			relative_humidity_2m,
			precipitation,
			wind_speed_10m
		from public.weather_hourly
		where observation_time >= %s
		and observation_time < %s
		order by observation_time;
	"""

	with psycopg.connect(SUPABASE_CONN_STR) as conn:
		with conn.cursor() as cur:
			cur.execute(query, (data_start, data_end_exclusive))
			rows = cur.fetchall()
			cols = [desc.name for desc in cur.description]

	df = pd.DataFrame(rows, columns=cols)

	if df.empty:
		raise ValueError(f"No hay datos en el rango {data_start} a {data_end_exclusive}")

	return df


def build_supervised_dataset(df):
	data = df.copy()

	data["observation_time"] = pd.to_datetime(data["observation_time"], utc=True)
	data["hour"] = data["observation_time"].dt.hour

	data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
	data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

	data["temp_lag_1"] = data["temperature_2m"].shift(1)
	data["temp_lag_2"] = data["temperature_2m"].shift(2)
	data["temp_lag_3"] = data["temperature_2m"].shift(3)

	data["target_temperature_next_hour"] = data["temperature_2m"].shift(-1)

	feature_cols = [
		"temperature_2m",
		"relative_humidity_2m",
		"precipitation",
		"wind_speed_10m",
		"hour_sin",
		"hour_cos",
		"temp_lag_1",
		"temp_lag_2",
		"temp_lag_3",
	]

	data = data.dropna(subset=feature_cols + ["target_temperature_next_hour"]).reset_index(drop=True)

	if len(data) < 12:
		raise ValueError(f"Muy pocas filas utiles para entrenar: {len(data)}")

	return data, feature_cols


def split_train_test(data, train_ratio=0.8):
	split_idx = int(len(data) * train_ratio)

	if split_idx <= 0:
		split_idx = 1

	if split_idx >= len(data):
		split_idx = len(data) - 1

	train_df = data.iloc[:split_idx].copy()
	test_df = data.iloc[split_idx:].copy()

	if train_df.empty or test_df.empty:
		raise ValueError("No se pudo crear un split valido de train/test.")

	return train_df, test_df


def train_and_evaluate(train_df, test_df, feature_cols):
	model = RandomForestRegressor(
		n_estimators=200,
		max_depth=6,
		min_samples_leaf=2,
		random_state=42
	)

	X_train = train_df[feature_cols]
	y_train = train_df["target_temperature_next_hour"]

	X_test = test_df[feature_cols]
	y_test = test_df["target_temperature_next_hour"]

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	mae = mean_absolute_error(y_test, y_pred)
	rmse = mean_squared_error(y_test, y_pred) ** 0.5
	r2 = r2_score(y_test, y_pred)

	return model, mae, rmse, r2, len(X_train), len(X_test), len(train_df) + len(test_df)


def save_model(model, model_version):
	model_path = MODEL_DIR / f"{model_version}.pkl"
	joblib.dump(model, model_path)
	return model_path


def run_training(config):
	raw_df = load_data(config["data_start"], config["data_end_exclusive"])
	data, feature_cols = build_supervised_dataset(raw_df)
	train_df, test_df = split_train_test(data, train_ratio=0.8)

	model, mae, rmse, r2, train_rows, test_rows, total_rows = train_and_evaluate(
		train_df,
		test_df,
		feature_cols
	)

	save_model(model, config["model_version"])

	return {
		"label": config["label"],
		"model_version": config["model_version"],
		"raw_rows": len(raw_df),
		"total_rows": total_rows,
		"train_rows": train_rows,
		"test_rows": test_rows,
		"mae": mae,
		"rmse": rmse,
		"r2": r2,
	}


def main():
	if not SUPABASE_CONN_STR or "PEGA_AQUI" in SUPABASE_CONN_STR:
		raise ValueError("Debes pegar una connection string valida en SUPABASE_CONN_STR.")

	results = []

	for config in MODEL_CONFIGS:
		result = run_training(config)
		results.append(result)

	for result in results:
		print("=" * 50)
		print(f"Ventana: {result['label']}")
		print(f"Modelo: {result['model_version']}")
		print(f"Filas crudas: {result['raw_rows']}")
		print(f"Filas utiles: {result['total_rows']}")
		print(f"Train: {result['train_rows']}")
		print(f"Test: {result['test_rows']}")
		print(f"MAE: {result['mae']:.4f}")
		print(f"RMSE: {result['rmse']:.4f}")
		print(f"R2: {result['r2']:.4f}")


if __name__ == "__main__":
	main()

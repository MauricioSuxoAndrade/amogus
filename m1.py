import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SUPABASE_CONN_STR = "postgresql://postgres.pawoxtsikeepvgiemush:7hyuKbHqNRsxadPt@aws-1-us-east-2.pooler.supabase.com:5432/postgres"

MODEL_VERSION = "weather_temp_v0"
DATA_START = "2026-03-01 00:00:00+00"
DATA_END_EXCLUSIVE = "2026-03-02 00:00:00+00"

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / f"{MODEL_VERSION}.pkl"


def load_data():
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
			cur.execute(query, (DATA_START, DATA_END_EXCLUSIVE))
			rows = cur.fetchall()
			cols = [desc.name for desc in cur.description]

	df = pd.DataFrame(rows, columns=cols)

	if df.empty:
		raise ValueError("No hay datos en ese rango para entrenar el modelo.")

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
		raise ValueError(f"Muy pocas filas útiles para entrenar: {len(data)}")

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
		raise ValueError("No se pudo crear un split válido de train/test.")

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


def save_model(model):
	joblib.dump(model, MODEL_PATH)


def main():
	if not SUPABASE_CONN_STR or "PEGA_AQUI" in SUPABASE_CONN_STR:
		raise ValueError("Debes pegar una connection string válida de Supabase en SUPABASE_CONN_STR.")

	raw_df = load_data()
	data, feature_cols = build_supervised_dataset(raw_df)
	train_df, test_df = split_train_test(data, train_ratio=0.8)

	model, mae, rmse, r2, train_rows, test_rows, total_rows = train_and_evaluate(
		train_df,
		test_df,
		feature_cols
	)

	save_model(model)

	print(f"Modelo entrenado: {MODEL_VERSION}")
	print(f"Filas crudas leídas: {len(raw_df)}")
	print(f"Filas útiles supervisadas: {total_rows}")
	print(f"Filas train: {train_rows}")
	print(f"Filas test: {test_rows}")
	print(f"MAE: {mae:.4f}")
	print(f"RMSE: {rmse:.4f}")
	print(f"R2: {r2:.4f}")
	print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
	main()

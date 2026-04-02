import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SUPABASE_CONN_STR = os.getenv("postgresql://postgres.pawoxtsikeepvgiemush:7hyuKbHqNRsxadPt@aws-1-us-east-2.pooler.supabase.com:5432/postgres")

MODEL_VERSION = "weather_temp_v1"
DATA_START = "2026-03-01 00:00:00+00"
DATA_END_EXCLUSIVE = "2026-03-03 00:00:00+00"
DATA_END_LABEL = "2026-03-02 23:00:00+00"

BASE_DIR = Path("/opt/airflow")
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / f"{MODEL_VERSION}.pkl"
PLOT_SERIES_PATH = PLOT_DIR / f"{MODEL_VERSION}_series_split.png"
PLOT_PRED_PATH = PLOT_DIR / f"{MODEL_VERSION}_actual_vs_pred.png"
PLOT_RESIDUALS_PATH = PLOT_DIR / f"{MODEL_VERSION}_residuals.png"
PLOT_IMPORTANCE_PATH = PLOT_DIR / f"{MODEL_VERSION}_feature_importance.png"


def ensure_tracking_tables(conn):
	with conn.cursor() as cur:
		cur.execute("""
			create table if not exists public.model_registry (
				id bigserial primary key,
				model_version text unique not null,
				trained_at timestamptz default now(),
				data_start timestamptz not null,
				data_end timestamptz not null,
				rows_used int not null,
				model_path text not null,
				is_active boolean default false
			);
		""")

		cur.execute("""
			create table if not exists public.model_metrics (
				id bigserial primary key,
				model_version text unique not null,
				evaluated_at timestamptz default now(),
				mae double precision,
				rmse double precision,
				r2 double precision
			);
		""")

	conn.commit()


def load_data(conn):
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

	df = pd.read_sql(query, conn, params=(DATA_START, DATA_END_EXCLUSIVE))

	if df.empty:
		raise ValueError("No hay datos para entrenar el modelo v1 en ese rango.")

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
		raise ValueError("No se pudo crear un split temporal válido de train/test.")

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

	return model, y_test, y_pred, mae, rmse, r2


def save_model(model):
	joblib.dump(model, MODEL_PATH)


def save_registry_and_metrics(conn, rows_used, mae, rmse, r2):
	with conn.cursor() as cur:
		cur.execute("""
			update public.model_registry
			set is_active = false
			where is_active = true;
		""")

		cur.execute("""
			insert into public.model_registry (
				model_version,
				data_start,
				data_end,
				rows_used,
				model_path,
				is_active
			)
			values (%s, %s, %s, %s, %s, %s)
			on conflict (model_version) do update
			set
				trained_at = now(),
				data_start = excluded.data_start,
				data_end = excluded.data_end,
				rows_used = excluded.rows_used,
				model_path = excluded.model_path,
				is_active = excluded.is_active;
		""", (
			MODEL_VERSION,
			DATA_START,
			DATA_END_LABEL,
			int(rows_used),
			str(MODEL_PATH),
			True
		))

		cur.execute("""
			insert into public.model_metrics (
				model_version,
				mae,
				rmse,
				r2
			)
			values (%s, %s, %s, %s)
			on conflict (model_version) do update
			set
				evaluated_at = now(),
				mae = excluded.mae,
				rmse = excluded.rmse,
				r2 = excluded.r2;
		""", (
			MODEL_VERSION,
			float(mae),
			float(rmse),
			float(r2)
		))

	conn.commit()


def plot_series_with_split(data, train_df):
	plt.figure(figsize=(12, 5))
	plt.plot(data["observation_time"], data["temperature_2m"], label="temperature_2m")
	plt.axvline(train_df["observation_time"].iloc[-1], linestyle="--", label="split train/test")
	plt.title("v1 - Serie temporal usada")
	plt.xlabel("Tiempo")
	plt.ylabel("Temperatura")
	plt.legend()
	plt.tight_layout()
	plt.savefig(PLOT_SERIES_PATH)
	plt.close()


def plot_actual_vs_predicted(test_df, y_test, y_pred):
	plt.figure(figsize=(12, 5))
	plt.plot(test_df["observation_time"], y_test.to_numpy(), label="Real")
	plt.plot(test_df["observation_time"], y_pred, label="Predicción")
	plt.title("v1 - Temperatura siguiente hora: real vs predicción")
	plt.xlabel("Tiempo")
	plt.ylabel("Temperatura")
	plt.legend()
	plt.tight_layout()
	plt.savefig(PLOT_PRED_PATH)
	plt.close()


def plot_residuals(test_df, y_test, y_pred):
	residuals = y_test.to_numpy() - y_pred

	plt.figure(figsize=(12, 5))
	plt.plot(test_df["observation_time"], residuals)
	plt.axhline(0, linestyle="--")
	plt.title("v1 - Residuales")
	plt.xlabel("Tiempo")
	plt.ylabel("Error (real - predicción)")
	plt.tight_layout()
	plt.savefig(PLOT_RESIDUALS_PATH)
	plt.close()


def plot_feature_importance(model, feature_cols):
	importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

	plt.figure(figsize=(10, 5))
	plt.barh(importances.index, importances.values)
	plt.title("v1 - Importancia de variables")
	plt.xlabel("Importancia")
	plt.tight_layout()
	plt.savefig(PLOT_IMPORTANCE_PATH)
	plt.close()


def main():
	if not SUPABASE_CONN_STR:
		raise ValueError("Falta la variable de entorno SUPABASE_CONN_STR")

	with psycopg.connect(SUPABASE_CONN_STR) as conn:
		ensure_tracking_tables(conn)

		raw_df = load_data(conn)
		data, feature_cols = build_supervised_dataset(raw_df)
		train_df, test_df = split_train_test(data, train_ratio=0.8)

		model, y_test, y_pred, mae, rmse, r2 = train_and_evaluate(train_df, test_df, feature_cols)

		save_model(model)
		save_registry_and_metrics(conn, len(data), mae, rmse, r2)

	plot_series_with_split(data, train_df)
	plot_actual_vs_predicted(test_df, y_test, y_pred)
	plot_residuals(test_df, y_test, y_pred)
	plot_feature_importance(model, feature_cols)

	print(f"Modelo entrenado: {MODEL_VERSION}")
	print(f"Filas crudas cargadas: {len(raw_df)}")
	print(f"Filas útiles supervisadas: {len(data)}")
	print(f"Train: {len(train_df)}")
	print(f"Test: {len(test_df)}")
	print(f"MAE: {mae:.4f}")
	print(f"RMSE: {rmse:.4f}")
	print(f"R2: {r2:.4f}")
	print(f"Modelo guardado en: {MODEL_PATH}")
	print("Gráficas guardadas en:")
	print(f"- {PLOT_SERIES_PATH}")
	print(f"- {PLOT_PRED_PATH}")
	print(f"- {PLOT_RESIDUALS_PATH}")
	print(f"- {PLOT_IMPORTANCE_PATH}")


if __name__ == "__main__":
	main()

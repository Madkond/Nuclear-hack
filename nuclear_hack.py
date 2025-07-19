import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import time

# 1. Загрузка данных
df = pd.read_csv('result.csv')

# 2. Очистка данных
df = df.dropna(subset=['tetta', 'phi', 'x', 'y', 'power', 'age'])

# 3. Обработка кортежей
def expand_tuple_column(df, col_name):
    if col_name in df.columns:
        # Извлекаем кортежи
        tuples = df[col_name].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Создаем временный DataFrame с развернутыми значениями
        expanded = pd.DataFrame(tuples.tolist())
        expanded.columns = [f'{col_name}_{i}' for i in range(expanded.shape[1])]
        
        # Удаляем исходный столбец и добавляем развернутые
        df = df.drop(col_name, axis=1)
        df = pd.concat([df, expanded], axis=1)
    return df

# Применяем к столбцам с кортежами
df = expand_tuple_column(df, 'energy')
df = expand_tuple_column(df, 'threshold_time')

# 4. Циклическое преобразование азимута
df['phi_sin'] = np.sin(np.radians(df['phi']))
df['phi_cos'] = np.cos(np.radians(df['phi']))

# 5. Подготовка признаков
# Выбираем только числовые столбцы
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].drop(['tetta', 'phi', 'x', 'y', 'power', 'age', 'phi_sin', 'phi_cos'], axis=1, errors='ignore')

# Заполнение пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# 6. Подготовка переменных
targets = {
    'direction': df[['tetta', 'phi_sin', 'phi_cos']],
    'position': df[['x', 'y']],
    'power': df['power'],
    'age': df['age']
}

# 7. Разделение данных на обучающую и тестовую выборки
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# 8. Параметры моделей
rf_params = {
    'n_estimators': 100,  # Увеличено количество деревьев
    'max_depth': 15,       # Увеличена глубина
    'min_samples_split': 5,
    'random_state': 42,
    'n_jobs': -1
}

gb_params = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': 42
}

# 9. Обучение моделей и оценка

# 9.1. Направление прихода (tetta, phi_sin, phi_cos)
print("="*60)
print("Обучение модели для направления прихода...")
start_time = time.time()

y_direction_train = targets['direction'].iloc[X_train.index]
y_direction_test = targets['direction'].iloc[X_test.index]

direction_model = MultiOutputRegressor(RandomForestRegressor(**rf_params))
direction_model.fit(X_train, y_direction_train)
y_pred_direction = direction_model.predict(X_test)

# Преобразование предсказанных sin/cos обратно в угол
phi_sin_pred = y_pred_direction[:, 1]
phi_cos_pred = y_pred_direction[:, 2]
phi_pred = np.degrees(np.arctan2(phi_sin_pred, phi_cos_pred)) % 360

# Оценка для зенитного угла
mse_tetta = mean_squared_error(y_direction_test['tetta'], y_pred_direction[:, 0])
r2_tetta = r2_score(y_direction_test['tetta'], y_pred_direction[:, 0])

# Оценка для азимута (после обратного преобразования)
true_phi = df.loc[X_test.index, 'phi']
mse_phi = mean_squared_error(true_phi, phi_pred)
r2_phi = r2_score(true_phi, phi_pred)

print(f"Время обучения: {time.time()-start_time:.1f} сек")
print("Направление прихода:")
print(f"MSE зенитного угла (tetta): {mse_tetta:.4f}, R²: {r2_tetta:.4f}")
print(f"MSE азимута (phi): {mse_phi:.4f}, R²: {r2_phi:.4f}")

# 9.2. Положение оси ливня (x, y)
print("\n" + "="*60)
print("Обучение модели для положения оси...")
start_time = time.time()

y_position_train = targets['position'].iloc[X_train.index]
y_position_test = targets['position'].iloc[X_test.index]

position_model = MultiOutputRegressor(RandomForestRegressor(**rf_params))
position_model.fit(X_train, y_position_train)
y_pred_position = position_model.predict(X_test)

# Оценка
mse_x = mean_squared_error(y_position_test['x'], y_pred_position[:, 0])
mse_y = mean_squared_error(y_position_test['y'], y_pred_position[:, 1])
r2_x = r2_score(y_position_test['x'], y_pred_position[:, 0])
r2_y = r2_score(y_position_test['y'], y_pred_position[:, 1])

print(f"Время обучения: {time.time()-start_time:.1f} сек")
print("Положение оси ливня:")
print(f"MSE координаты X: {mse_x:.4f}, R²: {r2_x:.4f}")
print(f"MSE координаты Y: {mse_y:.4f}, R²: {r2_y:.4f}")

# 9.3. Мощность ливня (power)
print("\n" + "="*60)
print("Обучение модели для мощности ливня...")
start_time = time.time()

y_power_train = targets['power'].iloc[X_train.index]
y_power_test = targets['power'].iloc[X_test.index]

power_model = RandomForestRegressor(**rf_params)
power_model.fit(X_train, y_power_train)
y_pred_power = power_model.predict(X_test)

# Оценка
mse_power = mean_squared_error(y_power_test, y_pred_power)
r2_power = r2_score(y_power_test, y_pred_power)

print(f"Время обучения: {time.time()-start_time:.1f} сек")
print("Мощность ливня:")
print(f"MSE: {mse_power:.4f}, R²: {r2_power:.4f}")

# 9.4. Возраст ливня (age) - используем Gradient Boosting
print("\n" + "="*60)
print("Обучение модели для возраста ливня...")
start_time = time.time()

y_age_train = targets['age'].iloc[X_train.index]
y_age_test = targets['age'].iloc[X_test.index]

age_model = GradientBoostingRegressor(**gb_params)
age_model.fit(X_train, y_age_train)
y_pred_age = age_model.predict(X_test)

# Оценка
mse_age = mean_squared_error(y_age_test, y_pred_age)
r2_age = r2_score(y_age_test, y_pred_age)

print(f"Время обучения: {time.time()-start_time:.1f} сек")
print("Возраст ливня:")
print(f"MSE: {mse_age:.4f}, R²: {r2_age:.4f}")

# 10. Визуализация результатов
plt.figure(figsize=(15, 12))

# Направление прихода
plt.subplot(2, 2, 1)
plt.scatter(y_direction_test['tetta'], y_pred_direction[:, 0], alpha=0.3)
plt.plot([0, 50], [0, 50], 'r--')
plt.xlabel('Истинный зенитный угол')
plt.ylabel('Предсказанный зенитный угол')
plt.title('Направление прихода: зенитный угол')

plt.subplot(2, 2, 2)
plt.scatter(true_phi, phi_pred, alpha=0.3)
plt.plot([0, 360], [0, 360], 'r--')
plt.xlabel('Истинный азимут')
plt.ylabel('Предсказанный азимут')
plt.title('Направление прихода: азимут')

# Положение оси
plt.subplot(2, 2, 3)
plt.scatter(y_position_test['x'], y_pred_position[:, 0], alpha=0.3)
plt.plot([-50, 50], [-50, 50], 'r--')
plt.xlabel('Истинная координата X')
plt.ylabel('Предсказанная координата X')
plt.title('Положение оси: координата X')

plt.subplot(2, 2, 4)
plt.scatter(y_position_test['y'], y_pred_position[:, 1], alpha=0.3)
plt.plot([-100, 100], [-100, 100], 'r--')
plt.xlabel('Истинная координата Y')
plt.ylabel('Предсказанная координата Y')
plt.title('Положение оси: координата Y')

plt.tight_layout()
plt.savefig('direction_position.png')
plt.show()

# Мощность ливня
plt.figure(figsize=(8, 6))
plt.scatter(y_power_test, y_pred_power, alpha=0.3)
plt.plot([3, 6], [3, 6], 'r--')
plt.xlabel('Истинная мощность')
plt.ylabel('Предсказанная мощность')
plt.title('Мощность ливня')
plt.savefig('power.png')
plt.show()

# Возраст ливня
plt.figure(figsize=(8, 6))
plt.scatter(y_age_test, y_pred_age, alpha=0.3)
plt.plot([0.5, 2.0], [0.5, 2.0], 'r--')
plt.xlabel('Истинный возраст')
plt.ylabel('Предсказанный возраст')
plt.title('Возраст ливня')
plt.savefig('age.png')
plt.show()

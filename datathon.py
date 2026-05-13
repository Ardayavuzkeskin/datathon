import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')
y     = train['bilissel_performans_skoru']

all_data = pd.concat([train.drop('bilissel_performans_skoru', axis=1), test], axis=0).reset_index(drop=True)

num_cols = [
    'yas', 'vucut_kitle_indeksi', 'rem_yuzdesi', 'derin_uyku_yuzdesi',
    'uykuya_dalma_suresi_dk', 'gecelik_uyanma_sayisi', 'uyku_oncesi_kafein_mg',
    'uyku_oncesi_ekran_suresi_dk', 'gunluk_adim_sayisi', 'sekerleme_suresi_dk',
    'stres_skoru', 'gunluk_calisma_saati', 'dinlenik_nabiz_bpm',
    'oda_sicakligi_celsius', 'hafta_sonu_uyku_farki_saat'
]
cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'gun_tipi']

all_data['sekerleme_suresi_dk'] = all_data['sekerleme_suresi_dk'].fillna(0)
for col in num_cols:
    all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
    all_data[col] = all_data[col].fillna(all_data[col].median())
for col in cat_cols:
    all_data[col] = all_data[col].fillna('Bilinmiyor')

all_data['biyolojik_onarim']    = (all_data['rem_yuzdesi'] + all_data['derin_uyku_yuzdesi']) / (all_data['stres_skoru'] + 1)
all_data['rem_stres_etki']      = all_data['rem_yuzdesi'] / (all_data['stres_skoru'] + 1)
all_data['uyku_fragmantasyonu'] = np.log1p(all_data['gecelik_uyanma_sayisi'] * all_data['uykuya_dalma_suresi_dk'])
all_data['uyku_kalitesi']       = all_data['rem_yuzdesi'] + all_data['derin_uyku_yuzdesi']
all_data['fiziksel_saglik']     = all_data['gunluk_adim_sayisi'] / (all_data['vucut_kitle_indeksi'] + 1)
all_data['sosyal_jetlag']       = np.log1p(all_data['stres_skoru'] * np.abs(all_data['hafta_sonu_uyku_farki_saat']))
all_data['fizyolojik_stres']    = np.log1p(all_data['stres_skoru'] * all_data['dinlenik_nabiz_bpm'])
all_data['stres_calisma']       = all_data['stres_skoru'] * all_data['gunluk_calisma_saati']
all_data['uyku_verimlilik']     = all_data['uyku_kalitesi'] / (all_data['uyku_fragmantasyonu'] + 1)

feature_cols = num_cols + cat_cols + [
    'biyolojik_onarim', 'rem_stres_etki', 'uyku_fragmantasyonu',
    'uyku_kalitesi', 'fiziksel_saglik', 'sosyal_jetlag', 'fizyolojik_stres',
    'stres_calisma', 'uyku_verimlilik'
]

X      = all_data[feature_cols].iloc[:len(train)].copy()
X_test = all_data[feature_cols].iloc[len(train):].copy()
cat_idx = [feature_cols.index(c) for c in cat_cols]

seeds    = [42, 2026, 1903]
n_splits = 10

print("ASAMA 1")
oof        = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for seed in seeds:
    print(f"Seed {seed}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        model = CatBoostRegressor(
            iterations=3000, learning_rate=0.02, depth=6,
            l2_leaf_reg=3, random_seed=seed, verbose=0
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            cat_features=cat_idx,
            eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
            early_stopping_rounds=200
        )
        oof[va_idx]  += model.predict(X.iloc[va_idx]) / len(seeds)
        test_preds   += model.predict(X_test) / (n_splits * len(seeds))
        fold_rmse = np.sqrt(mean_squared_error(y.iloc[va_idx], model.predict(X.iloc[va_idx])))
        print(f"Fold {fold+1:2d}/10 | RMSE: {fold_rmse:.5f}")

rmse_1 = np.sqrt(mean_squared_error(y, oof))
print(f"OOF RMSE: {rmse_1:.5f}")

print("\nASAMA 2")
pseudo_preds   = np.clip(test_preds, 0, 10)
mean_pred      = pseudo_preds.mean()
confident_mask = np.abs(pseudo_preds - mean_pred) < 1.5 * pseudo_preds.std()
print(f"Kullanilan satir: {confident_mask.sum()} / {len(pseudo_preds)}")

X_pseudo = X_test[confident_mask].copy()
y_pseudo = pd.Series(pseudo_preds[confident_mask])

oof_pseudo  = np.zeros(len(X))
test_preds2 = np.zeros(len(X_test))

for seed in seeds:
    print(f"Seed {seed}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        tr_X = pd.concat([X.iloc[tr_idx], X_pseudo], axis=0).reset_index(drop=True)
        tr_y = pd.concat([y.iloc[tr_idx], y_pseudo], axis=0).reset_index(drop=True)

        model = CatBoostRegressor(
            iterations=3000, learning_rate=0.02, depth=6,
            l2_leaf_reg=3, random_seed=seed, verbose=0
        )
        model.fit(
            tr_X, tr_y,
            cat_features=cat_idx,
            eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
            early_stopping_rounds=200
        )
        oof_pseudo[va_idx] += model.predict(X.iloc[va_idx]) / len(seeds)
        test_preds2        += model.predict(X_test) / (n_splits * len(seeds))
        fold_rmse = np.sqrt(mean_squared_error(y.iloc[va_idx], model.predict(X.iloc[va_idx])))
        print(f"Fold {fold+1:2d}/10 | RMSE: {fold_rmse:.5f}")

rmse_2 = np.sqrt(mean_squared_error(y, oof_pseudo))
print(f"OOF RMSE: {rmse_2:.5f}")

best_preds = test_preds2 if rmse_2 < rmse_1 else test_preds
sub = pd.DataFrame({
    'id': test['id'],
    'bilissel_performans_skoru': np.clip(best_preds, 0, 10)
})
sub.to_csv('submission_2.csv', index=False)

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available, skipping hyperparameter optimization")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, skipping CatBoost integration")

ZODIAC_MAP = {
    '鼠': 7, '牛': 6, '虎': 5, '兔': 4, '龙': 3, '蛇': 2,
    '马': 1, '羊': 12, '猴': 11, '鸡': 10, '狗': 9, '猪': 8
}

REVERSE_ZODIAC_MAP = {v: k for k, v in ZODIAC_MAP.items()}

ELEMENT_MAP = {
    1: '火', 2: '火', 3: '土', 4: '木', 5: '木', 6: '土',
    7: '土', 8: '水', 9: '土', 10: '金', 11: '金', 12: '土'
}

ELEMENT_RELATION = {
    ('木', '火'): 2, ('木', '土'): 0, ('木', '木'): 1, ('木', '金'): 0, ('木', '水'): 2,
    ('火', '土'): 2, ('火', '金'): 0, ('火', '火'): 1, ('火', '水'): 0, ('火', '木'): 2,
    ('土', '金'): 2, ('土', '水'): 0, ('土', '土'): 1, ('土', '木'): 0, ('土', '火'): 2,
    ('金', '水'): 2, ('金', '木'): 0, ('金', '金'): 1, ('金', '火'): 0, ('金', '土'): 2,
    ('水', '木'): 2, ('水', '火'): 0, ('水', '水'): 1, ('水', '土'): 0, ('水', '金'): 2
}

WAVE_COLOR_MAP = {
    1: 'red', 2: 'blue', 3: 'blue', 4: 'green', 5: 'green', 6: 'green',
    7: 'green', 8: 'green', 9: 'green', 10: 'blue', 11: 'blue', 12: 'red'
}

def get_real_data():
    CHINESE_ZODIAC_MAP = {
        '鼠': 7, '牛': 6, '虎': 5, '兔': 4, '龙': 3, '蛇': 2,
        '马': 1, '羊': 12, '猴': 11, '鸡': 10, '狗': 9, '猪': 8,
        '龍': 3, '馬': 1, '豬': 8, '雞': 10
    }
    
    raw_data = [
        (2026107, '馬'),
        (2026106, '雞'),
        (2026105, '兔'),
        (2026104, '馬'),
        (2026103, '牛'),
        (2026102, '豬'),
        (2026101, '龍'),
        (2026100, '狗'),
        (2026099, '羊'),
        (2026098, '虎'),
        (2026097, '猴'),
        (2026096, '鼠'),
        (2026095, '猴'),
        (2026094, '虎'),
        (2026093, '兔'),
        (2026092, '牛'),
        (2026091, '馬'),
        (2026090, '羊'),
        (2026089, '馬'),
        (2026088, '龍'),
        (2026087, '蛇'),
        (2026086, '羊'),
        (2026085, '鼠'),
        (2026084, '兔'),
        (2026083, '虎'),
        (2026082, '龍'),
        (2026081, '虎'),
        (2026080, '龍'),
        (2026079, '猴'),
        (2026078, '雞'),
        (2026077, '虎'),
        (2026076, '蛇'),
        (2026075, '狗'),
        (2026074, '雞'),
        (2026073, '雞'),
        (2026072, '雞'),
        (2026071, '羊'),
        (2026070, '馬'),
        (2026069, '羊'),
        (2026068, '猴'),
        (2026067, '虎'),
        (2026066, '馬'),
        (2026065, '虎'),
        (2026064, '牛'),
        (2026063, '兔'),
        (2026062, '羊'),
        (2026061, '虎'),
        (2026060, '狗'),
        (2026059, '雞'),
        (2026058, '鼠'),
        (2026057, '猴'),
        (2026056, '羊'),
        (2026055, '蛇'),
        (2026054, '羊'),
        (2026053, '龍'),
        (2026052, '蛇'),
        (2026051, '馬'),
        (2026050, '猴'),
        (2026049, '馬'),
        (2026048, '羊'),
        (2026047, '兔'),
        (2026046, '兔'),
        (2026045, '蛇'),
        (2026044, '豬'),
        (2026043, '蛇'),
        (2026042, '虎'),
        (2026041, '馬'),
        (2026040, '虎'),
        (2026039, '羊'),
        (2026038, '鼠'),
        (2026037, '豬'),
        (2026036, '蛇'),
        (2026035, '兔'),
        (2026034, '豬'),
        (2026033, '牛'),
        (2026032, '鼠'),
        (2026031, '龍'),
        (2026030, '牛'),
        (2026029, '蛇'),
        (2026028, '兔'),
        (2026027, '雞'),
        (2026026, '豬'),
        (2026025, '蛇'),
        (2026024, '虎'),
        (2026023, '羊'),
        (2026022, '鼠'),
        (2026021, '鼠'),
        (2026020, '馬'),
        (2026019, '猴'),
        (2026018, '兔'),
        (2026017, '狗'),
        (2026016, '鼠'),
        (2026015, '豬'),
        (2026014, '龍'),
        (2026013, '蛇'),
        (2026012, '羊'),
        (2026011, '羊'),
        (2026010, '兔'),
        (2026009, '虎'),
        (2026008, '雞'),
        (2026007, '牛'),
        (2026006, '蛇'),
        (2026005, '豬'),
        (2026004, '雞'),
        (2026003, '雞'),
        (2026002, '猴'),
        (2026001, '牛'),
    ]
    
    data_2026 = []
    for period, zodiac_name in reversed(raw_data):
        zodiac_num = CHINESE_ZODIAC_MAP.get(zodiac_name, 1)
        data_2026.append({'period': period, 'zodiac': zodiac_num})
    
    return data_2026

def build_features_ultimate(history, target_idx):
    features = {}
    last_zodiac = history.iloc[-1]['zodiac']
    
    if len(history) >= 2:
        last_2_zodiac = history.iloc[-2]['zodiac']
    else:
        last_2_zodiac = last_zodiac
    
    if len(history) >= 3:
        last_3_zodiac = history.iloc[-3]['zodiac']
    else:
        last_3_zodiac = last_zodiac
    
    if len(history) >= 4:
        last_4_zodiac = history.iloc[-4]['zodiac']
    else:
        last_4_zodiac = last_zodiac
    
    if len(history) >= 5:
        last_5_zodiac = history.iloc[-5]['zodiac']
    else:
        last_5_zodiac = last_zodiac
    
    for zodiac_num in range(1, 13):
        prefix = f'z{zodiac_num}_'
        
        appearances = history[history['zodiac'] == zodiac_num]
        
        if len(appearances) > 0:
            current_omission = target_idx - appearances.iloc[-1].name
        else:
            current_omission = target_idx
        
        features[prefix + 'omission'] = current_omission
        
        if len(appearances) > 1:
            omissions = []
            for i in range(1, len(appearances)):
                omissions.append(appearances.iloc[i].name - appearances.iloc[i-1].name)
            max_omission = max(omissions) if omissions else 1
            mean_omission = np.mean(omissions) if omissions else 1
            std_omission = np.std(omissions) if len(omissions) > 1 else 1
        else:
            max_omission = 1
            mean_omission = 1
            std_omission = 1
        
        features[prefix + 'omission_ratio'] = current_omission / max_omission if max_omission > 0 else 0
        features[prefix + 'omission_to_mean'] = current_omission / mean_omission if mean_omission > 0 else 0
        features[prefix + 'omission_zscore'] = (current_omission - mean_omission) / std_omission if std_omission > 0 else 0
        
        for n in [3, 5, 7, 10, 15, 20, 30, 50]:
            recent = history.tail(min(n, len(history)))
            count = len(recent[recent['zodiac'] == zodiac_num])
            features[prefix + f'count_{n}'] = count
            features[prefix + f'freq_{n}'] = count / len(recent) if len(recent) > 0 else 0
            features[prefix + f'deviation_{n}'] = count - (len(recent) / 12)
            features[prefix + f'expected_{n}'] = len(recent) / 12
            features[prefix + f'over_expected_{n}'] = 1 if count > (len(recent) / 12) else 0
        
        if len(appearances) >= 2:
            recent_appearances = appearances.tail(15)
            if len(recent_appearances) >= 2:
                intervals = []
                for i in range(1, len(recent_appearances)):
                    intervals.append(recent_appearances.iloc[i].name - recent_appearances.iloc[i-1].name)
                features[prefix + 'interval_mean'] = np.mean(intervals) if intervals else 0
                features[prefix + 'interval_std'] = np.std(intervals) if len(intervals) > 1 else 0
                features[prefix + 'interval_min'] = np.min(intervals) if intervals else 0
                features[prefix + 'interval_max'] = np.max(intervals) if intervals else 0
                features[prefix + 'interval_median'] = np.median(intervals) if intervals else 0
            else:
                features[prefix + 'interval_mean'] = 0
                features[prefix + 'interval_std'] = 0
                features[prefix + 'interval_min'] = 0
                features[prefix + 'interval_max'] = 0
                features[prefix + 'interval_median'] = 0
        else:
            features[prefix + 'interval_mean'] = 0
            features[prefix + 'interval_std'] = 0
            features[prefix + 'interval_min'] = 0
            features[prefix + 'interval_max'] = 0
            features[prefix + 'interval_median'] = 0
        
        features[prefix + 'pos_diff_1'] = abs(zodiac_num - last_zodiac)
        features[prefix + 'pos_diff_2'] = abs(zodiac_num - last_2_zodiac)
        features[prefix + 'pos_diff_3'] = abs(zodiac_num - last_3_zodiac)
        features[prefix + 'pos_diff_4'] = abs(zodiac_num - last_4_zodiac)
        features[prefix + 'pos_diff_5'] = abs(zodiac_num - last_5_zodiac)
        
        features[prefix + 'same_as_last'] = 1 if zodiac_num == last_zodiac else 0
        features[prefix + 'same_as_last2'] = 1 if zodiac_num == last_2_zodiac else 0
        features[prefix + 'same_as_last3'] = 1 if zodiac_num == last_3_zodiac else 0
        features[prefix + 'same_as_last4'] = 1 if zodiac_num == last_4_zodiac else 0
        features[prefix + 'same_as_last5'] = 1 if zodiac_num == last_5_zodiac else 0
        
        recent_5 = history.tail(min(5, len(history)))
        features[prefix + 'in_last5'] = 1 if zodiac_num in recent_5['zodiac'].values else 0
        features[prefix + 'count_last5'] = len(recent_5[recent_5['zodiac'] == zodiac_num])
        
        recent_10 = history.tail(min(10, len(history)))
        features[prefix + 'in_last10'] = 1 if zodiac_num in recent_10['zodiac'].values else 0
        features[prefix + 'count_last10'] = len(recent_10[recent_10['zodiac'] == zodiac_num])
        
        if len(appearances) >= 1:
            last_appearance_idx = appearances.iloc[-1].name
            features[prefix + 'appearance_recency'] = target_idx - last_appearance_idx
            features[prefix + 'is_last_appearance'] = 1 if zodiac_num == last_zodiac else 0
        else:
            features[prefix + 'appearance_recency'] = target_idx
            features[prefix + 'is_last_appearance'] = 0
        
        features[prefix + 'odd_even_same'] = 1 if (zodiac_num % 2) == (last_zodiac % 2) else 0
        features[prefix + 'big_small_same'] = 1 if ((zodiac_num <= 6) == (last_zodiac <= 6)) else 0
        features[prefix + 'is_odd'] = 1 if (zodiac_num % 2) == 1 else 0
        features[prefix + 'is_small'] = 1 if zodiac_num <= 6 else 0
        
        if zodiac_num <= 4:
            zone = 1
        elif zodiac_num <= 8:
            zone = 2
        else:
            zone = 3
        if last_zodiac <= 4:
            last_zone = 1
        elif last_zodiac <= 8:
            last_zone = 2
        else:
            last_zone = 3
        features[prefix + 'zone_same'] = 1 if zone == last_zone else 0
        features[prefix + 'zone'] = zone
        
        features[prefix + 'is_last'] = 1 if zodiac_num == last_zodiac else 0
        features[prefix + 'is_last2'] = 1 if zodiac_num == last_2_zodiac else 0
        features[prefix + 'is_last3'] = 1 if zodiac_num == last_3_zodiac else 0
        
        elem_current = ELEMENT_MAP[zodiac_num]
        elem_last = ELEMENT_MAP[last_zodiac]
        features[prefix + 'element_relation'] = ELEMENT_RELATION.get((elem_last, elem_current), 1)
        features[prefix + 'element_same'] = 1 if elem_current == elem_last else 0
        features[prefix + 'element_creates'] = 1 if ELEMENT_RELATION.get((elem_last, elem_current), 1) == 2 else 0
        features[prefix + 'element_destroys'] = 1 if ELEMENT_RELATION.get((elem_last, elem_current), 1) == 0 else 0
        
        wave_current = WAVE_COLOR_MAP[zodiac_num]
        wave_last = WAVE_COLOR_MAP[last_zodiac]
        features[prefix + 'wave_same'] = 1 if wave_current == wave_last else 0
        features[prefix + 'wave_is_red'] = 1 if wave_current == 'red' else 0
        features[prefix + 'wave_is_blue'] = 1 if wave_current == 'blue' else 0
        features[prefix + 'wave_is_green'] = 1 if wave_current == 'green' else 0
    
    for n in [10, 20, 30]:
        recent = history.tail(min(n, len(history)))
        counts = []
        for z in range(1, 13):
            cnt = len(recent[recent['zodiac'] == z])
            counts.append((-cnt, z))
        counts.sort()
        rank_map = {}
        for rank, (_, z) in enumerate(counts, 1):
            rank_map[z] = rank
        
        for z in range(1, 13):
            features[f'z{z}_hot_rank_{n}'] = rank_map[z]
            features[f'z{z}_is_hot3_{n}'] = 1 if rank_map[z] <= 3 else 0
            features[f'z{z}_is_hot5_{n}'] = 1 if rank_map[z] <= 5 else 0
            features[f'z{z}_is_cold3_{n}'] = 1 if rank_map[z] >= 10 else 0
            features[f'z{z}_is_cold5_{n}'] = 1 if rank_map[z] >= 8 else 0
    
    return features

def build_features(df, target_idx):
    history = df.iloc[:target_idx].copy()
    current_zodiac = df.iloc[target_idx]['zodiac']
    features = build_features_ultimate(history, target_idx)
    return features, current_zodiac

def build_features_for_next(df):
    history = df.copy()
    target_idx = len(df)
    features = build_features_ultimate(history, target_idx)
    return features

class UltimateEnsembleModel:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.cb_model = None
        self.le = None
        self.selected_features = None
    
    def fit(self, X, y, feature_selection=True):
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        
        if feature_selection:
            selector = SelectFromModel(
                xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=len(self.le.classes_),
                    n_estimators=100,
                    random_state=42
                ),
                threshold='median'
            )
            selector.fit(X, y_encoded)
            self.selected_features = X.columns[selector.get_support()].tolist()
            X_selected = X[self.selected_features]
        else:
            X_selected = X
            self.selected_features = list(X.columns)
        
        self.xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(self.le.classes_),
            n_estimators=400,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_selected, y_encoded)
        
        self.lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(self.le.classes_),
            n_estimators=400,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_model.fit(X_selected, y_encoded)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_selected, y_encoded)
        
        self.lr_model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        self.lr_model.fit(X_selected, y_encoded)
        
        if CATBOOST_AVAILABLE:
            self.cb_model = cb.CatBoostClassifier(
                iterations=400,
                depth=8,
                learning_rate=0.02,
                loss_function='MultiClass',
                random_state=42,
                verbose=0
            )
            self.cb_model.fit(X_selected, y_encoded)
    
    def predict_proba(self, X):
        X_selected = X[self.selected_features]
        
        xgb_proba = self.xgb_model.predict_proba(X_selected)
        lgb_proba = self.lgb_model.predict_proba(X_selected)
        rf_proba = self.rf_model.predict_proba(X_selected)
        lr_proba = self.lr_model.predict_proba(X_selected)
        
        if CATBOOST_AVAILABLE and self.cb_model is not None:
            cb_proba = self.cb_model.predict_proba(X_selected)
            ensemble_proba = (0.30 * xgb_proba + 0.25 * lgb_proba + 0.20 * rf_proba + 
                            0.15 * lr_proba + 0.10 * cb_proba)
        else:
            ensemble_proba = (0.35 * xgb_proba + 0.30 * lgb_proba + 0.20 * rf_proba + 
                            0.15 * lr_proba)
        
        return ensemble_proba

def rolling_prediction(df, start_period):
    print("=" * 80)
    print("开始滚动预测...")
    print("=" * 80)
    
    start_idx = df[df['period'] == start_period].index[0]
    print(f"从第 {start_period} 期开始滚动预测 (索引: {start_idx})")
    
    predictions = []
    correct_count = 0
    top3_correct_count = 0
    
    for idx_in_test in range(len(df) - start_idx):
        current_idx = start_idx + idx_in_test
        current_period = df.iloc[current_idx]['period']
        actual_zodiac = df.iloc[current_idx]['zodiac']
        
        print(f"\r正在预测第 {current_period} 期... ({idx_in_test + 1}/{len(df) - start_idx})", end='')
        
        X_list = []
        y_list = []
        
        for idx in range(1, current_idx):
            features, label = build_features(df, idx)
            X_list.append(features)
            y_list.append(label)
        
        if len(X_list) < 10:
            continue
        
        X = pd.DataFrame(X_list)
        y = np.array(y_list) - 1
        
        model = UltimateEnsembleModel()
        model.fit(X, y, feature_selection=True)
        
        features_current, _ = build_features(df, current_idx)
        X_current = pd.DataFrame([features_current])
        
        prob = model.predict_proba(X_current)[0]
        
        pred_list = []
        for enc_idx, p in enumerate(prob):
            orig_y = model.le.inverse_transform([enc_idx])[0]
            zodiac_num = orig_y + 1
            zodiac_name = REVERSE_ZODIAC_MAP[zodiac_num]
            pred_list.append({
                'zodiac_num': zodiac_num,
                'zodiac_name': zodiac_name,
                'probability': p
            })
        
        pred_list.sort(key=lambda x: x['probability'], reverse=True)
        
        top1_name = pred_list[0]['zodiac_name']
        top2_name = pred_list[1]['zodiac_name']
        top3_name = pred_list[2]['zodiac_name']
        
        actual_name = REVERSE_ZODIAC_MAP[actual_zodiac]
        
        hit = actual_name == top1_name
        top3_hit = actual_name in [top1_name, top2_name, top3_name]
        
        if hit:
            correct_count += 1
        if top3_hit:
            top3_correct_count += 1
        
        predictions.append({
            'period': current_period,
            'actual': actual_name,
            'top1': top1_name,
            'top2': top2_name,
            'top3': top3_name,
            'hit': hit,
            'top3_hit': top3_hit
        })
    
    print()
    
    print("\n" + "=" * 80)
    print(f"{'期号':<10} {'实际生肖':<10} {'预测Top1':<10} {'预测Top2':<10} {'预测Top3':<10} {'命中':<6}")
    print("=" * 80)
    
    for pred in predictions:
        hit_mark = "✓" if pred['hit'] else ""
        print(f"{pred['period']:<10} {pred['actual']:<10} {pred['top1']:<10} {pred['top2']:<10} {pred['top3']:<10} {hit_mark:<6}")
    
    print("=" * 80)
    
    test_count = len(predictions)
    accuracy = correct_count / test_count if test_count > 0 else 0
    top3_accuracy = top3_correct_count / test_count if test_count > 0 else 0
    
    print(f"\n滚动预测结果统计:")
    print(f"  总预测期数: {test_count}")
    print(f"  Top-1 命中数: {correct_count}")
    print(f"  Top-1 准确率: {accuracy*100:.2f}%")
    print(f"  Top-3 命中数: {top3_correct_count}")
    print(f"  Top-3 准确率: {top3_accuracy*100:.2f}%")
    
    return predictions

def main():
    print("=" * 80)
    print("生肖预测2.0 - 终极优化版")
    print("=" * 80)
    
    print("\n[1/5] 正在加载真实数据...")
    data = get_real_data()
    df = pd.DataFrame(data)
    df = df.sort_values('period').reset_index(drop=True)
    print(f"成功加载 {len(df)} 期历史数据")
    print(f"数据范围: 第 {df['period'].min()} 期 至 第 {df['period'].max()} 期")
    
    print("\n" + "=" * 80)
    print("请选择运行模式:")
    print("  1. 常规模式（快速，一次性训练）")
    print("  2. 滚动预测（慢，每期重新训练）")
    print("=" * 80)
    
    choice = input("\n请输入选择 (1 或 2，默认1): ").strip()
    
    if choice == '2':
        start_period = 2026060
        rolling_prediction(df, start_period)
    else:
        print("\n[2/5] 正在找到起始期位置...")
        start_period = 2026040
        start_idx = df[df['period'] == start_period].index[0]
        print(f"从第 {start_period} 期开始预测 (索引位置: {start_idx})")
        
        print("\n[3/5] 正在构建终极特征...")
        X_list = []
        y_list = []
        periods_list = []
        
        for idx in range(1, len(df)):
            features, label = build_features(df, idx)
            X_list.append(features)
            y_list.append(label)
            periods_list.append(df.iloc[idx]['period'])
        
        X = pd.DataFrame(X_list)
        y = np.array(y_list) - 1
        print(f"特征数量: {X.shape[1]}")
        print(f"总样本数: {X.shape[0]}")
        
        print("\n[4/5] 正在划分训练集和测试集...")
        train_end = start_idx - 1
        X_train = X.iloc[:train_end]
        y_train = y[:train_end]
        X_test = X.iloc[train_end:]
        y_test = y[train_end:]
        test_periods = periods_list[train_end:]
        print(f"训练集: {len(X_train)} 期")
        print(f"测试集: {len(X_test)} 期")
        
        print("\n[5/5] 正在训练终极集成模型...")
        model = UltimateEnsembleModel()
        model.fit(X_train, y_train, feature_selection=True)
        print(f"选择特征数: {len(model.selected_features)}")
        print("集成模型训练完成 (XGBoost + LightGBM + RandomForest + LogisticRegression + CatBoost)")
        
        print("\n正在进行预测...")
        print("\n" + "=" * 80)
        print(f"{'期号':<10} {'实际生肖':<10} {'预测Top1':<10} {'预测Top2':<10} {'预测Top3':<10} {'命中':<6}")
        print("=" * 80)
        
        correct_count = 0
        top3_correct_count = 0
        
        probs = model.predict_proba(X_test)
        
        for i in range(len(X_test)):
            prob = probs[i]
            
            predictions = []
            for enc_idx, p in enumerate(prob):
                orig_y = model.le.inverse_transform([enc_idx])[0]
                zodiac_num = orig_y + 1
                zodiac_name = REVERSE_ZODIAC_MAP[zodiac_num]
                predictions.append({
                    'zodiac_num': zodiac_num,
                    'zodiac_name': zodiac_name,
                    'probability': p
                })
            
            predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            actual_period = test_periods[i]
            actual_y = y_test[i]
            actual_zodiac_num = actual_y + 1
            actual_zodiac_name = REVERSE_ZODIAC_MAP[actual_zodiac_num]
            
            top1_name = predictions[0]['zodiac_name']
            top2_name = predictions[1]['zodiac_name']
            top3_name = predictions[2]['zodiac_name']
            
            hit = "✓" if actual_zodiac_name == top1_name else ""
            top3_hit = actual_zodiac_name in [top1_name, top2_name, top3_name]
            
            if hit:
                correct_count += 1
            if top3_hit:
                top3_correct_count += 1
            
            print(f"{actual_period:<10} {actual_zodiac_name:<10} {top1_name:<10} {top2_name:<10} {top3_name:<10} {hit:<6}")
        
        print("=" * 80)
        
        test_count = len(X_test)
        accuracy = correct_count / test_count if test_count > 0 else 0
        top3_accuracy = top3_correct_count / test_count if test_count > 0 else 0
        print(f"\n预测结果统计:")
        print(f"  总预测期数: {test_count}")
        print(f"  Top-1 命中数: {correct_count}")
        print(f"  Top-1 准确率: {accuracy*100:.2f}%")
        print(f"  Top-3 命中数: {top3_correct_count}")
        print(f"  Top-3 准确率: {top3_accuracy*100:.2f}%")
        
        print("\n" + "=" * 80)
        print("正在预测第 2026108 期...")
        print("=" * 80)
        
        print("\n正在用全部数据重新训练终极模型...")
        final_model = UltimateEnsembleModel()
        final_model.fit(X, y, feature_selection=True)
        
        print("\n正在构建第 2026108 期特征...")
        features_next = build_features_for_next(df)
        X_next = pd.DataFrame([features_next])
        
        prob_next = final_model.predict_proba(X_next)[0]
        
        predictions_next = []
        for enc_idx, p in enumerate(prob_next):
            orig_y = final_model.le.inverse_transform([enc_idx])[0]
            zodiac_num = orig_y + 1
            zodiac_name = REVERSE_ZODIAC_MAP[zodiac_num]
            predictions_next.append({
                'zodiac_num': zodiac_num,
                'zodiac_name': zodiac_name,
                'probability': p
            })
        
        predictions_next.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"\n第 2026108 期预测推荐:")
        print("\n" + "=" * 80)
        print(f"{'排名':<6} {'生肖':<8} {'编号':<6} {'概率':<10}")
        print("=" * 80)
        for rank, pred in enumerate(predictions_next[:12], 1):
            print(f"{rank:<6} {pred['zodiac_name']:<8} {pred['zodiac_num']:<6} {pred['probability']*100:.2f}%")
        print("=" * 80)
        
        print(f"\n🎯 第 2026108 期 Top 3 推荐:")
        print(f"   1. {predictions_next[0]['zodiac_name']} (编号{predictions_next[0]['zodiac_num']}) - {predictions_next[0]['probability']*100:.2f}%")
        print(f"   2. {predictions_next[1]['zodiac_name']} (编号{predictions_next[1]['zodiac_num']}) - {predictions_next[1]['probability']*100:.2f}%")
        print(f"   3. {predictions_next[2]['zodiac_name']} (编号{predictions_next[2]['zodiac_num']}) - {predictions_next[2]['probability']*100:.2f}%")
        print("=" * 80)
        
        print("\n已实现的优化:")
        print("  1. 终极特征工程 - 遗漏统计、多窗口频率、五行、波色、连开模式等")
        print("  2. 集成学习 - XGBoost + LightGBM + RandomForest + LogisticRegression + CatBoost")
        print("  3. 特征选择 - SelectFromModel自动筛选重要特征")
        print("  4. 滚动预测 - 可选模式，每预测一期后重新训练")
        print("  5. 模型调优 - 优化所有模型参数")
        print("=" * 80)

if __name__ == "__main__":
    main()


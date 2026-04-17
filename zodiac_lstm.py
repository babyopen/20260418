
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, will skip LSTM model")

ZODIAC_MAP = {
    '鼠': 7, '牛': 6, '虎': 5, '兔': 4, '龙': 3, '蛇': 2,
    '马': 1, '羊': 12, '猴': 11, '鸡': 10, '狗': 9, '猪': 8
}

REVERSE_ZODIAC_MAP = {v: k for k, v in ZODIAC_MAP.items()}

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

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        Embedding(input_dim=13, output_dim=32, input_length=input_shape[0]),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("=" * 80)
    print("生肖预测2.0 - LSTM深度学习版")
    print("=" * 80)
    
    if not TF_AVAILABLE:
        print("\nTensorFlow not available. Please install it with:")
        print("  pip install tensorflow")
        print("\nUsing fallback method (markov chain)...")
        
        data = get_real_data()
        df = pd.DataFrame(data)
        df = df.sort_values('period').reset_index(drop=True)
        
        zodiac_sequence = df['zodiac'].values
        
        from collections import defaultdict
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(1, len(zodiac_sequence)):
            prev = zodiac_sequence[i-1]
            curr = zodiac_sequence[i]
            transition_counts[prev][curr] += 1
        
        transition_probs = {}
        for prev in transition_counts:
            total = sum(transition_counts[prev].values())
            transition_probs[prev] = {}
            for curr in transition_counts[prev]:
                transition_probs[prev][curr] = transition_counts[prev][curr] / total
        
        last_zodiac = zodiac_sequence[-1]
        
        predictions = []
        for z in range(1, 13):
            prob = transition_probs.get(last_zodiac, {}).get(z, 0.0)
            if prob == 0:
                prob = 1.0 / 12
            predictions.append({
                'zodiac_num': z,
                'zodiac_name': REVERSE_ZODIAC_MAP[z],
                'probability': prob
            })
        
        total_prob = sum(p['probability'] for p in predictions)
        for p in predictions:
            p['probability'] /= total_prob
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"\n第 2026108 期预测推荐 (Markov Chain):")
        print("\n" + "=" * 80)
        print(f"{'排名':<6} {'生肖':<8} {'编号':<6} {'概率':<10}")
        print("=" * 80)
        for rank, pred in enumerate(predictions[:12], 1):
            print(f"{rank:<6} {pred['zodiac_name']:<8} {pred['zodiac_num']:<6} {pred['probability']*100:.2f}%")
        print("=" * 80)
        
        print(f"\n🎯 第 2026108 期 Top 3 推荐:")
        print(f"   1. {predictions[0]['zodiac_name']} (编号{predictions[0]['zodiac_num']}) - {predictions[0]['probability']*100:.2f}%")
        print(f"   2. {predictions[1]['zodiac_name']} (编号{predictions[1]['zodiac_num']}) - {predictions[1]['probability']*100:.2f}%")
        print(f"   3. {predictions[2]['zodiac_name']} (编号{predictions[2]['zodiac_num']}) - {predictions[2]['probability']*100:.2f}%")
        print("=" * 80)
        return
    
    print("\n[1/6] 正在加载真实数据...")
    data = get_real_data()
    df = pd.DataFrame(data)
    df = df.sort_values('period').reset_index(drop=True)
    print(f"成功加载 {len(df)} 期历史数据")
    print(f"数据范围: 第 {df['period'].min()} 期 至 第 {df['period'].max()} 期")
    
    print("\n[2/6] 正在找到起始期位置...")
    start_period = 2026040
    start_idx = df[df['period'] == start_period].index[0]
    print(f"从第 {start_period} 期开始预测 (索引位置: {start_idx})")
    
    print("\n[3/6] 正在准备序列数据...")
    seq_length = 10
    zodiac_sequence = df['zodiac'].values
    
    X_train_full, y_train_full = create_sequences(zodiac_sequence[:start_idx], seq_length)
    X_test, y_test = create_sequences(zodiac_sequence[start_idx - seq_length:], seq_length)
    
    test_periods = df['period'].values[start_idx:start_idx + len(y_test)]
    
    print(f"序列长度: {seq_length}")
    print(f"训练集序列数: {len(X_train_full)}")
    print(f"测试集序列数: {len(X_test)}")
    
    y_train_onehot = to_categorical(y_train_full - 1, num_classes=12)
    y_test_onehot = to_categorical(y_test - 1, num_classes=12)
    
    print("\n[4/6] 正在构建LSTM模型...")
    model = build_lstm_model((seq_length,), 12)
    model.summary()
    
    print("\n[5/6] 正在训练LSTM模型...")
    history = model.fit(
        X_train_full, y_train_onehot,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    
    print("\n[6/6] 正在进行预测...")
    print("\n" + "=" * 80)
    print(f"{'期号':<10} {'实际生肖':<10} {'预测Top1':<10} {'预测Top2':<10} {'预测Top3':<10} {'命中':<6}")
    print("=" * 80)
    
    correct_count = 0
    top3_correct_count = 0
    
    probs = model.predict(X_test, verbose=0)
    
    for i in range(len(X_test)):
        prob = probs[i]
        
        predictions = []
        for enc_idx, p in enumerate(prob):
            zodiac_num = enc_idx + 1
            zodiac_name = REVERSE_ZODIAC_MAP[zodiac_num]
            predictions.append({
                'zodiac_num': zodiac_num,
                'zodiac_name': zodiac_name,
                'probability': p
            })
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        actual_period = test_periods[i]
        actual_zodiac_num = y_test[i]
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
    
    print("\n正在用全部数据重新训练LSTM模型...")
    X_full, y_full = create_sequences(zodiac_sequence, seq_length)
    y_full_onehot = to_categorical(y_full - 1, num_classes=12)
    
    final_model = build_lstm_model((seq_length,), 12)
    final_model.fit(
        X_full, y_full_onehot,
        epochs=50,
        batch_size=8,
        verbose=1
    )
    
    print("\n正在构建第 2026108 期序列...")
    last_sequence = zodiac_sequence[-seq_length:]
    X_next = np.array([last_sequence])
    
    prob_next = final_model.predict(X_next, verbose=0)[0]
    
    predictions_next = []
    for enc_idx, p in enumerate(prob_next):
        zodiac_num = enc_idx + 1
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
    
    print("\nLSTM架构:")
    print("  - Embedding层: 13维输入 → 32维嵌入")
    print("  - LSTM层1: 64个单元 + Dropout 0.3")
    print("  - LSTM层2: 32个单元 + Dropout 0.3")
    print("  - 全连接层: 64个单元 (ReLU激活)")
    print("  - 输出层: 12个单元 (Softmax激活)")
    print("  - 优化器: Adam (学习率 0.001)")
    print("  - 损失函数: 分类交叉熵")
    print("=" * 80)

if __name__ == "__main__":
    main()


# %% [1] 경로 설정 및 데이터 로드 (미리보기 포함)
import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.stats import norm

# 경로 설정ㅇㅇㅇ
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'notebooks':
    base_path = os.path.join(os.path.dirname(current_dir), 'data')
else:
    base_path = os.path.join(current_dir, 'data')

train_file = os.path.join(base_path, 'train.csv')
metadata_file = os.path.join(base_path, 'item_metadata.csv')

# 1. 데이터 로드 전 상위 5행 맛보기 (구조 확인)
print("--- Train 데이터 상위 5행 미리보기 ---")
train_preview = pd.read_csv(train_file, nrows=5)
print(train_preview)

# 2. 전체 데이터 로드 (메모리 최적화 타입 지정)
t_dtypes = {
    'user_id': 'str', 'session_id': 'str', 'action_type': 'str',
    'reference': 'str', 'timestamp': 'int64', 'impressions': 'str', 'prices': 'str'
}

print("\n전체 데이터 로딩 시작... (시간이 소요될 수 있습니다)")
train = pd.read_csv(train_file, dtype=t_dtypes)
print(f"✅ 전체 데이터 로드 완료: {len(train):,} 행")

# %% [2] 아이템 관련 액션 필터링
# reference가 숫자인 아이템 관련 액션만 추출
item_actions = train[train['reference'].str.isnumeric() == True].copy()
item_actions['reference'] = item_actions['reference'].astype(str)
print(f"아이템 액션 필터링 완료: {len(item_actions):,} 행")

# %% [3] 세션 간 재방문 분석 함수 및 실행
def analyze_user_revisits(group):
    group = group.sort_values('timestamp')
    sessions = group['session_id'].unique()
    total_revisits = 0
    
    if len(sessions) > 1:
        session_item_sets = [set(group[group['session_id'] == s]['reference']) for s in sessions]
        accumulated_previous_items = set()

        for i in range(1, len(sessions)):
            accumulated_previous_items.update(session_item_sets[i-1])
            current_session_items = session_item_sets[i]
            revisited_in_this_session = current_session_items.intersection(accumulated_previous_items)
            total_revisits += len(revisited_in_this_session)

    return pd.Series({
        'total_sessions': len(sessions),
        'cross_session_revisit_count': total_revisits,
        'is_multi_session_user': 1 if len(sessions) > 1 else 0
    })

print("유저별 세션 재방문 분석 중...")
user_stats = item_actions.groupby('user_id').apply(analyze_user_revisits, include_groups=False)

multi_session_users = user_stats[user_stats['is_multi_session_user'] == 1]
print("\n--- 유저 단위 재방문 분석 결과 ---")
print(f"2개 이상의 세션을 가진 유저 수: {len(multi_session_users):,}")
if len(multi_session_users) > 0:
    revisit_rate = (multi_session_users['cross_session_revisit_count'] > 0).mean() * 100
    print(f"재접속 후 이전 아이템을 다시 본 유저 비율: {revisit_rate:.2f}%")
# %% 
import gc
del train  # 원본 데이터가 더 이상 필요 없다면 삭제
gc.collect() # 가비지 컬렉션 강제 실행
# %% [4] Delta t (재방문 시간 간격) 계산
train_ready = item_actions.copy()
train_ready = train_ready.sort_values(['user_id', 'reference', 'timestamp'])

# 이전 방문 시점과의 차이 계산 (시간 단위)
train_ready['prev_timestamp'] = train_ready.groupby(['user_id', 'reference'])['timestamp'].shift(1)
train_ready['delta_t'] = (train_ready['timestamp'] - train_ready['prev_timestamp']) / 3600
print("Delta t 계산 완료.")

# %% [5] 가격 정보(Price) 추출 로직 적용
def extract_clicked_price(row):
    try:
        if pd.isna(row['impressions']) or pd.isna(row['prices']):
            return np.nan
        imps = str(row['impressions']).split('|')
        pris = str(row['prices']).split('|')
        if row['reference'] in imps:
            return float(pris[imps.index(row['reference'])])
    except:
        return np.nan
    return np.nan

print("가격 정보 매칭 중...")
train_ready['price'] = train_ready.apply(extract_clicked_price, axis=1)

# %% [6] 아이템 메타데이터 로드 및 속성 결합
item_metadata = pd.read_csv(metadata_file)

# 상위 20개 빈번 속성 추출
all_props = []
item_metadata['properties'].dropna().apply(lambda x: all_props.extend(x.split('|')))
top_20_props = [p for p, _ in Counter(all_props).most_common(20)]

# Multi-hot Encoding
for prop in top_20_props:
    col_name = f"prop_{prop.replace(' ', '_')}"
    item_metadata[col_name] = item_metadata['properties'].apply(
        lambda x: 1 if isinstance(x, str) and prop in x else 0
    )

# 병합을 위한 특징 데이터프레임 생성
prop_cols = [f"prop_{p.replace(' ', '_')}" for p in top_20_props]
item_features = item_metadata[['item_id'] + prop_cols].copy()
item_features['item_id'] = item_features['item_id'].astype(str)

# 기존 train_ready와 결합
if 'item_id' in train_ready.columns:
    train_ready = train_ready.drop(columns=['item_id'])
train_ready = train_ready.merge(item_features, left_on='reference', right_on='item_id', how='left')
print("✅ 아이템 속성 및 가격 정보 결합 완료!")

# %% [전처리가 끝난 후 저장]
# 필터링된 데이터만 따로 저장해두면, 다음엔 1,400만 행 전체를 읽을 필요가 없습니다.
train_ready.to_parquet(os.path.join(base_path, 'train_ready.parquet')) 
# Parquet 형식은 CSV보다 용량이 훨씬 작고 로딩 속도가 10배 이상 빠릅니다.

# %% [7] 최종 분석용 샘플링 (1,000명)
revisit_users_list = train_ready[train_ready['delta_t'].notnull()]['user_id'].unique()
sample_df = train_ready[train_ready['user_id'].isin(revisit_users_list[:1000])].copy()

print(f"최종 샘플 유저 수: {len(sample_df['user_id'].unique())}")
print(f"최종 샘플 행 수: {len(sample_df)}")
sample_df[['user_id', 'reference', 'delta_t', 'price'] + prop_cols[:3]].head()
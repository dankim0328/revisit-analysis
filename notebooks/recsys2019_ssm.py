# %% [1] 경로 설정 및 데이터 로드 (미리보기 포함)
import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.stats import norm

# 경로 설정ㅃㅂ
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
user_stats = panel.groupby('user_id').apply(analyze_user_revisits, include_groups=False)

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
train_ready = panel.copy()
train_ready = train_ready.sort_values(['user_id', 'reference', 'timestamp']).reset_index(drop=True)
print("Delta t / revisit / negative samples 포함 데이터 준비 완료.")

# %% [5] 가격 정보(Price) 추출 로직 적용
# (이미 panel 생성 시 price를 매칭했으므로 skip)
print("가격 정보: opportunity panel 생성 단계에서 매칭 완료.")

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
# NOTE: 전체 train_ready를 parquet으로 저장하는 작업은 용량/시간 부담이 커서 생략하고,
#       바로 SMLE용 sample_df만 저장합니다.

# %% [7] 최종 분석용 샘플링 (1,000명)
all_users = train_ready['user_id'].unique()
revisit_choice_users = train_ready.loc[train_ready['revisit_choice'] == 1, 'user_id'].unique()
other_users = np.setdiff1d(all_users, revisit_choice_users)

# Selection-bias 완화: revisit_choice 발생 유저/미발생 유저를 함께 포함 (가능하면 반반)
rng = np.random.default_rng(42)
n_total = 1000
n_pos = min(len(revisit_choice_users), n_total // 2)
n_neg = min(len(other_users), n_total - n_pos)

sample_users = []
if n_pos > 0:
    sample_users.append(rng.choice(revisit_choice_users, size=n_pos, replace=False))
if n_neg > 0:
    sample_users.append(rng.choice(other_users, size=n_neg, replace=False))
sample_users = np.concatenate(sample_users) if len(sample_users) > 0 else rng.choice(all_users, size=min(n_total, len(all_users)), replace=False)

sample_df = train_ready[train_ready['user_id'].isin(sample_users)].copy()

print(f"최종 샘플 유저 수: {len(sample_df['user_id'].unique())}")
print(f"최종 샘플 행 수: {len(sample_df)}")
sample_df[['user_id', 'reference', 'delta_t', 'price', 'chosen', 'revisit', 'revisit_choice'] + prop_cols[:3]].head()

# 샘플 저장 (Notebook에서 바로 로드 가능)
sample_path_parquet = os.path.join(base_path, 'smle_sample.parquet')
sample_path_csv = os.path.join(base_path, 'smle_sample.csv')
sample_df.to_parquet(sample_path_parquet, index=False)
sample_df.to_csv(sample_path_csv, index=False)
print(f"✅ SMLE 샘플 저장 완료: {sample_path_parquet}, {sample_path_csv}")
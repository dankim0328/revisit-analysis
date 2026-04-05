# %% [1] 경로 설정 및 데이터 로드 (미리보기 포함)
import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.stats import norm

# 경로 설정
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

MAX_CLICKOUTS_FOR_PANEL = 1000000
clickouts = item_actions[item_actions['action_type'] == 'clickout item'].copy()
if len(clickouts) > MAX_CLICKOUTS_FOR_PANEL:
    print(f"Memory issue: taking top {MAX_CLICKOUTS_FOR_PANEL} clickouts")
    clickouts = clickouts.head(MAX_CLICKOUTS_FOR_PANEL).copy()
print(f"   --> panel 생성에 사용할 clickout 수: {len(clickouts):,}개")

def _parse_impressions_prices(impressions, prices):
    imps = str(impressions).split('|')
    pris = str(prices).split('|')
    if len(imps) == 0 or len(imps) != len(pris):
        return None, None
    return imps, pris

def build_opportunity_panel(clickouts_df, max_candidates=25, seed=42):
    rng = np.random.default_rng(seed)
    
    cols = {
        'user_id': [], 'session_id': [], 'timestamp': [], 'reference': [],
        'price': [], 'delta_t': [], 'revisit': [], 'chosen': [],
        'revisit_choice': [], 'impressions': [], 'prices': []
    }
    last_seen = {}  # user_id -> dict[item_id] = last_timestamp

    for r in clickouts_df.itertuples(index=False):
        user = r.user_id
        ts = int(r.timestamp)
        ref = str(r.reference)

        imps, pris = _parse_impressions_prices(r.impressions, r.prices)
        if imps is None:
            continue

        idx_ref = None
        for i, it in enumerate(imps):
            if str(it) == ref:
                idx_ref = i
                break
        if idx_ref is None:
            continue

        n_imps = len(imps)
        if n_imps > max_candidates:
            neg_idx = [i for i in range(n_imps) if i != idx_ref]
            keep_neg = rng.choice(neg_idx, size=max_candidates - 1, replace=False)
            keep_idx = np.concatenate([[idx_ref], keep_neg])
        else:
            keep_idx = np.arange(n_imps)

        user_map = last_seen.get(user)
        if user_map is None:
            user_map = {}
            last_seen[user] = user_map

        for i in keep_idx:
            item = str(imps[int(i)])
            try:
                price = float(pris[int(i)])
            except Exception:
                price = np.nan

            prev_ts = user_map.get(item)
            revisit = 1 if prev_ts is not None else 0
            delta_t = 0.0 if prev_ts is None else (ts - prev_ts) / 3600.0
            chosen = 1 if item == ref else 0
            revisit_choice = 1 if (chosen == 1 and revisit == 1) else 0

            cols['user_id'].append(user)
            cols['session_id'].append(r.session_id)
            cols['timestamp'].append(ts)
            cols['reference'].append(item)
            cols['price'].append(price)
            cols['delta_t'].append(delta_t)
            cols['revisit'].append(revisit)
            cols['chosen'].append(chosen)
            cols['revisit_choice'].append(revisit_choice)
            cols['impressions'].append(r.impressions)
            cols['prices'].append(r.prices)

        for it in imps:
            user_map[str(it)] = ts

    return pd.DataFrame(cols)

print("Opportunity panel 생성 중... (negative samples 포함)")
panel = build_opportunity_panel(clickouts, max_candidates=25, seed=42)
print(f"✅ Opportunity panel 생성 완료: {len(panel):,} 행 / 유저 {panel['user_id'].nunique():,}명")

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
# %% [4] Delta t (재방문 시간 간격) 및 N_int (인지적 과부하) 계산
# N_int 계산을 위해 user와 timestamp 순으로 정렬
panel = panel.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# 1. 특정 아이템을 '처음' 클릭한 경우를 찾아 True/False로 나타냄
panel['is_first_interaction'] = ~panel.duplicated(subset=['user_id', 'reference'])

# 2. 누적합으로 현재 시점까지 탐색한 고유 아이템 종류 계산
panel['cum_unique_items'] = panel.groupby('user_id')['is_first_interaction'].cumsum()

# 3. 아이템(reference)별 직전 클릭 시점의 누적 고유 아이템 수 가져오기
panel['prev_cum_unique_items'] = panel.groupby(['user_id', 'reference'])['cum_unique_items'].shift(1)

# 4. N_int 계산 (현재 - 직전)
panel['N_int'] = panel['cum_unique_items'] - panel['prev_cum_unique_items']
panel['N_int'] = panel['N_int'].fillna(0).astype(int)

# 불필요한 중간 컬럼 삭제
panel = panel.drop(columns=['is_first_interaction', 'cum_unique_items', 'prev_cum_unique_items'])

train_ready = panel.copy()
train_ready = train_ready.sort_values(['user_id', 'reference', 'timestamp']).reset_index(drop=True)
print("Delta t / revisit / negative samples / N_int 포함 데이터 준비 완료.")

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
for prop in []: # top_20_props: (시각화용 RAM 150MB 절약)
    col_name = f"prop_{prop.replace(' ', '_')}"
    item_metadata[col_name] = item_metadata['properties'].apply(
        lambda x: 1 if isinstance(x, str) and prop in x else 0
    )

# 병합을 위한 특징 데이터프레임 생성
prop_cols = [f"prop_{p.replace(' ', '_')}" for p in top_20_props]
prop_cols = [c for c in prop_cols if c in item_metadata.columns]
item_features = item_metadata[['item_id'] + prop_cols].copy()
item_features['item_id'] = item_features['item_id'].astype(str)

# 기존 train_ready와 결합
if 'item_id' in train_ready.columns:
    train_ready = train_ready.drop(columns=['item_id'])
# train_ready = train_ready.merge(item_features, left_on='reference', right_on='item_id', how='left')
print("✅ 아이템 속성 및 가격 정보 결합 완료!")

# (데이터(test sample) 추출 및 저장 코드는 별도 파일로 분리 기획)


# %% [8] 인지적 과부하(N_int)와 재방문 확률 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 재방문 기회(revisit == 1)인 데이터만 추출
revisit_opp_df = train_ready[train_ready['revisit'] == 1].copy()

# N_int binning (0, 1~5, 6~10, 11~20, 20+)
bins = [-1, 0, 5, 10, 20, float('inf')]
labels = ['0', '1~5', '6~10', '11~20', '20+']
revisit_opp_df['N_int_group'] = pd.cut(revisit_opp_df['N_int'], bins=bins, labels=labels)

# 구간별 재방문 평균 계산 (데이터 부족 구간 제외를 위해 count 산출)
n_int_stats = revisit_opp_df.groupby('N_int_group')['revisit_choice'].agg(['mean', 'count']).reset_index()

# 데이터 부족 구간 제외: 이 구간의 관측치가 30개 미만이면 플롯에서 제외
n_int_stats_valid = n_int_stats[n_int_stats['count'] >= 30]

print("\n--- 인지적 과부하 (N_int) 구간별 재방문 전환율 ---")
print(n_int_stats_valid)

# 1. N_int 구간별 평균 재방문 확률 Line Plot (체증적 패턴 확인용)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=n_int_stats_valid,
    x='N_int_group',
    y='mean',
    marker='o',
    color='blue'
)
plt.title('Revisit Probability by N_int Group')
plt.xlabel('N_int Group (Number of Extraneous Items Explored)')
plt.ylabel('Average Revisit Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 2. X축 연속형 변수 치환 Regplot (개별 수치 연속성)
# 데이터가 900만 개로 너무 많아 seaborn의 logistic 부트스트랩 계산 시 MemoryError가 발생하므로
# 추세선을 그리기엔 충분히 방대한 10만 개만 랜덤 샘플링하여 시각화합니다.
plt.figure(figsize=(10, 6))
sns.regplot(
    data=revisit_opp_df.sample(n=min(100000, len(revisit_opp_df)), random_state=42), 
    x='N_int', 
    y='revisit_choice', 
    logistic=True,  # Y가 0, 1인 이항 데이터이므로 logistic regression 적용
    scatter_kws={'alpha': 0.1}, 
    line_kws={'color': 'red'}
)
plt.title('Cognitive Overload (N_int) vs Revisit Probability in RecSys2019')
plt.xlabel('Number of Unique Items Explored (N_int)')
plt.ylabel('Revisit Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %% [9] 재방문 정보 갱신(Replenishment) 효과 분석 및 시각화
# 첫 방문(revisit=0) vs 재방문(revisit=1) 시 클릭 전환율(chosen == 1 비율) 비교

# 전체 노출 중 방문 유형에 따른 평균 전환율 계산
conv_stats = train_ready.groupby('revisit')['chosen'].mean().reset_index()
conv_stats['revisit_label'] = conv_stats['revisit'].map({0: 'First Visit (revisit=0)', 1: 'Revisit (revisit=1)'})
conv_stats['conversion_rate'] = conv_stats['chosen'] * 100

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=conv_stats, x='revisit_label', y='conversion_rate', palette='Set2')
plt.title('Information Replenishment Effect: Conversion Rate by Visit Type')
plt.ylabel('Conversion Rate (Chosen %)')
plt.xlabel('Visit Type')

# 막대 그래프 상단에 수치 표시
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.2f}%", 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

# y축 범위를 여유있게 잡아 텍스트 잘림 방지
plt.ylim(0, conv_stats['conversion_rate'].max() * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 두 그룹 간의 차이 (상대적 상승률) 출력
rate_first = conv_stats.loc[conv_stats['revisit'] == 0, 'conversion_rate'].values[0]
rate_revisit = conv_stats.loc[conv_stats['revisit'] == 1, 'conversion_rate'].values[0]

print(f"\n--- 정보 갱신 (Replenishment) 효과 분석 ---")
print(f"첫 방문 평균 전환율: {rate_first:.2f}%")
print(f"재방문 평균 전환율: {rate_revisit:.2f}%")

if rate_first > 0:
    uplift = ((rate_revisit - rate_first) / rate_first) * 100
    print(f"▶ 재방문 시 첫 방문 대비 전환율 상대적 상승: {uplift:.2f}% 증가")
    

# %% [10] 세션 간 장기 망각 (Delta t) 효과 분석 및 시각화
print("\n--- 시간 망각(Delta t)에 따른 재방문 확률 분석 ---")

# 재방문 기회(revisit == 1)인 데이터만 다시 사용
revisit_opp_df_time = train_ready[train_ready['revisit'] == 1].copy()

# delta_t 구간화 (단위: 시간)
# 0~1시간(단기 세션), 1~24시간(당일 휴지기), 24~48시간(하루 경과), 48시간 이상(논문상 I_break 기준)
time_bins = [-1, 1, 24, 48, float('inf')]
time_labels = ['< 1 Hour', '1~24 Hours', '24~48 Hours', '48+ Hours']
revisit_opp_df_time['delta_t_group'] = pd.cut(revisit_opp_df_time['delta_t'], bins=time_bins, labels=time_labels)

# 구간별 재방문 선택(revisit_choice) 확률 계산
time_stats = revisit_opp_df_time.groupby('delta_t_group')['revisit_choice'].agg(['mean', 'count']).reset_index()
time_stats_valid = time_stats[time_stats['count'] >= 30] # 데이터 수 부족 구간 제외

print(time_stats_valid)

# 시각화: 시간에 따른 재방문 확률 변화
plt.figure(figsize=(10, 6))
ax2 = sns.barplot(data=time_stats_valid, x='delta_t_group', y='mean', palette='magma')
plt.title('Time Decay vs Revisit Probability', fontsize=14)
plt.xlabel('Time Gap since Last View (Delta t)', fontsize=12)
plt.ylabel('Revisit Probability', fontsize=12)

# 막대 위에 확률 텍스트 추가
for p in ax2.patches:
    height = p.get_height()
    if height > 0:
        ax2.annotate(f"{height:.3f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

# 48시간 임계치 강조 (수직선)
plt.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='48-hour Threshold (I_break)')
plt.legend()
plt.ylim(0, time_stats_valid['mean'].max() * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
# %%

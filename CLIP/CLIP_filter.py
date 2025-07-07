import os
# json, COCO는 Flickr8k에서 사용하지 않으므로 제거하거나 그대로 두어도 무방합니다.
# 하지만 혼란을 줄이기 위해 지금은 제거하는 것이 좋습니다.
# from pycocotools.coco import COCO # COCO 데이터셋용이므로 제거
from PIL import Image
from tqdm import tqdm # CLIP 모델 임베딩 계산 시 사용 (아직 로드 부분에서는 사용 안 함)

# --- 설정 (Flickr8k에 맞게 수정) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_ROOT_DIR은 Flickr8k_Dataset 폴더 자체를 가리킵니다.
DATA_ROOT_DIR = os.path.join(BASE_DIR, 'data', 'Flickr8k_Dataset')

# 아래 두 줄은 사용자님의 실제 파일 이름에 맞춰 수정된 것입니다.
IMAGES_DIR = os.path.join(DATA_ROOT_DIR, 'Flicker8k_image') # 이미지들이 들어있는 폴더 경로 (Flicker8k_image)
CAPTIONS_FILE = os.path.join(DATA_ROOT_DIR, 'captions.txt') # 캡션 텍스트 파일 경로 (captions)

# COCO 데이터셋 관련 경로는 Flickr8k에서는 필요 없으므로 삭제합니다.
# ANNOTATIONS_FILE = os.path.join(DATA_ROOT_DIR, 'annotations', 'captions_val2017.json')

print(f"이미지 디렉토리: {IMAGES_DIR}")
print(f"캡션 파일: {CAPTIONS_FILE}") # COCO 어노테이션 대신 캡션 파일 경로를 출력

# --- Flickr8k 데이터셋 로드 ---
print("\nFlickr8k 데이터셋 로드 중...")
image_caption_pairs = []
try:
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        # 첫 줄 (헤더) 건너뛰기
        header = f.readline()
        if header.strip() != "image,caption": # 헤더가 예상과 다르면 경고
            print(f"경고: 캡션 파일의 헤더가 'image,caption'이 아닐 수 있습니다: {header.strip()}")

        for line in f:
            # 쉼표(,)로 분리
            parts = line.strip().split(',', 1) # 첫 번째 쉼표까지만 분리하여 캡션 안에 쉼표가 있어도 문제 없도록
            if len(parts) == 2:
                image_filename, caption_text = parts
                image_path = os.path.join(IMAGES_DIR, image_filename)
                image_caption_pairs.append({'image_path': image_path, 'caption': caption_text, 'image_id': image_filename})

    print(f"총 {len(image_caption_pairs)} 개의 이미지-캡션 쌍 로드됨.")

except FileNotFoundError:
    print(f"오류: 캡션 파일을 찾을 수 없습니다. 경로를 확인하세요: {CAPTIONS_FILE}")
    print("Flickr8k 데이터셋 다운로드 및 압축 해제가 제대로 되었는지, 그리고 경로 설정이 올바른지 확인해주세요.")
    exit()
except Exception as e:
    print(f"데이터셋 로드 중 예상치 못한 오류 발생: {e}")
    exit()

# --- 초기 데이터셋 샘플 확인 --- (이 부분은 동일하게 유지)
print("\n--- 데이터셋 샘플 (상위 5개) ---")
sample_pairs = image_caption_pairs[:5]

for i, pair in enumerate(sample_pairs):
    img_path = pair['image_path']
    caption = pair['caption']
    image_id = pair['image_id']
    print(f"샘플 {i+1} - 이미지 ID: {image_id}")
    print(f"  이미지 경로: {img_path}")
    print(f"  캡션: {caption}")
    try:
        with Image.open(img_path).convert("RGB") as img:
            print(f"  이미지 크기: {img.size}, 모드: {img.mode}")
    except FileNotFoundError:
        print(f"  **오류: 이미지 파일을 찾을 수 없습니다.** 경로를 확인하세요: {img_path}")
        print("  'Flicker8k_image' 폴더에 이미지가 제대로 압축 해제되었는지 확인해주세요.")
    except Exception as e:
        print(f"  **오류: 이미지 로드 중 문제 발생:** {e}")
    print("-" * 30)

print("\n데이터셋 로드 및 초기 탐색 완료. 다음 단계로 진행할 준비가 되었습니다.")

# --- CLIP 모델 로드 및 임베딩 계산 (이전 코드와 동일) ---
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # 이미 상단에 import 되어 있다면 다시 안 해도 됨

print("\nCLIP 모델 로드 중...")
device = "cpu" # LG gram이므로 CPU 사용
model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.to(device)

print(f"CLIP 모델 '{model_name}' 로드 완료. (디바이스: {device})")

print("\n이미지 및 텍스트 임베딩 계산 중... (시간이 다소 소요될 수 있습니다)")

all_image_features = []
all_text_features = []
all_similarities = []
processed_pairs = [] # 성공적으로 처리된 쌍 저장

BATCH_SIZE = 32 # 배치 사이즈 (메모리 사용과 속도에 영향)

# tqdm을 사용하여 진행 상황 바 표시
for i in tqdm(range(0, len(image_caption_pairs), BATCH_SIZE), desc="CLIP 임베딩 계산"):
    batch_pairs = image_caption_pairs[i:i + BATCH_SIZE]

    batch_images = []
    batch_texts = []
    current_processed_batch_pairs = []

    for pair in batch_pairs:
        try:
            img = Image.open(pair['image_path']).convert("RGB") # RGB로 변환하여 CLIP에 적합하게
            batch_images.append(img)
            batch_texts.append(pair['caption'])
            current_processed_batch_pairs.append(pair) # 성공적으로 로드된 페어만 추가
        except FileNotFoundError:
            # print(f"경고: 이미지 파일을 찾을 수 없습니다: {pair['image_path']}")
            continue # 다음 쌍으로 넘어감
        except Exception as e:
            # print(f"경고: 이미지 로드 또는 처리 중 오류 발생 ({pair['image_path']}): {e}")
            continue # 다음 쌍으로 넘어감

    if not batch_images: # 배치에 유효한 이미지가 없으면 건너뛰기
        continue

    try:
        # CLIP 프로세서로 이미지와 텍스트 전처리
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True # 텍스트 길이 제한
        ).to(device)

        with torch.no_grad(): # 역전파 계산 안함 -> 메모리 절약, 속도 향상
            outputs = model(**inputs)

        image_features = outputs.image_embeds # 이미지 임베딩
        text_features = outputs.text_embeds   # 텍스트 임베딩

        # 임베딩 정규화 (코사인 유사도 계산을 위해 필수)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # 코사인 유사도 계산 (두 벡터의 내적)
        similarities = (image_features * text_features).sum(dim=-1)

        all_image_features.extend(image_features.cpu().numpy())
        all_text_features.extend(text_features.cpu().numpy())
        all_similarities.extend(similarities.cpu().numpy())
        processed_pairs.extend(current_processed_batch_pairs) # 성공 처리된 쌍만 리스트에 추가

    except Exception as e:
        # print(f"경고: 배치 처리 중 오류 발생: {e}")
        continue # 이 배치는 건너뛰고 다음 배치로 진행

print(f"총 {len(processed_pairs)} 개의 이미지-캡션 쌍에 대해 임베딩 및 유사도 계산 완료.")



# 계산된 유사도 데이터를 최종 데이터셋에 추가
for i, pair in enumerate(processed_pairs):
    pair['clip_similarity'] = all_similarities[i]


# --- 단계 4: 결과 분석 및 시각화 ---
print("\n--- 결과 분석 및 시각화 ---")

# 1. 유사도 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(all_similarities, bins=50, kde=True)
plt.title('Distribution of CLIP Similarities')
plt.xlabel('CLIP Cosine Similarity')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show() # 그래프 창 띄우기

# 2. 필터링 임계값 설정
# 히스토그램을 보고 적절한 임계값을 선택하세요.
# 예시: 0.25 (LAION 논문에서 0.26~0.28 정도 사용)
FILTERING_THRESHOLD = 0.25

filtered_pairs = [pair for pair in processed_pairs if pair['clip_similarity'] >= FILTERING_THRESHOLD]
rejected_pairs = [pair for pair in processed_pairs if pair['clip_similarity'] < FILTERING_THRESHOLD]

print(f"\n총 처리된 쌍: {len(processed_pairs)} 개")
print(f"필터링 기준 (유사도 >= {FILTERING_THRESHOLD}): {len(filtered_pairs)} 개 (남은 데이터)")
print(f"필터링으로 제거된 쌍: {len(rejected_pairs)} 개 (제거된 데이터)")
print(f"제거된 비율: {len(rejected_pairs) / len(processed_pairs) * 100:.2f}%")

# 3. 제거된 데이터 샘플 확인
print("\n--- 제거된 데이터 샘플 (유사도 낮음) ---")
# 유사도가 낮은 순서로 정렬하여 확인
rejected_pairs_sorted = sorted(rejected_pairs, key=lambda x: x['clip_similarity'])

# 가장 유사도가 낮은 5개 샘플 출력
for i, pair in enumerate(rejected_pairs_sorted[:5]):
    img_path = pair['image_path']
    caption = pair['caption']
    similarity = pair['clip_similarity']
    print(f"샘플 {i+1} (유사도: {similarity:.4f})")
    print(f"  이미지 경로: {img_path}")
    print(f"  캡션: {caption}")
    try:
        img = Image.open(img_path)
        img.thumbnail((128, 128)) # 이미지 크기 줄여서 출력 (선택 사항)
        # img.show() # 이미지를 새 창으로 띄워볼 때 사용 (주피터 노트북이 아니라면 여러 창이 뜰 수 있음)
    except Exception as e:
        print(f"  이미지 로드 또는 표시 오류: {e}")
    print("-" * 30)

print("\n실습 완료!")
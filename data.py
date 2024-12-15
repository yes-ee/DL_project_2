import os
import json
from collections import Counter
from common import sequence

def extract_qa_pairs_with_topic(folder_path):
    all_qa_pairs = []  # 모든 질문-응답 쌍과 주제를 저장할 리스트

    # 폴더 내 모든 JSON 파일 탐색
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):  # JSON 파일만 처리
            file_path = os.path.join(folder_path, file_name)
            output_txt_path = os.path.join(folder_path, file_name.replace('.json', '.txt'))
            print(f"Processing file: {file_path}")

            # JSON 파일 열기
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data["dataset"]["conversations"]
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                for convo in conversations:
                    utterances = convo["utterances"]
                    for i in range(len(utterances) - 1):
                        # 질문과 응답 추출
                        if utterances[i]["speaker_id"] != 0 and utterances[i + 1]["speaker_id"] == 0:
                            question = utterances[i]["utterance_text"]
                            answer = utterances[i + 1]["utterance_text"]
                            # 질문-응답 쌍과 주제 추가
                            all_qa_pairs.append({
                                "question": question,
                                "answer": answer,
                            })
                            f.write(f"{question}_{answer}\n")
    return all_qa_pairs

def build_vocab(qa_pairs):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    idx = 3
    for q, a in qa_pairs:
        for sentence in [q, a]:
            for word in sentence.split():  # 단어 단위로 처리
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, reverse_vocab

def build_vocab_with_frequency(qa_pairs, min_freq=3, max_vocab_size=20000):
    # 기본 특수 토큰
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idx = 4
    word_counter = Counter()

    # 단어 빈도 계산
    for q, a in qa_pairs:
        for sentence in [q, a]:
            word_counter.update(sentence.split())

    # 빈도 기준과 최대 크기 기준으로 단어 추가
    for word, count in word_counter.most_common(max_vocab_size - len(vocab)):
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, reverse_vocab

def preprocess_data(qa_pairs, vocab, max_len):
    def encode_sentence(sentence, vocab):
        return [vocab["<SOS>"]] + [vocab[word] for word in sentence.split() if word in vocab] + [vocab["<EOS>"]]

    def pad_or_truncate(seq, max_len, pad_token=0):
        if len(seq) > max_len:
            return seq[:max_len]  # 초과 부분 잘라내기
        return seq + [pad_token] * (max_len - len(seq))  # 패딩 추가

    questions, answers = [], []
    for pair in qa_pairs:
        encoded_q = encode_sentence(pair["question"], vocab)
        encoded_a = encode_sentence(pair["answer"], vocab)
        # 패딩 처리
        questions.append(pad_or_truncate(encoded_q, max_len))
        answers.append(pad_or_truncate(encoded_a, max_len))
    return questions, answers

# 폴더 내 모든 JSON 파일에서 질문-응답 쌍과 주제 추출
folder_path = "./train_1_entertainment"  # JSON 파일이 있는 폴더 경로
qa_pairs = extract_qa_pairs_with_topic(folder_path)

# 결과 출력
print(f"총 {len(qa_pairs)}개의 질문-응답 쌍을 추출했습니다.")
print("샘플 데이터:", qa_pairs[:3])  # 일부 샘플 출력

# 단어 사전 생성
# vocab, reverse_vocab = build_vocab([(item["question"], item["answer"]) for item in qa_pairs])
# print("단어 사전 크기:", len(vocab))

# 빈도 기반 단어 사전 생성
vocab, reverse_vocab = build_vocab_with_frequency([(item["question"], item["answer"]) for item in qa_pairs])
print("빈도 기반 단어 사전 크기:", len(vocab))

# 데이터 전처리
max_len = 50  # 문장 최대 길이

x_data, t_data = preprocess_data(qa_pairs, vocab, max_len)
print("패딩 처리된 질문 샘플:", x_data[0])
print("패딩 처리된 응답 샘플:", t_data[0])

# 학습/검증/테스트 데이터셋 분리
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = sequence.load_data(
    x_data, t_data, valid_ratio=0.1, test_ratio=0.1
)

# 데이터 크기 출력
total_samples = len(x_data)
print(f"전체 데이터 크기: {total_samples} / Train 비율: {len(x_train) / total_samples:.2%} / Validation 비율: {len(x_valid) / total_samples:.2%} / Test 비율: {len(x_test) / total_samples:.2%}")

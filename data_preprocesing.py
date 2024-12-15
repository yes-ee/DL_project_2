import sys
sys.path.append('..')
from common.np import *
import os
import json
from collections import Counter
from common import sequence

# 전체 전처리 파이프라인
def full_preprocess_pipeline(folder_path, max_len, min_freq=3, max_vocab_size=20000, test_ratio=0.2, valid_ratio=0.1):
    def extract_qa_pairs_with_topic(folder_path):
        print(f"Processing folder: {folder_path}")

        all_qa_pairs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    print(f"Processing file: {file_path}")
                    data = json.load(f)
                    conversations = data["dataset"]["conversations"]
                    for convo in conversations:
                        utterances = convo["utterances"]
                        for i in range(len(utterances) - 1):
                            if utterances[i]["speaker_id"] != 0 and utterances[i + 1]["speaker_id"] == 0:
                                question = utterances[i]["utterance_text"]
                                answer = utterances[i + 1]["utterance_text"]
                                all_qa_pairs.append({
                                    "question": question,
                                    "answer": answer,
                                })
        return all_qa_pairs

    def build_vocab_with_frequency(qa_pairs, min_freq, max_vocab_size):
        """단어 빈도를 기반으로 단어 사전 생성"""
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        idx = 4
        word_counter = Counter()

        for q, a in qa_pairs:
            for sentence in [q, a]:
                word_counter.update(sentence.split())

        for word, count in word_counter.most_common(max_vocab_size - len(vocab)):
            if count >= min_freq:
                vocab[word] = idx
                idx += 1

        reverse_vocab = {idx: word for word, idx in vocab.items()}
        return vocab, reverse_vocab

    def preprocess_data(qa_pairs, vocab, max_len):
        """데이터 정수 인코딩 및 패딩 처리"""
        def encode_sentence(sentence, vocab):
            return [vocab["<SOS>"]] + [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()] + [vocab["<EOS>"]]

        def pad_or_truncate(seq, max_len, pad_token=0):
            if len(seq) > max_len:
                return seq[:max_len]
            return seq + [pad_token] * (max_len - len(seq))

        questions, answers = [], []
        for pair in qa_pairs:
            encoded_q = encode_sentence(pair["question"], vocab)
            encoded_a = encode_sentence(pair["answer"], vocab)
            # 패딩 처리
            questions.append(pad_or_truncate(encoded_q, max_len))
            answers.append(pad_or_truncate(encoded_a, max_len))
        return questions, answers

    # 1. 질문-응답 추출
    qa_pairs = extract_qa_pairs_with_topic(folder_path)
    print(f"총 {len(qa_pairs)}개의 질문-응답 쌍을 추출했습니다.")

    # 2. 단어 사전 생성
    vocab, reverse_vocab = build_vocab_with_frequency(
        [(item["question"], item["answer"]) for item in qa_pairs],
        min_freq=min_freq, max_vocab_size=max_vocab_size
    )
    print(f"단어 사전 크기: {len(vocab)}")

    # 3. 정수 인코딩 및 패딩
    x_data, t_data = preprocess_data(qa_pairs, vocab, max_len)
    print(f"패딩 처리된 질문 데이터 샘플: {x_data[0]}")
    print(f"패딩 처리된 응답 데이터 샘플: {t_data[0]}")

    x_data = np.array(x_data, dtype=np.int32)
    t_data = np.array(t_data, dtype=np.int32)
    print(f"x_data shape: {x_data.shape}")
    print(f"t_data shape: {t_data.shape}")
    unk_count = np.sum(np.asarray(x_data == vocab["<UNK>"], dtype=np.int32))
    total_count = int(np.prod(np.array(x_data.shape)))

    unk_ratio = unk_count / total_count
    print(f"<UNK> 비율: {unk_ratio:.2%}")

    # 학습/검증/테스트 데이터셋 분리
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = sequence.load_data(
        x_data, t_data, valid_ratio=0.1, test_ratio=0.1
    )

    # 데이터 크기 출력
    total_samples = len(x_data)
    print(f"전체 데이터 크기: {total_samples} / Train 비율: {len(x_train) / total_samples:.2%} / Validation 비율: {len(x_valid) / total_samples:.2%} / Test 비율: {len(x_test) / total_samples:.2%}")

    print(f"x_train shape: {x_train.shape}, t_train shape: {t_train.shape}")
    print(f"x_valid shape: {x_valid.shape}, t_valid shape: {t_valid.shape}")
    print(f"x_test shape: {x_test.shape}, t_test shape: {t_test.shape}")

    return (x_train, t_train), (x_valid, t_valid), (x_test, t_test), vocab, reverse_vocab

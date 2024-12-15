import os
import json

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

def encode_sentence(sentence, vocab):
    return [vocab["<SOS>"]] + [vocab[word] for word in sentence.split() if word in vocab] + [vocab["<EOS>"]]


# 폴더 내 모든 JSON 파일에서 질문-응답 쌍과 주제 추출
folder_path = "./train_1_entertainment"  # JSON 파일이 있는 폴더 경로
qa_pairs = extract_qa_pairs_with_topic(folder_path)

# 결과 출력
print(f"총 {len(qa_pairs)}개의 질문-응답 쌍을 추출했습니다.")
print("샘플 데이터:", qa_pairs[:3])  # 일부 샘플 출력

# 단어 사전 생성
vocab, reverse_vocab = build_vocab([(item["question"], item["answer"]) for item in qa_pairs])
print("단어 사전 크기:", len(vocab))

# 샘플 정수 인코딩
encoded_sample = encode_sentence(qa_pairs[0]["question"], vocab)
print("샘플 질문 인코딩:", encoded_sample)
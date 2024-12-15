import os
import json

def extract_qa_pairs_with_topic(folder_path):
    all_qa_pairs = []  # 모든 질문-응답 쌍과 주제를 저장할 리스트

    # 폴더 내 모든 JSON 파일 탐색
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):  # JSON 파일만 처리
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")

            # JSON 파일 열기
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations = data["dataset"]["conversations"]

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
    return all_qa_pairs


# 폴더 내 모든 JSON 파일에서 질문-응답 쌍과 주제 추출
folder_path = "./train_1_entertainment"  # JSON 파일이 있는 폴더 경로
qa_pairs_with_topics = extract_qa_pairs_with_topic(folder_path)

# 결과 출력
print(f"총 {len(qa_pairs_with_topics)}개의 질문-응답 쌍을 추출했습니다.")
print("샘플 데이터:", qa_pairs_with_topics[:3])  # 일부 샘플 출력

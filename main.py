# coding: utf-8
import sys

import numpy as np

sys.path.append('..')
from common.np import *
import matplotlib.pyplot as plt
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from common.util import eval_seq2seq_batch
from seq2seq import Seq2seq
from data_preprocesing import full_preprocess_pipeline

import cupy as cp

# GPU 캐시 초기화
cp.get_default_memory_pool().free_all_blocks()

# 간단한 배열 연산
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = x + y
print(z)  # 출력: [5 7 9]

# GPU 메모리 확인
print(np.cuda.runtime.getDeviceCount(), "개의 GPU가 감지되었습니다.")


# 데이터셋 읽기
folder_path = "./train_1_entertainment"
max_len = 50
min_freq = 2
max_vocab_size = 20000

# 전처리 실행
(x_train, t_train), (x_valid, t_valid), (x_test, t_test), vocab, reverse_vocab = full_preprocess_pipeline(
    folder_path=folder_path,
    max_len=max_len,
    min_freq=min_freq,
    max_vocab_size=max_vocab_size
)

# 결과 확인
print("첫 번째 질문 샘플:", x_train[0])
print("첫 번째 응답 샘플:", t_train[0])

# 하이퍼파라미터 설정
vocab_size = len(vocab)
wordvec_size = 128
hidden_size = 256
batch_size = 64
max_epoch = 1
max_grad = 5.0

model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
best_acc = 0
patience = 3
#
# # 로그 파일 열기
log_file = open("training_log.txt", "w")
print("로그 파일 생성 완료.")
#
# for epoch in range(max_epoch):
#     trainer.fit(x_train, t_train, max_epoch=1,
#                 batch_size=batch_size, max_grad=max_grad)
#
#     # 검증 데이터 정확도 평가
#     patience_counter = 0
#     batch_size = 128
#     # correct_num = 0
#     #
#     # for i in range(len(x_test)):
#     #     question, correct = x_valid[[i]], t_valid[[i]]
#     #     verbose = i < 10
#     #     # correct_num += eval_seq2seq(model, question, correct, reverse_vocab)
#     #
#     # acc = float(correct_num) / len(x_valid)
#
#     acc = eval_seq2seq_batch(model, x_valid, t_valid, reverse_vocab, verbos=False, batch_size=batch_size)
#     acc_list.append(acc)
#     print("Acc List:", acc_list)
#     log_line = f"Epoch {epoch + 1}: Validation Accuracy = {acc:.3f}\n"
#     log_file.write(log_line)
#     print('검증 정확도 %.3f%%' % (acc * 100))
#
#     # GPU 메모리 해제
#     cp._default_memory_pool.free_all_blocks()
#     print("GPU 메모리 해제 완료.")
#     free_mem, total_mem = cp.cuda.runtime.memGetInfo()
#     print(f"GPU 메모리 사용량: {total_mem - free_mem:.2f} bytes")
#
#     # Early Stopping 체크
#     if acc > best_acc:
#         best_acc = acc
#         patience_counter = 0
#     else:
#         patience_counter += 1
#         if patience_counter >= patience:
#             log_file.write("Early stopping triggered. Training stopped.\n")
#             print('Early stopping 발동. 학습 종료.')
#             break

# model.save_params("seq2seq_model_params.pkl")  # 학습 후 저장
model.load_params("seq2seq_model_params.pkl")  # 추후 로드

# 그래프 그리기
x = np.arange(len(acc_list)).get()
acc_list_np = np.array(acc_list).get()
plt.plot(x, acc_list_np, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.ylim(0, 1.0)
plt.show()

print("테스트 데이터 평가 중 ...")
# 테스트 데이터 평가 (배치 처리)
batch_size = 128  # 테스트 시 배치 크기 설정
test_acc = eval_seq2seq_batch(model, x_test, t_test, reverse_vocab, verbos=False, batch_size=batch_size)

test_log_line = f"Test Accuracy: {test_acc:.3f}\n"
print(test_log_line.strip())
log_file.write(test_log_line)

log_file.close()
print('테스트 정확도 %.3f%%' % (test_acc * 100))


def generate_response(model, vocab, reverse_vocab, question, max_len=50):
    """Seq2Seq 모델로 답변 생성"""
    # 질문을 정수로 인코딩
    def encode_sentence(sentence, vocab):
        return [vocab["<SOS>"]] + [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()] + [vocab["<EOS>"]]

    # 정수 인코딩 및 패딩
    encoded_question = encode_sentence(question, vocab)
    padded_question = encoded_question + [vocab["<PAD>"]] * (max_len - len(encoded_question))

    # numpy로 변환 및 모델 입력
    xs = np.array([padded_question])  # 배치 형태로 변환
    start_id = vocab["<SOS>"]
    sample_size = max_len

    # 답변 생성
    generated_ids_np = model.generate(xs, start_id, sample_size).get()
    # 생성된 답변을 문자열로 변환
    response = ' '.join(reverse_vocab[id] for id in generated_ids_np[0] if id in reverse_vocab and id != vocab["<EOS>"])
    return response

# 질문 입력
print("Seq2Seq Chatbot이 준비되었습니다! 종료하려면 'exit'를 입력하세요.")
while True:
    user_input = input("질문: ")
    if user_input.lower() == "exit":
        print("대화를 종료합니다.")
        break
    bot_response = generate_response(model, vocab, reverse_vocab, user_input, max_len=50)
    print("답변:", bot_response)
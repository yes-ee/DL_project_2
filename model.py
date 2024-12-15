import sys

sys.path.append('..')
from common.np import *

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

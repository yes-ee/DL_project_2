# coding: utf-8
from common.config import GPU


if GPU:
    import cupy as np
    from cupyx import scatter_add
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    # scatter_add를 사용하는 helper 함수 정의
    def add_at(a, indices, b):
        scatter_add(a, indices, b)
    np.add_at = add_at  # CuPy용 add_at 함수로 재정의

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else:
    import numpy as np
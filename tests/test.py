import numpy as np
from collections import namedtuple

Test = namedtuple('Test', ['a', 'b', 'done'])

test = Test(1, 2, True)
for _ in range(10):
    if test.done:
        print('done')
        test = test._replace(done=False)
    else:
        print('not done')
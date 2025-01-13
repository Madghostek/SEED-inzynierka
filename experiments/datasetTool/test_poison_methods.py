from poison_methods import WhiteSquare
import numpy as np
from argparse import Namespace # this dependency is annoying

def test_white_square():
    fake_image = np.ones((100,100,3))
    params = Namespace(opacity=1)
    poison = WhiteSquare(None, None, params)

    assert not np.all(fake_image[0:3,0:3]==255)

    result_img,_ = poison.poison(fake_image, 0)

    assert np.all(result_img[0:3,0:3]==255)
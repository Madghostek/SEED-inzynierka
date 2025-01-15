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

def test_white_square_half_opacity():
    fake_image2 = np.ones((100,100,3))
    fake_image2[0,0] = (100,100,100)
    params = Namespace(opacity=0.5)
    poison = WhiteSquare(None, None, params)

    result_img,_ = poison.poison(fake_image2, 0)

    assert result_img[0,0,0]==float((100+255)/2)

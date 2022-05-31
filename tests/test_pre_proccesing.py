import numpy as np
import pandas as pd
from app.preproccessing import transform_data
from pandas.testing import assert_frame_equal

def test_transorm_data():
    """
    Test transform_data function
    """
    test_data = {'test': [1,np.nan,3,4,5,6,np.nan,8,9,10,11,12],
                'category': ['a','b','c','d','e','f','g','h','i','j','k','l'],
                'Transported': [True, False, True, False, True, False, True, False, True, False, True, False]}
    df = pd.DataFrame(test_data)
    df = transform_data(df)
    expected_df = pd.DataFrame({'Transported': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                        'test': [-1.69683206e+00, -1.77635684e-16, -1.12163475e+00, -8.34036097e-01,
                                -5.46437443e-01, -2.58838789e-01, -1.77635684e-16,  3.16358520e-01,
                                6.03957174e-01,  8.91555828e-01,  1.17915448e+00,  1.46675314e+00],
                        'category': [-1.59325501, -1.30357228, -1.01388955, -0.72420682, -0.43452409,
                                -0.14484136,  0.14484136,  0.43452409,  0.72420682,  1.01388955,
                                1.30357228,  1.59325501] })
    assert assert_frame_equal(df, expected_df) is None
    


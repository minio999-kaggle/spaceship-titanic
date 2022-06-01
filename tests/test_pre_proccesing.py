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
    df = df.round(decimals=5)
    expected_df = pd.DataFrame({'Transported': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                        'test': [-1.69683, -0.00000, -1.12163, -0.83404, -0.54644, -0.25884,
                                -0.00000,  0.31636,  0.60396,  0.89156,  1.17915,  1.46675],
                        'category': [-1.59326, -1.30357, -1.01389, -0.72421, -0.43452, -0.14484,
                                0.14484,  0.43452,  0.72421,  1.01389,  1.30357,  1.59326]})
    assert df.equals(expected_df)
    


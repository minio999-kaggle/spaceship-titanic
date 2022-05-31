import numpy as np
import pandas as pd
from app.preproccessing import transform_data

def test_transorm_data():
    """
    Test transform_data function
    """
    test_data = {'test': [1,np.nan,3,4,5,6,np.nan,8,9,10,11,12],
                'category': ['a','b','c','d','e','f','g','h','i','j','k','l'],
                'Transported': [True, False, True, False, True, False, True, False, True, False, True, False]}
    df = pd.DataFrame(test_data)
    df = transform_data(df)
    assert df.equals({'Transported': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                        'test': [-1.696832e+00, -1.776357e-16, -1.121635e+00, -8.340361e-01, -5.464374e-01, -2.588388e-01,
                        -1.776357e-16, 3.163585e-01, 6.039572e-01, 8.915558e-01, 1.179154e+00, 1.466753e+00],
                        'category': [-1.593255, -1.303572, -1.013890, -0.724207, -0.434524, -0.144841, 0.144841
                        ,0.434524, 0.724207, 1.013890, 1.303572, 1.593255] })

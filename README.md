# Spaceship titanic predictions
Spaceship Titanic classification [challenge on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic).
To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

## Credits
* Maciej Bialoglowski  ([@chemista](https://github.com/chemista))

## Method
Below are provided steps that I followed for this Project.

### 1. **Data visualization**: Data analisys to understand features, missing values, mean values (for further use) and other usefull information.
- Understanding features
    - **PassengerId**: A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
    - **HomePlanet**:  The planet the passenger departed from, typically their planet of permanent residence.
    - **CryoSleep**: Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
    - **Cabin**: The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
    - **Destination**: The planet the passenger will be debarking to.
    - **Age**: The age of the passenger.
    - **VIP**: Whether the passenger has paid for special VIP service during the voyage.
    - **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
    - **Name**: The first and last names of the passenger.
    - **Transported**: Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
- Looking out for null values
([Jupyters notebook](https://github.com/minio999-kaggle/spaceship-titanic/blob/dev/playground/preProccesingData.ipynb))

We can see that there are some null values in data set.
- Getting better knowlage about data
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/dev/playground/eda1.ipynb))

We can see as well that there are some objects in data set that we will want to preproccess it.

- Conclusion
Firstly we want encode objects into float. Then we will need to get rid of null values by imputing. At the end we need to scale values for further use of PCA.

### 2. **Preprocessing**: with the knowledge acquired with data visualization, we can apply it to dealing with missing values and categorical data.

- Encoding
([Jupyters notebook](https://github.com/minio999-kaggle/titanic/blob/dev/playground/eda1.ipynb))
- Scaling Features
([Jupyters notebook](https://github.com/minio999-kaggle/spaceship-titanic/blob/dev/playground/fe1.ipynb))

- Imputer
([Jupyters notebook](https://github.com/minio999-kaggle/spaceship-titanic/blob/dev/playground/fe1.ipynb))
- Aplying Preproccesing to data
([Jupyters notebook](https://github.com/minio999-kaggle/spaceship-titanic/blob/dev/playground/fe1.ipynb))
- Cross-Validation
([Jupyters notebook](https://github.com/minio999-kaggle/spaceship-titanic/blob/dev/playground/fe1.ipynb))
- PCA
([Jupyters notebook](https://github.com/minio999-kaggle/spaceship-titanic/blob/dev/playground/fe1.ipynb))
## Folder Structures
* `\` contains all of setup files
* `\data` contains data
* `\src\app` contains code
* `\playground` contains jupyter notebooks for test purpose
* `\tests` contains test

## Installation instructions
1. Install Python and clone this repository
2. Open files, find cloned repository, open terminal inside that folder and use command `./run.sh`

to run the [jupyter](http://jupyter.org/)'s notebooks or mess with it yourself download docker, open powershell and run `.\jupyter-start.ps1`

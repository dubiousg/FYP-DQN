<html>
<h1> The Model </h1>
  <div>
This model folder contains a python package. <br>
If this project is downloaded an empty __init__.py file
needs to be added in order to run the other files.
  </div>
<h2> clean_data.py </h2>
<div> 
Contains the data_cleaner class which is used to process stock data stored in csv files.<br>
This class can save cleaned csv files, containing the extracted features.

These features include: stock's close, open, high and low price, volume, 
relative strength index, on balance volume, simple moving average.

</div>
<h2> market.py</h2>
<div> 
Contains the Portfolio and Market_Environment classes.<br>
The portolio contains the stocks and cash of the trader.<br>
The market environment holds the stock data and interacts with the trader.
</div>
<h2> trader.py</h2>
<div>
Contains a model class which inherits from the tf.keras.model class.<br>
The model selects how much of each stock to buy.<br>
Also contains the DQN class which is used to create the target and training<br>
DQNs. This class can learn to trade stocks and save and load models.
</div>
</html>

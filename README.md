# IPL Score Predictor

A machine learning-powered application that predicts IPL cricket match scores with high accuracy using various statistical features and historical data.

![IPL Score Predictor](streamlit_app/ipl_score_predictor.gif)

## Features

- Real-time score prediction for ongoing IPL matches
- Modern, responsive web interface built with Streamlit
- Support for all IPL teams
- Considers multiple factors including:
  - Current score and overs
  - Wickets fallen
  - Recent performance (last 5 overs)
  - Team matchups
- Beautiful UI with glassmorphism design

## Machine Learning Models Used

- XGBoost Regressor (primary model)
- Features comprehensive data preprocessing and label encoding
- Trained on historical IPL data from 2008-2020

## Tech Stack

- Python 3.10+
- Streamlit for web interface
- XGBoost for ML model
- Pandas & NumPy for data processing
- Scikit-learn for preprocessing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zzzfbx2/ipl-score-predictor.git
cd ipl-score-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

4. Run the application:
```bash
streamlit run ipl_score_predictor.py
```

## Usage

1. Select the batting and bowling teams
2. Enter the current match statistics:
   - Current over (minimum 5 overs required)
   - Current runs
   - Wickets fallen
   - Performance in last 5 overs
3. Click "Predict Score" to get the predicted score range

## Dataset

The dataset (`ipl_data.csv`) includes:
- Match details from 2008 to 2020
- Ball-by-ball information
- Team statistics
- Over-by-over runs and wickets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original dataset from IPL statistics
- Inspired by the work of cricket analysts and data scientists
- Thanks to the Streamlit team for the amazing framework
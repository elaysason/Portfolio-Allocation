# Portfolio-Allocation
Exploring diffrent methods to optimize portfolio allocation 

1. [General](#General)
    - [Background](#background)
3. [Program Structure](#Program-Structure)
    - [Experiments](#Experiments)
5. [Installation](#Installation)
   
## General
This project aims to optimize stock portfolios using deep learning methods, with a particular focus on recurrent neural networks (RNNs). The team consists of data scientist engineers from Technion, who decided to leverage deep learning techniques due to their wide applicability and potential in solving complex problems. The primary approach involved using a basic RNN network with LSTM architecture to predict the stock performance for the next day based on historical data. However, due to the stochastic nature of the stock market, the RNN-LSTM model did not consistently outperform traditional portfolio optimization methods in the long run.

### Background
The team initially tested standard portfolio optimization methods taught in finance courses, such as the minimum variance portfolio, market portfolio, and best basket portfolio. Unfortunately, these methods yielded unfavorable results with negative or very low Sharpe ratios.

## Program Structure
The project consists of a `Portfolio` class, responsible for handling various portfolio optimization methods. The class includes functions for equal weight market portfolio, tangent portfolio, minimum variance portfolio, and more.

### Experiments
To determine the optimal hyperparameters, the team conducted a comprehensive cross-validation process. They tested various numbers of stocks in the short-buy method, different weights, different splits in the short-buy and regularized minimum variance portfolios, and more. The goal was to avoid overfitting on specific periods, so the team tested on several training and test dates. Ultimately, the parameters yielding the best overall results across multiple dates were selected.

Please note that this project focused on portfolio optimization rather than emotion prediction, as initially mentioned. As a result, the experiments and methods discussed pertain solely to portfolio optimization strategies.

The final method employed a combination of two strategies:
1. **Short-Buy Method:** The team predicted the best and least performing stocks based on the last 5 days using a simple linear regression. They allocated a high weight to the top-performing stocks and shorted the underperforming stocks.
2. **Regularized Min Variance Portfolio:** For the remaining days in the month, the team utilized the regularized minimum variance portfolio method to optimize the portfolio allocation.

### Network Structure
The `get_portfolio()` function is used to obtain the model's portfolio for the next day. It takes a DataFrame of training data as input, and the portfolio is calculated accordingly.

## Installation
1. Open the terminal

2. Clone the project by:
```
    $ git clone https://github.com/elaysason/Portfolio-Selection.git
```
3. Run the portfolio.py file by:
```
    $ python portfolio.py```



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import optimize
import scipy
import cvxpy as cp
from sklearn.covariance import empirical_covariance
from sklearn.linear_model import LinearRegression
import pickle

class Portfolio:
    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        data = pd.read_pickle('data.pd')
        # self.min_var_portfolio = self.min_variance_portfolio(data)
        self.reg_min_var_portfolio = self.min_variance_portfolio(data)
        # self.portfolio = self.max_sharpe_cv(data)
        # self.portfolio = self.market_portfolio(data)
        # self.best_bask_portfolio = self.best_basket_portfolio(data)
        self.portfolio = self.find_k_best_stock(data, k=5, days=5, weight=1000)
        self.first_day = 16
        self.count = 0

        # self.portfolio = self.tangent_portfolio(train_data=data)
        # self.portfolio = self.max_sharpe(data)
        # np.save("itr=10000", self.portfolio)

        print('portfolios are calculated')

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        if self.count > self.first_day:
            return self.portfolio
        else:
            self.count += 1
            return self.reg_min_var_portfolio


    def market_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        Equal weight for each stock.
        """
        train_data = train_data['Adj Close']
        return np.ones(len(train_data.columns)) / len(train_data.columns)

    def new_and_improved_market_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        Equal weight for each stock, besides the nans
        """
        train_data = train_data['Adj Close']
        relative_price = train_data.mean()
        return relative_price / relative_price.sum()

    def tangent_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        train_data = train_data['Adj Close']
        R = train_data.mean()
        e = np.ones(R.size)
        cov = train_data.cov()
        cov_inverse = np.linalg.inv(cov)
        tangent_portfolio = (cov_inverse @ R) / (e.T @ cov_inverse @ R)
        return tangent_portfolio

    def min_variance_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        train_data = train_data['Adj Close']
        cov = train_data.cov()
        e = np.ones(len(cov))
        cov_inverse = np.linalg.inv(cov)
        min_variance_portfolio = (cov_inverse @ e) / (e.T @ cov_inverse @ e)
        return min_variance_portfolio

    def regularized_min_variance_portfolio(self, train_data: pd.DataFrame, tau=0.1) -> np.ndarray:
        train_data = train_data['Adj Close']
        cov = train_data.cov()
        port = cp.Variable(len(cov))
        objective = cp.Minimize(cp.quad_form(port, cov) + tau * cp.norm(port, 1))
        contrains = [sum(port) == 1]
        prob = cp.Problem(objective, contrains)
        result = prob.solve()
        return port.value

    def best_basket_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        train_data = train_data['Adj Close']
        mean = train_data.mean()
        cov = train_data.cov()
        e = np.ones(len(cov))
        cov_inverse = np.linalg.inv(cov)
        best_basket_portfolio = (cov_inverse @ mean) / (e.T @ cov_inverse @ mean)
        return best_basket_portfolio

    def find_k_best_stock(self, train_data: pd.DataFrame, k: int, days: int, weight):
        train_data = train_data['Adj Close']
        regressions = []
        for stock in train_data:
            reg = LinearRegression().fit(np.arange(days).reshape(-1, 1), train_data[stock][-days:])
            regressions.append(reg.coef_[0])#/np.std(train_data[stock][-days:]))
        indices = np.argsort(regressions)
        best = indices[-k:]
        worst = indices[:k]
        return self.allocate_best_and_short_worst(best, worst, len(regressions), weight)

    def allocate_best_and_short_worst(self, best, worst, size, weight):
        portfolio = np.zeros(size)
        portfolio[best] = weight+1/len(best)
        portfolio[worst] = -weight
        return portfolio


    def max_sharpe(self, train_data: pd.DataFrame) -> np.ndarray:
        # define maximization of Sharpe Ratio using principle of duality
        def f(x, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
            funcDenomr = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T))
            funcNumer = np.matmul(np.array(MeanReturns), x.T) - RiskFreeRate
            func = -(funcNumer / funcDenomr)
            return func

        # define equality constraint representing fully invested portfolio
        def constraintEq(x):
            A = np.ones(x.shape)
            b = 1
            constraintVal = np.matmul(A, x.T) - b
            return constraintVal

        # define bounds and other parameters
        train_data = train_data['Adj Close']
        train_data = train_data[:-10]
        mean = train_data.mean()
        cov = train_data.cov()
        PortfolioSize = len(mean)
        RiskFreeRate = 0
        xinit = np.repeat(1 / PortfolioSize, PortfolioSize)
        cons = ({'type': 'eq', 'fun': constraintEq})

        # invoke minimize solver
        opt = optimize.minimize(f, x0=xinit, args=(mean, cov,
                                                   RiskFreeRate, PortfolioSize), method='SLSQP',
                                constraints=cons, tol=10 ** -3, options={'maxiter': 10000, 'disp': True})

        return scipy.special.softmax(opt.jac)

    def max_sharpe_cv(self, train_data: pd.DataFrame) -> np.ndarray:
        train_data = train_data['Adj Close']
        mu = train_data.mean()
        cov = train_data.cov()
        cov = np.cov(train_data.T)
        cov = empirical_covariance(train_data)
        print(np.all(np.linalg.eigvals(cov) > 0))
        n = len(mu)
        # cov = np.eye(n)
        w = cp.Variable(n)
        gamma = cp.Parameter(nonneg=True)
        ret = w @ mu.T
        risk = cp.quad_form(w, cov)
        gamma.value = 1
        prob = cp.Problem(cp.Maximize(risk), [cp.sum(w) == 1])
        # prob = cp.Problem(cp.Maximize(risk/ret), [cp.sum(w) == 1, w >= 0])
        prob.solve(solver='ECOS', verbose=True)
        return w.value

#     def lstm_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
#         train_data = train_data['Adj Close']
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         stocks = len(train_data.columns)
#         model = LSTM(stocks, 64, 3, 0.2)
#         model = model.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#         epochs = 10
#         losses = []
#
#         data = {'accurcy_eph_train': [], 'accurcy_eph_test': []}
#
#         for e in range(epochs):
#             h0, c0 = model.init_hidden()
#             h0 = h0.to(device)
#             c0 = c0.to(device)
#             batch_acc_train = []
#             data['all_outputs_train'] = []
#             data['all_labels_train'] = []
#             data['all_outputs_test'] = []
#             data['all_labels_test'] = []
#             losses_batch_train = []
#             for batch_idx, batch in enumerate(train_data):
#                 input = batch[0].to(device)
#                 target = batch[1].to(device)
#                 optimizer.zero_grad()
#                 with torch.set_grad_enabled(True):
#                     out, hidden = model(input, (h0, c0))
#                     _, preds = torch.max(out, 1)
#                     loss = criterion(out, target)
#                     data['all_outputs_train'].extend(preds.cpu().detach().numpy())
#                     data['all_labels_train'].extend(target.cpu().detach().numpy())
#                     losses_batch_train.append(loss)
#
#                     loss.backward()
#                     optimizer.step()
#             data['accurcy_eph_train'].append(accuracy_score(data['all_outputs_train'], data['all_labels_train']))
#             data['epoch_loss_train'].append((sum(losses_batch_train) / len(losses_batch_train)).cpu().detach().numpy())
#             losses.append(loss.item())
#
# class LSTM(torch.nn.Module):
#     def __init__(self, stocks, hidden_dim, num_layer, dropout):
#         super().__init__()
#         self.lstm = nn.LSTM(stocks, hidden_dim, batch_first=True, num_layers=num_layer, dropout=dropout)
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(hidden_dim, stocks)
#
#     def forward(self, x, hidden):
#         out, hidden = self.lstm(x, hidden)
#         out = self.dropout(out)
#         out = self.linear(out)
#         out = out[:, -1]
#         out = nn.Softmax(out)
#         return out, hidden

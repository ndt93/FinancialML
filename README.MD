# Financial Machine Learning Library

Applying machine learning and statistical methods to financial and markets data is deceptively difficult.
Both classical and modern ML models and testing methods in their original form, although having been applied
successfully in many fields, typically don't work on financial data.

This library implements a selected set of machine learning and statistical data structures and methods,
motivated by the latest financial academic researches and practical industry experiences, that are modified and designed
specially for handling financial and markets microstructure data. These tools are essential for either financial machine
learning researchers, investment portfolio managers or quantitative trading practitioners.

## Implementation Roadmap [WIP]

This library only provides the tools necessary for academic research or investment strategies design and management and
not any specific investment/trading models or signals generation.
Most of these tools are currently only available through very
expensive commercial systems and software.
The derivatives and portfolio management tools maybe split into separate projects

### Financial ML & Statistical Methods
- [x] Standard & information driven bars sampling
- [x] Securities basket ETF trick and roll over adjustment
- [x] Triple barrier labeling
- [x] Label weighting by uniqueness, returns attribution and time decay
- [x] Fractionally differentiated time series
- [x] Cross validation
  - [x] Purged K-Fold
  - [x] Embargo
- [x] Features importance
  - [x] Mean decrease impurity
  - [x] Mean decrease accuracy
- Backtesting
  - [x] Strategy-independent position sizing
  - [x] Walk-forward
  - [x] Combinatorial purged CV
  - Synthetic data backtest
- [x] Evaluation methods & metrics
  - [x] Backtest metrics
  - [x] Attribution
- [x] Strategy risks
- [x] Assets allocation
  - [x] Hierarchical Risk Parity
- Features
  - [x] Structural breaks
  - [x] Entropy-based
  - Market microstructure features
  - 
... [TBD]

### Derivatives Toolbox
- Futures (for equity, index, currency, commodities) valuation and hedging strategies 
- Interest rates term structure derivation
- Bond pricing/duration/convexity calculation. Forward rate agreement (FRA) valuation
- Interest rate futures valuation and duration-based hedging strategies
- Valuation of vanilla interest rate and currency swaps
- OIS discounting calculator
- Valuation with credit risk factors: CVA, DVA
- Options hedging & speculation strategies: principal protected notes, spreads, option baskets
- [x] Binomial trees valuation methods
- [x] Black-Scholes-Merton model
- Employee stock option valuation
- Equity indices and currencies options valuation
- Futures option valuation
- Hedging with Greek letters (delta/theta/gamma/vega/rho hedging)
- Volatility smile tools
- Valuation methods using Monte Carlo simulation and finite difference
- Value at risk (VaR) models
- Volatilities & correlations estimation tools
- Credit risk estimation tools: using bond yield spread/CDS, using equity prices
- Credit derivatives valuation: CDS, CDS forwards & options, basket CDS, total return swaps, CDO,
CDS and CDO basket correlation, synthetic CDO
- Exotic options valuation: perpetual options, gap options, barrier options, volitility & variance swaps etc.
- Alternative valuation models: Stochastic volatility models, IVF model, path-dependent derivatives,
options on multiple correlated assets
- Convertible bond valuations
- Interest rate derivatives valuation: bond options, interest rate caps/floors, interest rate derivative hedging
- Convexity, timing & quanto adjustment tools
- Short rate models: equilibrium models (R&B, Vasicek, CIR),
no arbitrage models (ho-lee, hull-white, BDT, BK), bond options, volatility structure, interest trees
- HJM, LMM, multiple zero curves models
- Complex swaps valuation: compouding swaps, equity swaps, swaps with embedded options
- Commodity and energy derivatives valuations
- Real options valuation

... [TBD]

### Portfolio Management Toolbox
- [x] Portfolio optimization using Markowitz's mean-variance models
- Bonds portfolio management and hedging tools
- Market equilibrium models: capital asset pricing model, arbitrage pricing model
- Portfolio evaluation tools

--- [TBD]

---
## References
* Lopez de Prado, M. (2018). Advances in financial machine learning. John Wiley & Sons.
* Ané, T. and Geman, H. (2000), “Order ﬂow, transaction clock, and normality of asset returns”, Journal of Finance,Vol. 55 No. 5, pp. 2259-2284.
* Easley, David and López de Prado, Marcos and López de Prado, Marcos and O'Hara, Maureen, The Volume Clock: Insights into the High Frequency Paradigm (May 2012). The Journal of Portfolio Management, (Fall, 2012)
* Kolanovic, M. and R. Krishnamachari (2017): “Big data and AI strategies: Machine learning and alternative data approach to investing.” White paper, JP Morgan, Quantitative and Derivatives Strategy. May 18.
* Bailey, David H., and M. L´opez de Prado (2012): “Balanced baskets: A new approach to trading and hedging risks.” Journal of Investment Strategies (Risk Journals), Vol. 1, No. 4 (Fall), pp. 21–62.
* L´opez de Prado, M. and D. Leinweber (2012): “Advances in cointegration and subset correlation hedging methods.” Journal of Investment Strategies (Risk Journals), Vol. 1, No. 2 (Spring), pp. 67–115.
* Clark, P. K. (1973): “A subordinated stochastic process model with finite variance for speculative prices.” Econometrica, Vol. 41, pp. 135–155.
* L´opez de Prado, M. (2015): “The Future of Empirical Finance.” Journal of Portfolio Management, Vol. 41, No. 4, pp. 140–144.
* Jones, M.,M. Nielsen andM. Popiel (2015): “A fractionally cointegrated VAR analysis of economic voting and political support.” Canadian Journal of Economics, Vol. 47, No. 4, pp. 1078–1130.
* Cavaliere, G., M. Nielsen, and A. Taylor (2017): “Quasi-maximum likelihood estimation and bootstrap inference in fractional time series models with heteroskedasticity of unknown form.” Journal of Econometrics, Vol. 198, No. 1, pp. 165–188.
* Lo ́pez de Prado, M. and M. Foreman (2014): “A mixture of Gaussians approach to mathematical portfolio oversight: The EF3M algorithm.” Quantitative Finance, Vol. 14, No. 5, pp. 913–930.
* Bailey, D., J. Borwein, and M. L´opez de Prado (2016): “Stock portfolio design and backtest overfitting.” Journal of Investment Management, Vol. 15, No. 1, pp. 1–13. Available at https://ssrn.com/abstract=2739335.
* Bailey, D., J. Borwein, A. Salehipour, and M. L´opez de Prado (2017): “Evaluation and ranking of market forecasters.” Journal of Investment Management, forthcoming. Available at https://ssrn.com/abstract=2944853.
* Bailey, D. and M. Lo ́pez de Prado (2014): “The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality.” Journal of Portfolio Management, Vol. 40, No. 5, pp. 94–107. Available at https://ssrn.com/abstract=2460551.
* Bailey, D., J. Borwein, M. Lo ́pez de Prado, and J. Zhu (2014): “Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance.” Notices of the American Mathematical Society, Vol. 61, No. 5, pp. 458–471. Available at http://ssrn .com/abstract=2308659.
* Bailey, D. and M. Lo ́pez de Prado (2012): “The Sharpe ratio efficient frontier.” Journal of Risk, Vol. 15, No. 2, pp. 3–44.
* Barra (2013): MSCI BARRA Factor Indexes Methodology, 1st ed. MSCI Barra. Avail- able at https://www.msci.com/eqb/methodology/meth_docs/MSCI_Barra_Factor%20Indices_ Methodology_Nov13.pdf .
* CFA Institute (2010): “Global investment performance standards.” CFA Institute, Vol. 2010, No. 4, February. Available at https://www.gipsstandards.org/.
* Zhang, Y. and S. Rachev (2004): “Risk attribution and portfolio performance measurement— An overview.” Working paper, University of California, Santa Barbara. Available at http:// citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.318.7169.
* Calkin, N. and M. L´opez de Prado (2014): “The topology of macro financial flows: An application of stochastic flow diagrams.” Algorithmic Finance, Vol. 3, No. 1, pp. 43–85. Available at http://ssrn.com/abstract=2379319.
* Kolanovic, M.,A. Lau, T. Lee, and R. Krishnamachari (2017): “Cross asset portfolios of tradable risk premia indices. Hierarchical risk parity: Enhancing returns at target volatility.” White paper, Global Quantitative & Derivatives Strategy. J.P. Morgan, April 26.
* Homm, U. and J. Breitung (2012): “Testing for speculative bubbles in stock markets: A comparison of alternative methods.” Journal of Financial Econometrics, Vol. 10, No. 1, 198–231.
* Phillips, P.,Y.Wu, and J.Yu (2011): “Explosive behavior in the 1990s Nasdaq: When did exuberance escalate asset values?” International Economic Review, Vol. 52, pp. 201–226.
* Fiedor, Pawel (2014a): “Mutual information rate-based networks in financial markets.” Working paper, arXiv. Available at https://arxiv.org/abs/1401.2548.
* Fiedor, Pawel (2014b): “Information-theoretic approach to lead-lag effect on financial markets.” Working paper, arXiv. Available at https://arxiv.org/abs/1402.3820.
* Fiedor, Pawel (2014c): “Causal non-linear financial networks.” Working paper, arXiv. Available at https://arxiv.org/abs/1407.5020.
* Abad, D. and J. Yague (2012): “From PIN to VPIN.” The Spanish Review of Financial Economics, Vol. 10, No. 2, pp.74-83.
* Aitken, M. and A. Frino (1996): “The accuracy of the tick test: Evidence from the Australian Stock Exchange.” Journal of Banking and Finance, Vol. 20, pp. 1715–1729.
* Beckers, S. (1983): “Variances of security price returns based on high, low, and closing prices.” Journal of Business, Vol. 56, pp. 97–112.
* Carlin, B., M. Sousa Lobo, and S. Viswanathan (2005): “Episodic liquidity crises. Cooperative and predatory trading.” Journal of Finance, Vol. 42, No. 5 (October), pp. 2235–2274.
* Cheung,W., R. Chou, A. Lei (2015): “Exchange-traded barrier option and VPIN.” Journal of Futures Markets, Vol. 35, No. 6, pp. 561-581.
* Easley, D., R. Engle, M. O’Hara, and L. Wu (2008): “Time-varying arrival rates of informed and uninformed traders.” Journal of Financial Econometrics, Vol. 6, No. 2, pp. 171–207.
* Eisler, Z., J. Bouchaud, and J. Kockelkoren (2012): “The impact of order book events: Market orders, limit orders and cancellations.” Quantitative Finance, Vol. 12, No. 9, pp. 1395– 1419.
* O’Hara, M. (1995): Market Microstructure, 1st ed. Blackwell, Oxford.
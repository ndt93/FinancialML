# Financial Machine Learning Library

Applying machine learning and statistical methods to financial and markets data is deceptively difficult.
Both classical and modern ML models and testing methods in their original form, although having been applied
successfully in many fields, typically don't work on financial data.

This library implements a selected set of machine learning and statistical data structures and methods,
motivated by the latest financial academic researches and practical industry experiences, that are modified and designed
specially for handling financial and markets microstructure data. These tools are essential for either financial machine
learning researchers, investment portfolio managers or quantitative trading practitioners.

## Implementation Roadmap [WIP]

Listing of topics I've researched on or implemented as adhoc projects over the years, which are to be
packaged into this unified library. Most of these tools are currently only available through very
expensive commercial systems and software.
Implementations are prioritized by what I use more often for my projects.

This library only provides the tools necessary for academic research or investment strategies design and management and
not any specific investment/trading models or signals generation.

### Financial ML & Statistical Methods
- Standard & information driven bars sampling
- Securities basket ETF trick and roll over adjustment
- Triple barrier labeling
- Label weighting by uniqueness, returns attribution and time decay
- Fractionally differentiated time series
- Cross validation
  - Purged K-Fold
  - Embargo
- Features importance
  - Mean decrease impurity
  - Mean decrease accuracy
- Backtesting
  - Bet sizing
  - Walk-forward
  - Combinatorial purged CV
  - Synthetic data backtest
- Evaluation methods & metrics
- Strategy risks
- Assets allocation
  - Hierarchical Risk Parity
- Features
  - Structural breaks
  - Entropy-based
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
- Binomial trees valuation methods
- Black-Scholes-Merton model
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
- Portfolio optimization using with Markowitz's mean-variance models
- Market equilibrium models: capital asset pricing model, arbitrage pricing model

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
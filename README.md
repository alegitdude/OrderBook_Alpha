Finding Alpha in the Orderbook

Hypothesis: There exists some pattern or patterns in limit order book behavior that can be used to forecast large unidirectional mid-price movements.

Purpose: If any pattern or patterns are found to forecast a large unidirectional mid-price movement in an instrument, a trader could design a system to monetize the impending price movement.

Assumptions: If there is to be forecasting knowledge found in a limit order book, the most easily monetizable information that could be found for a trader would be movements that are large and in one direction, allowing for the most profit and least amount of risk during the trade. Mid-price movements that retrace or oscillate incur more risk and calculating their probability of success is much more difficult.

Abbreviations:

- Limit Order Book = LOB
- Large Unidirectional Mid-Price-Movement = LUMPM

Definitions:

- Large Unidirectional mid-price-movement:

  - Mid-Price Movement: The price exactly between the bid and ask price moving up or down
  - Large: size will vary between tests, but movements must be at lest 3 ticks
  - Unidirectional: In the same direction up or down

- Pattern: any measurable sequence or single number derived from any state in the orderbook that is found before more than one large mid-price move

Constraints: - Due to the near infinite amount of time horizons, and near infinite amount of ways to measure things, I will limit myself to only analyzing LOB behavior that precedes one singular LUMPM. It is possible that a LUMPM helps facilitate another perhaps larger LUMPM, however I will not take those into account in this exercise. Subsequent experiments could easily take into account multiple LUMPM

    - Because this is the beginning of my LOB exploration, I will only use one instrument to predict itself. Many combinations of related instruments could potentially be used to forecast price motion, however at the risk of the scope of this exercise growing too big, I will limit the analysis to one LOB predicting its own LUMPM.

    - I will only analyze mid-price motion that has the highest chance of monetization for an under resourced retail trader. So an LUMPM will be defined as at least 3 movements in the same direction without tracing back. By only analyzing LUMPM that meet this definition, I do not have to account for retracements, and most important of all, when attempting to monetize the movement I will not have to decide when the price action is over because any retracement will determine the end of a LUMPM.

    - I will only analyze LOB behavior up to the point of the first mid-price move of a LUMPM. The idea here is to start monetizing the move before it starts and the alternative is to also calculate LOB behavior during the LUMPM, however that would likely increase the scope of this experiment by orders of magnitude as then I would effectively have changed the goal of this experiment to using LOB behavior to predict if a price movement would continue, as opposed to if it will start. Also, because I have limited this experiment to analysis of moves without any back tracing, I do not have to concern myself with LOB behavior during an LUMPM. That will be likely tackled in subsequent experiments.

    - The first series of experiments will be limited to NASDAQ futures Market by Order data from DataBento for a few reasons: First, I am likely to find many examples of LUMPMs in the /NQ data due to its volatile price nature. Second, its data is also likely to be more accurate by nature of it being a very in-demand and looked over. Thirdly, MBO data from DataBento is very cost effective relative to other data providers retail traders have access to, and also comes in formats suitable data analysis (CSV, DBN, etc).

    - This analysis will be limited to regular trading hours (8:30 - 15:00 CST)

Methodology: - I will go through a day of order book messages and find all instances of LUMPM's that meet my criteria and document when they happen. Then analyze the period preceding the move using different windows and calculators to create features to put into a transformer based machine learning model to find patterns that might predict the LUMPM.

    - On first pass through the data, I will create parquet files of snapshots of every orderbook state that exists during regular trading hours. This will allow rapid iteration over the orderbook as new features are created and added, hopefully greatly reducing the time it takes to evaluate if there is any statistical significance to any feature.

    - The search for patterns will be left ongoing and modular, as new feature ideas are thought of and implemented.

Questions:

- What to do about LOB behavior DURING a LUMPM?
- How does an instrument's liquidity affect its LUMPM predictability?
- How would after hours trading analysis be different?

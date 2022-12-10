from enum import Enum, unique


class TickCol:
    TIMESTAMP = 'Timestamp'
    PRICE = 'Price'
    VOLUME = 'Volume'
    BAR_ID = 'Bar_ID'


class BarCol:
    TIMESTAMP = 'Timestamp'
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    VOLUME = 'Volume'
    VWAP = 'VWAP'
    TXN = 'Txn'

    RET_PRICES = 'rPrice'
    DIVIDEND = 'Dividend'


@unique
class BarUnit(Enum):
    TIME = 1
    TICK = 2
    VOLUME = 3
    DOLLARS = 4


class ContractCol:
    CONTRACT = 'Contract'


class EventCol:
    START_TIME = 'Start_Time'
    EXPIRY = 'Expiry'
    TARGET = 'Target'
    SIDE = 'Side'
    PT_TIME = 'PT_Time'  # Profit taking
    SL_TIME = 'SL_Time'  # Stop loss
    END_TIME = 'End_Time'

    RETURN = 'Return'
    LABEL = 'Label'


class PositionCol(EventCol):
    SIZE = 'Size'


class StatsCol:
    STAT = 'Stat'
    P_VAL = 'pVal'
    LAG = 'Lag'
    NUM_OBS = 'nObs'
    CONF_95 = '95% Conf'
    CORR = 'Corr'
    PDF = 'PDF'


class PortfolioCol:
    BEGIN_VALUE = 'BeginValue'
    CASHFLOW = 'Cashflow'


class QuoteCol:
    BID = 'Bid'
    ASK = 'Ask'
    SPREAD = 'Spread'
    START_TIME = 'Start_Time'


class OptionCol:

    STRIKE = 'Strike'
    PRICE = 'Price'


@unique
class OptionType(Enum):

    CALL = 'Call'
    PUT = 'Put'

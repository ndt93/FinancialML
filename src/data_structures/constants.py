from enum import Enum, unique


class TickCol:
    TIMESTAMP = 'Timestamp'
    PRICE = 'Price'
    VOLUME = 'Volume'


class BarCol:
    TIMESTAMP = 'Timestamp'
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    VOLUME = 'Volume'
    VWAP = 'VWAP'

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


class StatsCol:
    STAT = 'Stat'
    P_VAL = 'pVal'
    LAG = 'Lag'
    NUM_OBS = 'nObs'
    CONF_95 = '95% Conf'
    CORR = 'Corr'

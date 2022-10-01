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

@unique
class BarUnit(Enum):
    TIME = 1
    TICK = 2
    VOLUME = 3
    DOLLARS = 4


class ContractCol:
    CONTRACT = 'Contract'
